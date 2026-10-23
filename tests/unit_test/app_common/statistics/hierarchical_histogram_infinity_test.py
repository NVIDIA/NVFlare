# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import pytest

from nvflare.app_common.abstract.statistics_spec import Bin, Histogram, HistogramType
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.hierarchical_numeric_stats import get_global_stats
from nvflare.app_opt.statistics.df.df_core_statistics import DFStatisticsCore


@pytest.mark.parametrize("num_bins", [1, 3])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "client_values",
    [
        ([-np.inf, 0.0], [-np.inf, 0.0, np.inf]),
        ([-1.0, 0.0, 1.0], [-np.inf, 0.0, np.inf]),
        ([0.0, np.inf], [-np.inf, 0.0]),
    ],
)
def test_hierarchical_histograms_keep_mixed_infinity_samples(num_bins, reverse, client_values):
    """Root, organization and local bins retain every client's finite/infinite samples."""
    clients = list(reversed(client_values)) if reverse else client_values
    metrics = {}
    expected_counts = np.zeros(num_bins, dtype=np.int64)
    local_histograms = {}
    for i, values in enumerate(clients, 1):
        statistics = DFStatisticsCore()
        statistics.data = {"train": pd.DataFrame({"value": values})}
        histogram = statistics.histogram("train", "value", num_bins, -1.0, 1.0)
        name = f"Site-{i}"
        metrics[name] = {"train": {"value": histogram}}
        local_histograms[name] = histogram
        finite = np.asarray(values)[np.isfinite(values)]
        counts = np.histogram(finite, bins=num_bins, range=(-1.0, 1.0))[0]
        counts[0] += np.isneginf(values).sum()
        counts[-1] += np.isposinf(values).sum()
        expected_counts += counts

    hierarchy = {"Orgs": [{"Name": "Org-1", "Sites": list(metrics)}]}
    result = get_global_stats({}, {StC.STATS_HISTOGRAM: metrics}, StC.STATS_2nd_STATISTICS, hierarchy)
    organization = result["Orgs"][0]
    for level in (result, organization):
        bins = level[StC.GLOBAL][StC.STATS_HISTOGRAM]["train"]["value"].bins
        assert len(bins) == num_bins
        np.testing.assert_array_equal([b.sample_count for b in bins], expected_counts)
        assert sum(b.sample_count for b in bins) == sum(len(v) for v in clients)
        if any(np.isneginf(values).any() for values in clients):
            assert bins[0].low_value == -np.inf
        if any(np.isposinf(values).any() for values in clients):
            assert bins[-1].high_value == np.inf

    for site in organization["Sites"]:
        name = site[StC.NAME]
        assert site[StC.LOCAL][StC.STATS_HISTOGRAM]["train"]["value"] == local_histograms[name]


def test_hierarchical_histograms_preserve_exact_finite_edges():
    """Hierarchical mode retains original boundaries rather than rounding to four places."""
    hist = Histogram(HistogramType.STANDARD, [Bin(0.123456789, 0.654321987, 3)])
    metrics = {name: {"train": {"value": hist}} for name in ("Site-1", "Site-2")}
    hierarchy = {"Orgs": [{"Name": "Org-1", "Sites": list(metrics)}]}
    result = get_global_stats({}, {StC.STATS_HISTOGRAM: metrics}, StC.STATS_2nd_STATISTICS, hierarchy)
    assert result[StC.GLOBAL][StC.STATS_HISTOGRAM]["train"]["value"].bins == [Bin(0.123456789, 0.654321987, 6)]


@pytest.mark.parametrize("bounds", [(-0.5, 1.0), (-1.0, 0.5), (-1.0, 1.5)])
def test_hierarchical_histograms_reject_unrelated_grids(bounds):
    """Do not silently discard clients with a genuinely different finite grid."""
    hist_a = Histogram(HistogramType.STANDARD, [Bin(-1.0, 1.0, 2)])
    hist_b = Histogram(HistogramType.STANDARD, [Bin(*bounds, 3)])
    metrics = {
        "Site-1": {"train": {"value": hist_a}},
        "Site-2": {"train": {"value": hist_b}},
    }
    hierarchy = {"Orgs": [{"Name": "Org-1", "Sites": list(metrics)}]}
    with pytest.raises(ValueError, match="incompatible histogram bin grids"):
        get_global_stats({}, {StC.STATS_HISTOGRAM: metrics}, StC.STATS_2nd_STATISTICS, hierarchy)


def test_hierarchical_histograms_keep_separate_organization_totals():
    """A mixed-infinity organization merge must not absorb or lose other sites."""
    values_by_site = {
        "Site-1": [-np.inf, 0.0],
        "Site-2": [-np.inf, 0.0, np.inf],
        "Site-3": [0.0, np.inf],
    }
    metrics = {}
    for name, values in values_by_site.items():
        statistics = DFStatisticsCore()
        statistics.data = {"train": pd.DataFrame({"value": values})}
        metrics[name] = {"train": {"value": statistics.histogram("train", "value", 1, -1.0, 1.0)}}

    hierarchy = {
        "Orgs": [
            {"Name": "Org-1", "Sites": ["Site-1", "Site-2"]},
            {"Name": "Org-2", "Sites": ["Site-3"]},
        ]
    }
    result = get_global_stats({}, {StC.STATS_HISTOGRAM: metrics}, StC.STATS_2nd_STATISTICS, hierarchy)

    def hist(node, level):
        return node[level][StC.STATS_HISTOGRAM]["train"]["value"]

    assert sum(bucket.sample_count for bucket in hist(result, StC.GLOBAL).bins) == 7
    assert sum(bucket.sample_count for bucket in hist(result["Orgs"][0], StC.GLOBAL).bins) == 5
    assert sum(bucket.sample_count for bucket in hist(result["Orgs"][1], StC.GLOBAL).bins) == 2
    assert hist(result["Orgs"][0]["Sites"][0], StC.LOCAL) == metrics["Site-1"]["train"]["value"]
    assert hist(result["Orgs"][1]["Sites"][0], StC.LOCAL) == metrics["Site-3"]["train"]["value"]
