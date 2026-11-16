# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.min_count_cleanser import MinCountCleanser
from nvflare.app_common.statistics.numeric_stats import (
    exclude_withheld_features,
    get_global_stats,
    get_min_or_max_values,
)
from nvflare.app_opt.statistics.df.df_core_statistics import DFStatisticsCore

TEST_CASE_1 = [
    (
        {
            "site-1": {
                "train": {
                    "Age": 630,
                    "fnlwgt": 3673746,
                    "Education-Num": 177,
                    "Capital Gain": 16258,
                    "Capital Loss": 0,
                    "Hours per week": 631,
                },
            }
        },
        {
            "train": {
                "Age": 630,
                "fnlwgt": 3673746,
                "Education-Num": 177,
                "Capital Gain": 16258,
                "Capital Loss": 0,
                "Hours per week": 631,
            }
        },
    )
]


class TestNumericStats:
    @pytest.mark.parametrize("client_stats, expected_global_stats", TEST_CASE_1)
    def test_accumulate_metrics(self, client_stats, expected_global_stats):
        from nvflare.app_common.statistics.numeric_stats import accumulate_metrics

        global_stats = {}
        for client_name in client_stats:
            global_stats = accumulate_metrics(
                metrics=client_stats[client_name], global_metrics=global_stats, precision=4
            )

        assert global_stats.keys() == expected_global_stats.keys()
        assert global_stats == expected_global_stats

    def test_get_min_or_max_values(self):
        client_statistics = {
            "site-1": {"train": {"Age": 0}, "test": {"Age": 2}},
            "site-2": {"train": {"Age": 1}, "test": {"Age": 3}},
        }

        global_statistics: dict[str, dict[str, int]] = {}
        for client in client_statistics:
            statistics = client_statistics[client]
            print("get_min_or_max_values =", global_statistics)
            global_statistics = get_min_or_max_values(statistics, global_statistics, min)

        assert global_statistics == {"test": {"Age": 0}, "train": {"Age": 0}}

        global_statistics: dict[str, dict[str, int]] = {}
        for client in client_statistics:
            statistics = client_statistics[client]
            print("get_min_or_max_values =", global_statistics)
            global_statistics = get_min_or_max_values(statistics, global_statistics, max)

        assert global_statistics == {"test": {"Age": 3}, "train": {"Age": 3}}


def _client_first_round(values, min_count=10):
    """First-round statistics of one client after the privacy cleanser, and the features it reports as withheld."""
    count = len(values)
    total = float(sum(values))
    statistics = {
        StC.STATS_COUNT: {"train": {"x": count}},
        StC.STATS_FAILURE_COUNT: {"train": {"x": 0}},
        StC.STATS_SUM: {"train": {"x": total}},
        StC.STATS_MEAN: {"train": {"x": total / count}},
    }
    cleanser = MinCountCleanser(min_count=min_count)
    withheld = cleanser.withheld_features(statistics, "site")
    cleansed, _ = cleanser.apply(statistics, "site")
    return cleansed, {ds: set(features) for ds, features in withheld.items()}


def _aggregate(clients, use_marker=True):
    client_metrics = {
        m: {name: stats[m] for name, (stats, _) in clients.items()} for m in next(iter(clients.values()))[0]
    }
    withheld = {name: marker for name, (_, marker) in clients.items() if marker} if use_marker else None
    return get_global_stats({}, client_metrics, StC.STATS_1st_STATISTICS, {}, withheld_features=withheld)


class TestWithheldFeatures:
    def test_exclude_withheld_features(self):
        metrics = {"train": {"a": 5, "b": 1}, "test": {"a": 2}}

        assert exclude_withheld_features(metrics, {"train": {"b"}}) == {"train": {"a": 5}, "test": {"a": 2}}
        assert exclude_withheld_features(metrics, {"other": {"a"}}) == metrics
        assert exclude_withheld_features(metrics, None) is metrics

    def test_global_mean_ignores_site_withheld_by_min_count(self):
        clients = {
            "site-1": _client_first_round([11.0]),
            "site-2": _client_first_round([9.0] * 40),
            "site-3": _client_first_round([11.0] * 60),
        }
        assert clients["site-1"][1] == {"train": {"x"}}

        result = _aggregate(clients)

        assert result[StC.STATS_COUNT]["train"]["x"] == 100
        assert result[StC.STATS_SUM]["train"]["x"] == 1020.0
        assert result[StC.STATS_MEAN]["train"]["x"] == 10.2

    def test_second_round_variance_and_stddev_use_only_contributing_sites(self):
        values = {
            "site-1": [11.0],
            "site-2": list(9.0 + np.linspace(-1.0, 1.0, 40)),
            "site-3": list(11.0 + np.linspace(-2.0, 2.0, 60)),
        }
        first_round = {name: _client_first_round(v) for name, v in values.items()}
        global_stats = _aggregate(first_round)
        global_count = global_stats[StC.STATS_COUNT]["train"]["x"]
        global_mean = global_stats[StC.STATS_MEAN]["train"]["x"]
        assert global_count == 100

        # Second round as a client computes it from the global count and mean, then the privacy filter.
        cleanser = MinCountCleanser(min_count=10)
        second_round, withheld = {}, {}
        for name, v in values.items():
            core = DFStatisticsCore()
            core.data = {"train": pd.DataFrame({"x": v})}
            statistics = {
                StC.STATS_COUNT: {"train": {"x": len(v)}},
                StC.STATS_FAILURE_COUNT: {"train": {"x": 0}},
                StC.STATS_VAR: {"train": {"x": core.variance_with_mean("train", "x", global_mean, global_count)}},
                StC.STATS_STDDEV: {"train": {"x": core.stddev("train", "x")}},
            }
            marker = cleanser.withheld_features(statistics, name)
            if marker:
                withheld[name] = {ds: set(features) for ds, features in marker.items()}
            second_round[name], _ = cleanser.apply(statistics, name)
        assert withheld == {"site-1": {"train": {"x"}}}
        client_metrics = {m: {name: stats[m] for name, stats in second_round.items()} for m in second_round["site-1"]}

        result = get_global_stats(
            global_stats, client_metrics, StC.STATS_2nd_STATISTICS, {}, withheld_features=withheld
        )

        contributing = np.array(values["site-2"] + values["site-3"])
        assert result[StC.STATS_VAR]["train"]["x"] == pytest.approx(contributing.var(ddof=1), abs=1e-3)
        assert result[StC.STATS_STDDEV]["train"]["x"] == pytest.approx(contributing.std(ddof=1), abs=1e-3)

    def test_failure_count_ignores_withheld_site_too(self):
        clients = {"site-1": _client_first_round([11.0]), "site-2": _client_first_round([9.0] * 40)}
        client_metrics = {m: {name: stats[m] for name, (stats, _) in clients.items()} for m in clients["site-1"][0]}
        client_metrics[StC.STATS_FAILURE_COUNT]["site-1"]["train"]["x"] = 1
        client_metrics[StC.STATS_FAILURE_COUNT]["site-2"]["train"]["x"] = 2

        result = get_global_stats(
            {}, client_metrics, StC.STATS_2nd_STATISTICS, {}, withheld_features={"site-1": {"train": {"x"}}}
        )

        assert result[StC.STATS_FAILURE_COUNT]["train"]["x"] == 2

    def test_only_reported_features_are_excluded(self):
        # Excluding is driven by the reported marker, not by a missing sum.
        site_1, _ = _client_first_round([11.0])
        site_2, _ = _client_first_round([9.0] * 40)

        result = _aggregate({"site-1": (site_1, set()), "site-2": (site_2, set())}, use_marker=True)

        assert result[StC.STATS_COUNT]["train"]["x"] == 41

    def test_count_only_job_also_excludes_the_withheld_site(self):
        clients = {"site-1": _client_first_round([11.0]), "site-2": _client_first_round([9.0] * 40)}
        count_only = {StC.STATS_COUNT: {name: stats[StC.STATS_COUNT] for name, (stats, _) in clients.items()}}

        result = get_global_stats(
            {}, count_only, StC.STATS_1st_STATISTICS, {}, withheld_features={"site-1": {"train": {"x"}}}
        )

        assert result[StC.STATS_COUNT]["train"]["x"] == 40

    def test_global_mean_when_no_site_is_withheld(self):
        clients = {"site-1": _client_first_round([11.0] * 10), "site-2": _client_first_round([9.0] * 40)}

        result = _aggregate(clients)

        assert result[StC.STATS_COUNT]["train"]["x"] == 50
        assert result[StC.STATS_MEAN]["train"]["x"] == 9.4
