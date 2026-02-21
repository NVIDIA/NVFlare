# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import json
import sys

import numpy as np
import pandas as pd
import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import StatisticsConstants
from nvflare.app_opt.statistics.df.df_core_statistics import DFStatisticsCore
from nvflare.app_opt.statistics.quantile_stats import compute_quantiles, merge_quantiles

try:
    from fastdigest import TDigest

    TDIGEST_AVAILABLE = True
except ImportError:
    TDIGEST_AVAILABLE = False

TDIGEST_AVAILABLE = TDIGEST_AVAILABLE and sys.platform != "darwin"


def _make_digest(data, max_centroids=None):
    values = data.tolist() if hasattr(data, "tolist") else list(data)

    try:
        return TDigest(values)
    except TypeError:
        pass

    if max_centroids is None:
        max_centroids = max(100, int(np.sqrt(len(values))) if len(values) > 0 else 100)

    try:
        digest = TDigest(max_centroids=max_centroids)
    except TypeError:
        digest = TDigest(max_centroids)

    if hasattr(digest, "batch_update"):
        digest.batch_update(values)
    else:
        try:
            digest.update(values)
        except TypeError:
            for v in values:
                digest.update(v)

    return digest


def _merge_digest(left, right):
    merged = left.merge(right)
    return merged if merged is not None else left


def _compress_digest(digest, max_centroids):
    try:
        digest.compress(max_centroids)
    except TypeError:
        digest.compress()


def _from_dict_digest(digest_dict):
    try:
        return TDigest.from_dict(digest_dict)
    except TypeError:
        return _make_digest([]).from_dict(digest_dict)


class MockDFStats(DFStatisticsCore):
    def __init__(self, given_median: int):
        super().__init__()
        self.median = given_median
        self.data = {"train": None}

    def initialize(self, fl_ctx: FLContext):
        self.load_data()

    def load_data(self):
        data = np.concatenate(
            (np.arange(0, self.median), [self.median], np.arange(self.median + 1, self.median * 2 + 1))
        )

        # Shuffle the data to make it unordered
        np.random.shuffle(data)

        # Create the DataFrame
        df = pd.DataFrame(data, columns=["Feature"])
        self.data = {"train": df}


class MockDFStats2(DFStatisticsCore):
    def __init__(self, data_array: list[int]):
        super().__init__()
        self.raw_data = data_array
        self.data = {"train": None}

    def initialize(self, fl_ctx: FLContext):
        self.load_data()

    def load_data(self):
        # Create the DataFrame
        df = pd.DataFrame(self.raw_data, columns=["Feature"])
        self.data = {"train": df}


class TestQuantile:
    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest1(self):
        # Small dataset
        data = [1, 2, 3, 4, 5]
        fd = _make_digest(data)

        # Insert values
        np_data = pd.Series(data)

        assert fd.quantile(0.25) == np_data.quantile(0.25)
        assert fd.quantile(0.5) == np_data.quantile(0.5)
        assert fd.quantile(0.75) == np_data.quantile(0.75)

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest2(self):
        # Small dataset
        data = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        fd = _make_digest(data)
        # Insert values
        np_data = pd.Series(data)

        assert fd.quantile(0.25) == np_data.quantile(0.25)
        assert fd.quantile(0.5) == np_data.quantile(0.5)
        assert fd.quantile(0.75) == np_data.quantile(0.75)

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest3(self):
        # Small dataset
        data = [-50.0, -40.4, -30.3, -20.3, -10.1, 0, 1.1, 2.2, 3.3, 4.4, 5.5]
        fd = _make_digest(data)

        np_data = pd.Series(data)

        assert round(fd.quantile(0.25), 2) == np_data.quantile(0.25)
        assert round(fd.quantile(0.5), 2) == np_data.quantile(0.5)
        assert round(fd.quantile(0.75), 2) == np_data.quantile(0.75)

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest4(self):
        # Small dataset
        data = [-5, -4, -3, -2, -1, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        fd = _make_digest(data)

        np_data = pd.Series(data)

        assert round(fd.quantile(0.25), 2) == np_data.quantile(0.25)
        assert round(fd.quantile(0.5), 2) == np_data.quantile(0.5)
        assert round(fd.quantile(0.75), 2) == np_data.quantile(0.75)

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest5(self):
        # Small dataset
        data1 = [x for x in range(-5, 0)]
        data2 = [x * 1.0 for x in range(0, 6)]
        data = data1 + data2
        fd = _make_digest(data)

        assert fd.quantile(0.5) == 0
        assert fd.quantile(0.1) == -4
        assert fd.quantile(0.9) == 4

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest6(self):
        # Small dataset
        data1 = [x for x in range(-10000, 0)]
        data2 = [x * 1.0 for x in range(0, 10000 + 1)]
        fd = _make_digest(data1)
        merged_fd = _merge_digest(fd, _make_digest(data2))

        fdx = _make_digest(data1 + data2)

        np_data = pd.Series(data1 + data2)

        assert fdx.quantile(0.5) == np_data.quantile(0.5)
        assert merged_fd.quantile(0.5) == np_data.quantile(0.5)

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest7(self):
        median = 10
        data = np.concatenate((np.arange(0, median), [median], np.arange(median + 1, median * 2 + 1)))
        # Shuffle the data to make it unordered
        np.random.shuffle(data)

        fd = _make_digest(data)

        v = fd.quantile(0.5)

        print(sorted(data), v, median)

        assert v == median

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest8(self):

        median = 10
        data = np.concatenate((np.arange(0, median), [median], np.arange(median + 1, median * 2 + 1)))
        # Shuffle the data to make it unordered
        np.random.shuffle(data)
        data1 = [0, 1, 2, 3, 4, 5]
        data2 = [100, 110, 120, 130, 140, 150]

        fd1 = _make_digest(data1)
        fd2 = _make_digest(data2)

        fd = _make_digest(data)

        fd = _merge_digest(fd, fd1)
        fd = _merge_digest(fd, fd2)

        v = fd.quantile(0.5)

        print(sorted(data), v, median)

        assert v == median

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest_merge_serde(self):

        median = 10
        data = np.concatenate((np.arange(0, median), [median], np.arange(median + 1, median * 2 + 1)))
        # Shuffle the data to make it unordered
        np.random.shuffle(data)
        data1 = [0, 1, 2, 3, 4, 5]
        data2 = [100, 110, 120, 130, 140, 150]

        fd1 = _make_digest(data1)
        fd2 = _make_digest(data2)

        fd = _make_digest(data)

        fd = _merge_digest(fd, _from_dict_digest(fd1.to_dict()))
        fd = _merge_digest(fd, _from_dict_digest(fd2.to_dict()))

        v = fd.quantile(0.5)

        print(sorted(data), v, median)

        assert v == median

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_tdigest_compress(self):

        digest = _make_digest(range(101), max_centroids=10)
        print(f"Before: {len(digest)} centroids")

        before_median = digest.quantile(0.5)
        before_25 = digest.quantile(0.25)
        before_75 = digest.quantile(0.75)

        _compress_digest(digest, 10)  # compress to 10 (or fewer) centroids

        print(f" After: {len(digest)} centroids")

        print(json.dumps(digest.to_dict(), indent=2))

        after_median = digest.quantile(0.5)
        after_25 = digest.quantile(0.25)
        after_75 = digest.quantile(0.75)

        assert before_median == after_median
        assert before_25 == after_25
        assert before_75 == after_75
        assert len(digest) <= 10

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest is not installed")
    def test_percentile_metrics(self):
        stats_generator = MockDFStats(given_median=100)
        stats_generator.load_data()
        percentiles = stats_generator.quantiles("train", "Feature", percents=[0.5])
        result = percentiles.get(StatisticsConstants.STATS_QUANTILE)
        digest_dict = percentiles.get(StatisticsConstants.STATS_DIGEST_COORD)
        assert digest_dict is not None
        assert result is not None
        print(sorted(stats_generator.data["train"]["Feature"]))

        assert result.get(0.5) == stats_generator.median

    @pytest.mark.skipif(not TDIGEST_AVAILABLE, reason="fastdigest package not installed")
    def test_percentile_metrics_aggregation(self):
        stats_generators = [
            MockDFStats2(data_array=[0, 1, 2, 3, 4, 5, 6]),
            MockDFStats(given_median=10),
            MockDFStats2(data_array=[100, 110, 120, 130, 140, 150, 160]),
        ]
        global_digest = {}
        for g in stats_generators:  # each site/client
            g.load_data()
            local_quantiles = g.quantiles("train", "Feature", percents=[0.5])
            local_metrics = {"train": {"Feature": local_quantiles}}
            global_digest = merge_quantiles(local_metrics, global_digest)

        result = compute_quantiles(global_digest, {"Feature": [0.5]}, 2)

        expected_median = 10
        assert result["train"]["Feature"].get(0.5) == expected_median
