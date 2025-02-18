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
from typing import List

import numpy as np
import pandas as pd

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import StatisticsConstants
from nvflare.app_common.statistics.numeric_stats import compute_quantiles, merge_quantiles
from nvflare.app_common.statistics.q_digest import QDigest
from nvflare.app_opt.statistics.df.df_core_statistics import DFStatisticsCore


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
    def __init__(self, data_array: List[int]):
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

    def test_q_digest1(self):
        # Small dataset
        data = [1, 2, 3, 4, 5]
        qd = QDigest()
        # Insert values
        for val in data:
            qd.insert(val)

        assert qd.quantile(0.25) == 2
        assert qd.quantile(0.5) == 3
        assert qd.quantile(0.75) == 4

    def test_q_digest2(self):
        # Small dataset
        data = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        qd = QDigest()
        # Insert values
        for val in data:
            qd.insert(val)

        assert qd.quantile(0.25) == -2
        assert qd.quantile(0.5) == 0
        assert qd.quantile(0.75) == 3

    def test_q_digest3(self):
        # Small dataset
        data = [-40.4, -30.3, -20.3, -10.1, 0, 1.1, 2.2, 3.3, 4.4, 5.5]
        qd = QDigest()
        # Insert values
        for val in data:
            qd.insert(val)

        assert round(qd.quantile(0.25), 2) == -20.3
        assert qd.quantile(0.5) == 0
        assert round(qd.quantile(0.75), 2) == 3.3

    def test_q_digest4(self):
        # Small dataset
        data = [-5, -4, -3, -2, -1, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        qd = QDigest()
        # Insert values
        for val in data:
            qd.insert(val)

        assert round(qd.quantile(0.25), 2) == -3
        assert qd.quantile(0.5) == 0
        assert round(qd.quantile(0.75), 2) == 3.0

    def test_q_digest5(self):
        # Small dataset
        data1 = [x for x in range(-5, 0)]
        data2 = [x * 1.0 for x in range(0, 6)]
        qd = QDigest()

        print("add value to Q digest tree")

        for val in data1:
            qd.insert(val)
        for val in data2:
            qd.insert(val)

        print(data1, data2)

        print("calculate the quantiles")

        assert qd.quantile(0.5) == 0
        assert qd.quantile(0.1) == -4
        assert qd.quantile(0.9) == 4

    def test_q_digest6(self):
        # Small dataset
        data1 = [x for x in range(-10000, 0)]
        data2 = [x * 1.0 for x in range(0, 10000 + 1)]
        qd = QDigest()

        print("add value to Q digest tree")

        for val in data1:
            qd.insert(val)
        for val in data2:
            qd.insert(val)

        print(data1, data2)

        print("calculate the quantiles")

        assert qd.quantile(0.5) == 0

    def test_q_digest7(self):
        median = 10
        data = np.concatenate((np.arange(0, median), [median], np.arange(median + 1, median * 2 + 1)))
        # Shuffle the data to make it unordered
        np.random.shuffle(data)

        q_digest = QDigest()
        for value in data:
            q_digest.insert(value)

        v = q_digest.quantile(0.5)

        print(sorted(data), v, median)

        assert v == median

    def test_q_digest8(self):

        median = 10
        data = np.concatenate((np.arange(0, median), [median], np.arange(median + 1, median * 2 + 1)))
        # Shuffle the data to make it unordered
        np.random.shuffle(data)
        data1 = [0, 1, 2, 3, 4, 5]
        data2 = [100, 110, 120, 130, 140, 150]

        q_digest1 = QDigest()
        for value in data1:
            q_digest1.insert(value)

        q_digest = QDigest()
        for value in data:
            q_digest.insert(value)

        q_digest2 = QDigest()
        for value in data2:
            q_digest2.insert(value)

        q_digest.merge(q_digest1)
        q_digest.merge(q_digest2)

        for value in data:
            q_digest.insert(value)

        v = q_digest.quantile(0.5)

        print(sorted(data), v, median)

        assert v == median

    def test_q_digest_merge_serde(self):

        median = 10
        data = np.concatenate((np.arange(0, median), [median], np.arange(median + 1, median * 2 + 1)))
        # Shuffle the data to make it unordered
        np.random.shuffle(data)
        data1 = [0, 1, 2, 3, 4, 5]
        data2 = [100, 110, 120, 130, 140, 150]

        q_digest1 = QDigest()
        for value in data1:
            q_digest1.insert(value)

        q_digest = QDigest()
        for value in data:
            q_digest.insert(value)

        q_digest2 = QDigest()
        for value in data2:
            q_digest2.insert(value)

        q_digest.merge(QDigest.deserialize(q_digest1.serialize()))
        q_digest.merge(QDigest.deserialize(q_digest2.serialize()))

        for value in data:
            q_digest.insert(value)

        v = q_digest.quantile(0.5)

        print(sorted(data), v, median)

        assert v == median

    def test_percentile_metrics(self):
        stats_generator = MockDFStats(given_median=100)
        stats_generator.load_data()
        percentiles = stats_generator.quantiles("train", "Feature", percents=[0.5])
        result = percentiles.get(StatisticsConstants.STATS_QUANTILE)
        q_digest = percentiles.get(StatisticsConstants.STATS_Q_DIGEST)
        assert q_digest is not None
        assert result is not None
        print(sorted(stats_generator.data["train"]["Feature"]))

        assert result.get(0.5) == stats_generator.median

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
