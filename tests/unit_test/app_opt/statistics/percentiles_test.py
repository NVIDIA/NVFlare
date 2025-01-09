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
from nvflare.app_common.statistics.numeric_stats import aggregate_centroids, compute_percentiles
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


class TestPercentiles:

    def test_percentile_metrics(self):
        stats_generator = MockDFStats(given_median=100)
        stats_generator.load_data()
        percentiles = stats_generator.percentiles("train", "Feature", percents=[50])
        result = percentiles.get(StatisticsConstants.STATS_PERCENTILES_KEY)
        print(f"{percentiles=}")
        assert result is not None
        assert result.get(50) == stats_generator.median

    def test_percentile_metrics_aggregation(self):
        stats_generators = [
            MockDFStats2(data_array=[0, 1, 2, 3, 4, 5]),
            MockDFStats(given_median=10),
            MockDFStats2(data_array=[100, 110, 120, 130, 140, 150]),
        ]
        global_digest = {}
        result = {}
        for g in stats_generators:  # each site/client
            g.load_data()
            local_percentiles = g.percentiles("train", "Feature", percents=[50])
            local_metrics = {"train": {"Feature": local_percentiles}}
            aggregate_centroids(local_metrics, global_digest)
            result = compute_percentiles(global_digest, {"Feature": [50]}, 2)

        expected_median = 10
        assert result["train"]["Feature"].get(50) == expected_median
