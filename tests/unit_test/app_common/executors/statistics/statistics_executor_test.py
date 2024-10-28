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


from typing import Dict, List

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.statistics_spec import Feature, HistogramType, StatisticConfig
from nvflare.app_common.executors.statistics.statistics_task_handler import StatisticsTaskHandler
from tests.unit_test.app_common.executors.statistics.mock_df_stats_executor import MockDFStatistics


class MockStatsExecutor(StatisticsTaskHandler):
    def __init__(self):
        super().__init__(generator_id="")
        self.stats_generator = None

    def initialize(self, fl_ctx: FLContext):
        self.stats_generator = MockDFStatistics(data_path="")
        self.stats_generator.initialize(None)


class TestStatisticsExecutor:
    @classmethod
    def setup_class(cls):
        print("starting class: {} execution".format(cls.__name__))
        cls.stats_executor = MockStatsExecutor()
        cls.stats_executor.initialize(None)

    def test_get_numeric_features(self):
        features: Dict[str, List[Feature]] = self.stats_executor.get_numeric_features()
        assert len(features["train"]) == 1
        assert features["train"][0].feature_name == "Age"
        assert len(features["test"]) == 1

    def test_method_implementation(self):
        with pytest.raises(NotImplementedError):
            r = self.stats_executor.get_sum("train", "Age", StatisticConfig("sum", {}), None, None)

    def test_histogram_num_of_bins(self):
        hist_config = {"Age": {"bins": 5}}
        print(hist_config["Age"]["bins"])
        bins = self.stats_executor.get_number_of_bins("Age", hist_config)
        assert bins == 5
        hist_config = {"*": {"bins": 5}}
        bins = self.stats_executor.get_number_of_bins("Age", hist_config)
        assert bins == 5
        hist_config = {"Age": {"bins": 6}, "*": {"bins": 10}}
        bins = self.stats_executor.get_number_of_bins("Edu", hist_config)
        assert bins == 10
        bins = self.stats_executor.get_number_of_bins("Age", hist_config)
        assert bins == 6

        with pytest.raises(Exception) as e:
            hist_config = {}
            bins = self.stats_executor.get_number_of_bins("Age", hist_config)
        assert str(e.value) == "feature name = 'Age': missing required 'bins' config in histogram config = {}"

        with pytest.raises(Exception) as e:
            hist_config = {"Age": {"bin": 5}}
            bins = self.stats_executor.get_number_of_bins("Age", hist_config)
        assert (
            str(e.value)
            == "feature name = 'Age': missing required 'bins' config in histogram config = {'Age': {'bin': 5}}"
        )

    def test_histogram_bin_range(self):
        hist_config = {"Age": {"bins": 5, "range": [0, 120]}}
        bin_range = self.stats_executor.get_bin_range("Age", 0, 100, hist_config)
        assert bin_range == [0, 120]

        hist_config = {"*": {"bins": 5, "range": [0, 120]}}
        bin_range = self.stats_executor.get_bin_range("Age", 0, 50, hist_config)
        assert bin_range == [0, 120]

        hist_config = {"*": {"bins": 5}}
        bin_range = self.stats_executor.get_bin_range("Age", 0, 50, hist_config)
        assert bin_range == [0, 50]
        hist_config = {"*": {"bins": 5}, "Age": {"bins": 10}}
        bin_range = self.stats_executor.get_bin_range("Age", 0, 50, hist_config)
        assert bin_range == [0, 50]

    def test_histogram(self):
        hist_config = {"*": {"bins": 3}}
        inputs = Shareable()
        inputs["min"] = {"train": {"Age": 0}}
        inputs["max"] = {"train": {"Age": 50}}
        inputs["statistic_config"] = hist_config

        statistic_config = StatisticConfig("histogram", hist_config)
        histogram = self.stats_executor.get_histogram("train", "Age", statistic_config, inputs, None)
        assert histogram.hist_type == HistogramType.STANDARD
        assert len(histogram.bins) == 3
