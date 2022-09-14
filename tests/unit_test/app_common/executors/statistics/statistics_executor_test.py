# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.app_common.abstract.statistics_spec import Feature, HistogramType, MetricConfig
from nvflare.app_common.executors.statistics.statistics_executor import StatisticsExecutor
from nvflare.app_common.executors.statistics.statistics_executor_exception import StatisticExecutorException
from tests.unit_test.app_common.executors.statistics.mock_df_stats_executor import MockDFStatistics


class MockStatsExecutor(StatisticsExecutor):
    def __init__(self, min_count, min_random, max_random, max_bins_percent):
        super(MockStatsExecutor, self).__init__(
            generator_id="",
            min_count=min_count,
            min_random=min_random,
            max_random=max_random,
            max_bins_percent=max_bins_percent,
        )

    def initialize(self, fl_ctx: FLContext):
        self.stats_generator = MockDFStatistics(data_path="")
        self.stats_generator.initialize({}, None)


class TestStatisticsExecutor:
    @classmethod
    def setup_class(cls):
        print("starting class: {} execution".format(cls.__name__))
        cls.stats_executor = MockStatsExecutor(min_count=1, min_random=0.1, max_random=0.3, max_bins_percent=0.7)
        cls.stats_executor.initialize(None)

    def test_get_numeric_features(self):
        features: Dict[str, List[Feature]] = self.stats_executor.get_numeric_features()
        assert len(features["train"]) == 1
        assert features["train"][0].feature_name == "Age"
        assert len(features["test"]) == 1

    def test_validate(self):
        # data has 6 rows.
        stats_executor = MockStatsExecutor(min_count=7, min_random=0.1, max_random=0.3, max_bins_percent=0.5)
        stats_executor.initialize(None)
        results = stats_executor.validate("site-1", stats_executor.get_numeric_features(), {}, None)
        for ds in results:
            for feature in results[ds]:
                assert results[ds][feature] == False

    def test_method_implementation(self):
        with pytest.raises(NotImplementedError):
            r = self.stats_executor.get_sum("train", "Age", MetricConfig("sum", {}), None, None)

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
        hist_config = {"*": {"bins": 5}, "Age": {"bins": 10}}
        inputs = Shareable()
        inputs["min"] = {"train": {"Age": 0}}
        inputs["max"] = {"train": {"Age": 50}}
        inputs["metric_config"] = hist_config
        metric_config = MetricConfig("histogram", hist_config)
        with pytest.raises(ValueError) as e:
            histogram = self.stats_executor.get_histogram("train", "Age", metric_config, inputs, None)
        assert (
            # local count * max_bins_percent = 6* 0.7 = 4.2 -> round to 4.
            str(e.value) == "number of bins: 10 needs to be smaller than item "
            "count: 4 for feature 'Age' in dataset 'train'"
        )

        hist_config = {"*": {"bins": 3}}
        inputs = Shareable()
        inputs["min"] = {"train": {"Age": 0}}
        inputs["max"] = {"train": {"Age": 50}}
        inputs["metric_config"] = hist_config
        metric_config = MetricConfig("histogram", hist_config)
        histogram = self.stats_executor.get_histogram("train", "Age", metric_config, inputs, None)
        assert histogram.hist_type == HistogramType.STANDARD
        assert len(histogram.bins) == 3

    def test_get_max_value(self):
        est_max_value = self.stats_executor._get_max_value(100)
        assert 100 < est_max_value <= 100 * (1 + self.stats_executor.max_random)

        est_max_value = self.stats_executor._get_max_value(0)
        assert est_max_value > 1e-5

        est_max_value = self.stats_executor._get_max_value(1e-4)
        assert est_max_value > 1e-4

        est_max_value = self.stats_executor._get_max_value(0.6 * 1e-3)
        assert 0.6 * 1e-3 < est_max_value

        est_max_value = self.stats_executor._get_max_value(-0.6 * 1e-3)
        assert est_max_value > -0.6 * 1e-3

        est_max_value = self.stats_executor._get_max_value(-1e-3)
        assert est_max_value >= -1e-3

        est_max_value = self.stats_executor._get_max_value(-100)
        assert est_max_value >= -100

    def test_get_min_value(self):
        est_min_value = self.stats_executor._get_min_value(100)
        assert (
            100
            > 100 * (1 - self.stats_executor.min_random)
            > est_min_value
            > 100 * (1 - self.stats_executor.max_random)
        )

        est_min_value = self.stats_executor._get_min_value(-100)
        assert (
            -100
            > -100 * (1 + self.stats_executor.min_random)
            > est_min_value
            > -100 * (1 + self.stats_executor.max_random)
        )

        est_min_value = self.stats_executor._get_min_value(0)
        assert est_min_value < 0

        est_min_value = self.stats_executor._get_min_value(1e-4)
        assert est_min_value < 1e-4

        est_min_value = self.stats_executor._get_min_value(-0.6 * 1e-3)
        assert -0.6 * 1e-3 > est_min_value
