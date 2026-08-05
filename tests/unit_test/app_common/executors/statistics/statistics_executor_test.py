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


from unittest.mock import patch

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import Bin, Feature, Histogram, HistogramType, StatisticConfig
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.executors.statistics.statistics_task_handler import StatisticsTaskHandler
from nvflare.app_common.statistics.numeric_stats import get_global_stats
from nvflare.app_common.workflows.statistics_controller import StatisticsController
from nvflare.fuel.utils import fobs
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
        features: dict[str, list[Feature]] = self.stats_executor.get_numeric_features()
        assert len(features["train"]) == 1
        assert features["train"][0].feature_name == "Age"
        assert len(features["test"]) == 1

    def test_execute_task_does_not_duplicate_configured_failure_count(self):
        inputs = Shareable()
        inputs[StC.STATISTICS_TASK_KEY] = StC.STATS_1st_STATISTICS
        inputs[StC.STATS_TARGET_STATISTICS] = fobs.dumps(
            [StatisticConfig(StC.STATS_COUNT, {}), StatisticConfig(StC.STATS_FAILURE_COUNT, {})]
        )

        with (
            patch.object(self.stats_executor.stats_generator, "count", return_value=10),
            patch.object(
                self.stats_executor.stats_generator,
                "failure_count",
                wraps=self.stats_executor.stats_generator.failure_count,
            ) as failure_count,
        ):
            self.stats_executor.execute_task(StC.FED_STATS_TASK, inputs, FLContext(), Signal())

        assert failure_count.call_count == 2

    def test_execute_task_adds_missing_failure_count(self):
        inputs = Shareable()
        inputs[StC.STATISTICS_TASK_KEY] = StC.STATS_1st_STATISTICS
        inputs[StC.STATS_TARGET_STATISTICS] = fobs.dumps([StatisticConfig(StC.STATS_COUNT, {})])

        with (
            patch.object(self.stats_executor.stats_generator, "count", return_value=10),
            patch.object(
                self.stats_executor.stats_generator,
                "failure_count",
                wraps=self.stats_executor.stats_generator.failure_count,
            ) as failure_count,
        ):
            self.stats_executor.execute_task(StC.FED_STATS_TASK, inputs, FLContext(), Signal())

        assert failure_count.call_count == 2

    def test_second_round_failure_count_includes_histogram_failures(self):
        histogram_config = {"*": {"bins": 1}}
        statistic_configs = {
            StC.STATS_COUNT: {},
            StC.STATS_FAILURE_COUNT: {},
            StC.STATS_HISTOGRAM: histogram_config,
        }
        target_statistics = StatisticsController._get_target_statistics(
            statistic_configs, StC.ordered_statistics[StC.STATS_2nd_STATISTICS]
        )
        inputs = Shareable()
        inputs[StC.STATISTICS_TASK_KEY] = StC.STATS_2nd_STATISTICS
        inputs[StC.STATS_TARGET_STATISTICS] = fobs.dumps(target_statistics)
        inputs[StC.STATS_MIN] = {"train": {"Age": 0}, "test": {"Age": 0}}
        inputs[StC.STATS_MAX] = {"train": {"Age": 10}, "test": {"Age": 10}}
        failures = {"train": 0, "test": 0}

        def get_failure_count(dataset_name, feature_name):
            return failures[dataset_name]

        def get_histogram(dataset_name, feature_name, num_of_bins, min_value, max_value):
            if dataset_name == "train":
                failures[dataset_name] += 1
            return Histogram(HistogramType.STANDARD, [Bin(min_value, max_value, 3)])

        with (
            patch.object(self.stats_executor.stats_generator, "count", return_value=4),
            patch.object(self.stats_executor.stats_generator, "failure_count", side_effect=get_failure_count),
            patch.object(self.stats_executor.stats_generator, "histogram", side_effect=get_histogram),
        ):
            result = self.stats_executor.execute_task(StC.FED_STATS_TASK, inputs, FLContext(), Signal())

        statistics = fobs.loads(result[StC.STATS_2nd_STATISTICS])
        assert statistics[StC.STATS_FAILURE_COUNT]["train"]["Age"] == 1

        client_statistics = {
            StC.STATS_COUNT: {
                "site-1": statistics[StC.STATS_COUNT],
                "site-2": {"train": {"Age": 4}},
            },
            StC.STATS_FAILURE_COUNT: {
                "site-1": statistics[StC.STATS_FAILURE_COUNT],
                "site-2": {"train": {"Age": 0}},
            },
            StC.STATS_HISTOGRAM: {
                "site-1": statistics[StC.STATS_HISTOGRAM],
                "site-2": {"train": {"Age": Histogram(HistogramType.STANDARD, [Bin(0, 10, 4)])}},
            },
        }
        global_statistics = get_global_stats({}, client_statistics, StC.STATS_1st_STATISTICS, statistic_configs)
        global_statistics = get_global_stats(
            global_statistics, client_statistics, StC.STATS_2nd_STATISTICS, statistic_configs
        )

        assert global_statistics[StC.STATS_COUNT]["train"]["Age"] == 8
        assert global_statistics[StC.STATS_FAILURE_COUNT]["train"]["Age"] == 1
        assert global_statistics[StC.STATS_HISTOGRAM]["train"]["Age"].bins == [Bin(0, 10, 7)]

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
