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
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.statistics_spec import MetricConfig
from nvflare.app_common.app_constant import StatisticsConstants as SC

from .mock_global_statistics import MockGlobalStatistics


class TestGlobalStatistics:
    @classmethod
    def setup_class(cls):
        print("starting class: {} execution".format(cls.__name__))
        metric_configs = {
            "count": {},
            "mean": {},
            "sum": {},
            "stddev": {},
            "histogram": {"*": {"bins": 10}, "Age": {"bins": 5, "range": [0, 120]}},
        }
        cls.stats_controller = MockGlobalStatistics(metric_configs=metric_configs, writer_id="")

    @classmethod
    def teardown_class(cls):
        print("starting class: {} execution".format(cls.__name__))

    def setup_method(self, method):
        print("starting execution of tc: {}".format(method.__name__))

    def teardown_method(self, method):
        pass

    def test_target_metrics(self):

        target_metrics: List[MetricConfig] = self.stats_controller._get_target_metrics(
            SC.ordered_metrics[SC.STATS_1st_METRICS]
        )

        for mc in target_metrics:
            assert mc.name in SC.ordered_metrics[SC.STATS_1st_METRICS]
            if mc.name not in [SC.STATS_MAX, SC.STATS_MIN]:
                assert mc.config == {}
            else:
                assert mc.config == {"*": {"bins": 10}, "Age": {"bins": 5, "range": [0, 120]}}

        target_metrics: List[MetricConfig] = self.stats_controller._get_target_metrics(
            SC.ordered_metrics[SC.STATS_2nd_METRICS]
        )

        for mc in target_metrics:
            assert mc.name in SC.ordered_metrics[SC.STATS_2nd_METRICS]
            if mc.name not in [SC.STATS_HISTOGRAM]:
                assert mc.config == {}
            else:
                assert mc.config == {"*": {"bins": 10}, "Age": {"bins": 5, "range": [0, 120]}}

    def test_prepare_input(self):
        xs = self.stats_controller._prepare_inputs(SC.STATS_1st_METRICS, None)
        assert xs[SC.METRIC_TASK_KEY] == SC.STATS_1st_METRICS
        assert xs[SC.STATS_TARGET_METRICS].sort() == SC.ordered_metrics[SC.STATS_1st_METRICS].sort()

        # simulate aggregation and populate the global results
        self.stats_controller.global_metrics[SC.STATS_COUNT] = {"train": {"Age": 100}, "test": {"Age": 10}}
        self.stats_controller.global_metrics[SC.STATS_MEAN] = {"train": {"Age": 25}, "test": {"Age": 30}}

        self.stats_controller.global_metrics[SC.STATS_MAX] = {"train": {"Age": 120}, "test": {"Age": 120}}
        self.stats_controller.global_metrics[SC.STATS_MIN] = {"train": {"Age": 0}, "test": {"Age": 0}}

        assert self.stats_controller.global_metrics != {}

        xs = self.stats_controller._prepare_inputs(SC.STATS_2nd_METRICS, None)
        assert xs[SC.METRIC_TASK_KEY] == SC.STATS_2nd_METRICS
        assert xs[SC.STATS_TARGET_METRICS].sort() == SC.ordered_metrics[SC.STATS_2nd_METRICS].sort()
