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

from nvflare.app_common.abstract.statistics_spec import StatisticConfig
from nvflare.app_common.app_constant import StatisticsConstants as SC
from nvflare.app_common.workflows.statistics_controller import StatisticsController
from nvflare.fuel.utils import fobs

from .mock_statistics_controller import MockStatisticsController


class TestStatisticsController:
    @classmethod
    def setup_class(cls):
        print("starting class: {} execution".format(cls.__name__))
        statistic_configs = {
            "count": {},
            "mean": {},
            "sum": {},
            "stddev": {},
            "histogram": {"*": {"bins": 10}, "Age": {"bins": 5, "range": [0, 120]}},
        }
        cls.stats_controller = MockStatisticsController(statistic_configs=statistic_configs, writer_id="")

    def test_target_statistics(self):

        target_statistics: List[StatisticConfig] = StatisticsController._get_target_statistics(
            self.stats_controller.statistic_configs, SC.ordered_statistics[SC.STATS_1st_STATISTICS]
        )

        for mc in target_statistics:
            assert mc.name in SC.ordered_statistics[SC.STATS_1st_STATISTICS]
            if mc.name not in [SC.STATS_MAX, SC.STATS_MIN]:
                assert mc.config == {}
            else:
                assert mc.config == {"*": {"bins": 10}, "Age": {"bins": 5, "range": [0, 120]}}

        target_statistics: List[StatisticConfig] = StatisticsController._get_target_statistics(
            self.stats_controller.statistic_configs, SC.ordered_statistics[SC.STATS_2nd_STATISTICS]
        )

        for mc in target_statistics:
            assert mc.name in SC.ordered_statistics[SC.STATS_2nd_STATISTICS]
            if mc.name not in [SC.STATS_HISTOGRAM]:
                assert mc.config == {}
            else:
                assert mc.config == {"*": {"bins": 10}, "Age": {"bins": 5, "range": [0, 120]}}

    def test_wait_for_all_results(self):

        # waiting for 1 more client
        client_statistics = {
            "count": {"site-1": {}},
            "mean": {"site-2": {}},
            "sum": {"site-3": {}},
            "stddev": {"site-4": {}},
        }
        import time

        t0 = time.time()
        StatisticsController._wait_for_all_results(self.stats_controller.logger, 0.5, 3, client_statistics, 0.1)
        t = time.time()
        second_spent = t - t0
        # for 4 statistic, each have 0.5 second timeout
        assert second_spent > 0.5 * 4

    def test_prepare_input(self):
        xs = self.stats_controller._prepare_inputs(SC.STATS_1st_STATISTICS)
        assert xs[SC.STATISTICS_TASK_KEY] == SC.STATS_1st_STATISTICS
        seq = StatisticsController._get_target_statistics(
            self.stats_controller.statistic_configs, SC.ordered_statistics[SC.STATS_1st_STATISTICS]
        )
        rhs = [mc.name for mc in seq]
        rhs.sort()
        target_statistics: List[StatisticConfig] = fobs.loads(xs[SC.STATS_TARGET_STATISTICS])
        lhs = [mc.name for mc in target_statistics]
        lhs.sort()
        assert lhs == rhs

        # simulate aggregation and populate the global results
        self.stats_controller.global_statistics[SC.STATS_COUNT] = {"train": {"Age": 100}, "test": {"Age": 10}}
        self.stats_controller.global_statistics[SC.STATS_MEAN] = {"train": {"Age": 25}, "test": {"Age": 30}}

        self.stats_controller.global_statistics[SC.STATS_MAX] = {"train": {"Age": 120}, "test": {"Age": 120}}
        self.stats_controller.global_statistics[SC.STATS_MIN] = {"train": {"Age": 0}, "test": {"Age": 0}}

        assert self.stats_controller.global_statistics != {}

        xs = self.stats_controller._prepare_inputs(SC.STATS_2nd_STATISTICS)
        assert xs[SC.STATISTICS_TASK_KEY] == SC.STATS_2nd_STATISTICS
        rhs = SC.ordered_statistics[SC.STATS_2nd_STATISTICS]
        rhs.sort()
        target_statistics: List[StatisticConfig] = fobs.loads(xs[SC.STATS_TARGET_STATISTICS])
        lhs = [mc.name for mc in target_statistics]
        lhs.sort()
        assert lhs == rhs

    def test_validate_min_clients(self):

        # waiting for 1 more client
        client_statistics = {
            "count": {"site-1": {}},
            "mean": {"site-2": {}},
            "sum": {"site-3": {}},
            "stddev": {"site-4": {}},
        }
        assert not self.stats_controller._validate_min_clients(5, client_statistics)

        # waiting for 1 more client
        client_statistics = {
            "count": {"site-1": {"train": {}}},
            "mean": {"site-2": {"train": {}}},
            "sum": {"site-3": {"train": {}}},
            "stddev": {"site-4": {"train": {}}},
        }
        assert not self.stats_controller._validate_min_clients(5, client_statistics)
