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

import pytest

from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.workflows.hierarchical_statistics_controller import HierarchicalStatisticsController
from nvflare.fuel.utils import fobs


def _target_names(statistic_configs: dict, statistic_task: str) -> list:
    controller = HierarchicalStatisticsController(
        statistic_configs=statistic_configs, writer_id="stats_writer", hierarchy_config="hierarchy_config.json"
    )
    controller.global_statistics = {
        StC.GLOBAL: {
            StC.STATS_COUNT: {"ds": {"f": 10}},
            StC.STATS_MEAN: {"ds": {"f": 1.5}},
        }
    }
    inputs = controller._prepare_inputs(statistic_task)
    return [t.name for t in fobs.loads(inputs[StC.STATS_TARGET_STATISTICS])]


class TestHierarchicalStatisticsController:
    @pytest.mark.parametrize(
        "statistic_configs",
        [
            {"count": {}, "sum": {}, "mean": {}, "var": {}},
            {"count": {}, "sum": {}, "mean": {}, "stddev": {}},
            {"count": {}, "sum": {}, "mean": {}, "var": {}, "stddev": {}},
        ],
    )
    def test_2nd_round_requests_var_and_stddev(self, statistic_configs):
        names = _target_names(statistic_configs, StC.STATS_2nd_STATISTICS)
        assert StC.STATS_VAR in names
        assert names.count(StC.STATS_STDDEV) == 1

    def test_stddev_not_added_in_1st_round_or_without_var(self):
        statistic_configs = {"count": {}, "sum": {}, "mean": {}, "var": {}}
        assert StC.STATS_STDDEV not in _target_names(statistic_configs, StC.STATS_1st_STATISTICS)
        assert StC.STATS_STDDEV not in _target_names({"count": {}, "sum": {}}, StC.STATS_2nd_STATISTICS)
