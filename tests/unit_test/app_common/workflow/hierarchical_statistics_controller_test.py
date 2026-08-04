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

from copy import deepcopy

from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.workflows.hierarchical_statistics_controller import HierarchicalStatisticsController


class TestHierarchicalStatisticsController:
    def test_rebuild_global_statistics_uses_latest_client_results(self):
        controller = HierarchicalStatisticsController(
            statistic_configs={StC.STATS_COUNT: {}, StC.STATS_FAILURE_COUNT: {}},
            writer_id="",
        )
        controller.client_statistics = {
            StC.STATS_COUNT: {
                "site-1": {"train": {"Age": 4}},
                "site-2": {"train": {"Age": 4}},
            },
            StC.STATS_FAILURE_COUNT: {
                "site-1": {"train": {"Age": 0}},
                "site-2": {"train": {"Age": 0}},
            },
        }
        hierarchy_config = {"Sites": ["site-1", "site-2"]}
        controller._rebuild_global_statistics_with_hierarchy_config(StC.STATS_1st_STATISTICS, hierarchy_config)

        controller.client_statistics[StC.STATS_FAILURE_COUNT]["site-1"]["train"]["Age"] = 1
        controller._rebuild_global_statistics_with_hierarchy_config(StC.STATS_2nd_STATISTICS, hierarchy_config)

        global_statistics = controller.global_statistics[StC.GLOBAL]
        assert global_statistics[StC.STATS_COUNT]["train"]["Age"] == 8
        assert global_statistics[StC.STATS_FAILURE_COUNT]["train"]["Age"] == 1

        site_statistics = controller.global_statistics["Sites"]
        assert site_statistics[0][StC.LOCAL][StC.STATS_FAILURE_COUNT]["train"]["Age"] == 1
        assert site_statistics[1][StC.LOCAL][StC.STATS_FAILURE_COUNT]["train"]["Age"] == 0

        rebuilt_statistics = deepcopy(controller.global_statistics)
        controller._rebuild_global_statistics_with_hierarchy_config(StC.STATS_2nd_STATISTICS, hierarchy_config)
        assert controller.global_statistics == rebuilt_statistics
