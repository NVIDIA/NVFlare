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

from unittest.mock import MagicMock

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.workflows.statistics_controller import StatisticsController
from nvflare.fuel.utils import fobs


def _client_task(client_name, statistics_task, withheld=None):
    statistics = {StC.STATS_COUNT: {"train": {"age": 3}}, StC.STATS_FAILURE_COUNT: {"train": {"age": 0}}}
    data = {StC.STATISTICS_TASK_KEY: statistics_task, statistics_task: fobs.dumps(statistics)}
    if withheld is not None:
        data[StC.STATS_WITHHELD_FEATURES] = fobs.dumps(withheld)
    result = DXO(data_kind=DataKind.STATISTICS, data=data).to_shareable()
    result.set_return_code(ReturnCode.OK)

    client_task = MagicMock()
    client_task.client.name = client_name
    client_task.task.name = StC.FED_STATS_TASK
    client_task.result = result
    return client_task


class TestStatisticsControllerWithheldFeatures:
    def _controller(self):
        return StatisticsController(statistic_configs={StC.STATS_COUNT: {}}, writer_id="writer")

    def test_collects_withheld_features_per_client_across_rounds(self):
        controller = self._controller()

        controller.results_cb(_client_task("site-1", StC.STATS_1st_STATISTICS, {"train": ["age"]}), FLContext())
        controller.results_cb(_client_task("site-1", StC.STATS_2nd_STATISTICS, {"train": ["age", "edu"]}), FLContext())
        controller.results_cb(_client_task("site-2", StC.STATS_1st_STATISTICS), FLContext())

        assert controller.withheld_features == {"site-1": {"train": {"age", "edu"}}}

    def test_withheld_features_are_not_treated_as_a_client_metric(self):
        controller = self._controller()

        controller.results_cb(_client_task("site-1", StC.STATS_1st_STATISTICS, {"train": ["age"]}), FLContext())

        assert set(controller.client_statistics) == {StC.STATS_COUNT, StC.STATS_FAILURE_COUNT}
