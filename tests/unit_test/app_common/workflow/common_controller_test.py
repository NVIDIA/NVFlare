# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from unittest.mock import Mock

from nvflare.apis.fl_constant import ReturnCode
from tests.unit_test.app_common.workflow.mock_common_controller import MockController


class TestCommonController:
    def test_handle_client_errors(self):
        mock_client_task_result = Mock()
        mock_client_task_result.get_return_code = ReturnCode.EXECUTION_EXCEPTION

        ctr = MockController(mock_client_task_result)

        ctr.control_flow(fl_ctx=ctr.fl_ctx)
        # todo: how to register a event listener and listen to the system_panic event
