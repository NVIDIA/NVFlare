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

from unittest.mock import patch

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.workflows.error_handling_controller import ErrorHandlingController


class MockController(ErrorHandlingController):
    def __init__(self, mock_client_task_result):
        super().__init__()
        self.fl_ctx = FLContext()
        self.task_name = "MockTask"
        self.mock_client_task_result = mock_client_task_result

    @patch.object(ErrorHandlingController, "broadcast_and_wait")
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"{self.task_name} control flow started.")
        task_input = Shareable()
        task = Task(name=self.task_name, data=task_input, result_received_cb=self.results_cb)
        self.broadcast_and_wait(task, self.fl_ctx, ["no_where"], 0, 0)

        self.log_info(fl_ctx, f"task {self.task_name} control flow end.")

    def start_controller(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def results_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        client_name = client_task.client.name
        client_task.result = self.mock_client_task_result
        result = client_task.result
        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            dxo = from_shareable(result)
            self.update_result(client_name, dxo)
        else:
            if rc in self.controller.abort_job_in_error.keys():
                self.controller.handle_client_errors(rc, client_task, fl_ctx)

        # Cleanup task result
        client_task.result = None
