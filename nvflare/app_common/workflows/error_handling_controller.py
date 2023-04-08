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

from abc import ABC

from nvflare.apis.controller_spec import ClientTask
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller


class ErrorHandlingController(Controller, ABC):
    def __init__(self):
        super().__init__()
        self.abort_job_in_error = {
            ReturnCode.EXECUTION_EXCEPTION: True,
            ReturnCode.TASK_UNKNOWN: True,
            ReturnCode.EXECUTION_RESULT_ERROR: False,
            ReturnCode.TASK_DATA_FILTER_ERROR: True,
            ReturnCode.TASK_RESULT_FILTER_ERROR: True,
        }

    def handle_client_errors(self, rc: str, client_task: ClientTask, fl_ctx: FLContext):
        client_name = client_task.client.name
        task_name = client_task.task.name
        abort = self.abort_job_in_error[rc]
        self.log_error(fl_ctx, f"error code = {rc}")
        if abort:
            self.system_panic(
                f"Failed in client-site for {client_name} during task {task_name}.controller is exiting.",
                fl_ctx=fl_ctx,
            )
            self.log_error(fl_ctx, f"Execution failed for {client_name}")
        else:
            raise ValueError(f"Execution result is not received for {client_name}")
