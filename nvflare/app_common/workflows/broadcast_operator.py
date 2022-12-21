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
from abc import ABC
from typing import Dict, List, Union

from nvflare.apis.client import Client
from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.workflows.common_controller import CommonController


class BroadcastAndWait(FLComponent, ABC):
    def __init__(self, fl_ctx: FLContext, controller: CommonController):
        super().__init__()
        self.fl_ctx = fl_ctx
        self.controller = controller
        self.results: Dict[str, DXO] = {}

    def broadcast_and_wait(
        self,
        task_name: str,
        task_props: dict,
        inputs: Shareable,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 0,
        abort_signal: Signal = None,
    ) -> Dict[str, DXO]:

        task = Task(name=task_name, data=inputs, result_received_cb=self.results_cb, props=task_props)
        self.controller.broadcast_and_wait(task, self.fl_ctx, targets, min_responses, 0, abort_signal)
        return self.results

    def results_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        client_name = client_task.client.name
        task_name = client_task.task.name
        self.log_info(fl_ctx, f"Processing {task_name} result from client {client_name}")
        result = client_task.result

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"Received result from client:{client_name} for task {task_name} ")
            dxo = from_shareable(result)
            self.results.update({client_name: dxo})
        else:
            if rc in self.controller.abort_job_in_error.keys():
                self.controller.handle_client_errors(rc, client_task, fl_ctx)

        # Cleanup task result
        client_task.result = None
