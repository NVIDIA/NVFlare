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

import threading
from typing import Dict, List, Optional, Union

from nvflare.apis.client import Client
from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.workflows.error_handling_controller import ErrorHandlingController


class BroadcastAndWait(FLComponent):
    def __init__(self, fl_ctx: FLContext, controller: ErrorHandlingController):
        super().__init__()
        self.lock = threading.Lock()
        self.fl_ctx = fl_ctx
        self.controller = controller
        self.task = None

        # [target, DXO]
        self.results: Dict[str, DXO] = {}

    def broadcast_and_wait(
        self,
        task_name: str,
        task_input: Shareable,
        targets: Union[List[Client], List[str], None] = None,
        task_props: Optional[Dict] = None,
        min_responses: int = 1,
        abort_signal: Signal = None,
    ) -> Dict[str, DXO]:
        task = Task(name=task_name, data=task_input, result_received_cb=self.results_cb, props=task_props)
        self.controller.broadcast_and_wait(task, self.fl_ctx, targets, min_responses, 0, abort_signal)
        return self.results

    def multicasts_and_wait(
        self,
        task_name: str,
        task_inputs: Dict[str, Shareable],
        abort_signal: Signal = None,
    ) -> Dict[str, DXO]:

        tasks: Dict[str, Task] = self.get_tasks(task_name, task_inputs)
        for client_name in tasks:
            self.controller.broadcast(task=tasks[client_name], fl_ctx=self.fl_ctx, targets=[client_name])

        for client_name in tasks:
            self.log_info(self.fl_ctx, f"wait for client {client_name} task")
            self.controller.wait_for_task(tasks[client_name], abort_signal)

        return self.results

    def get_tasks(self, task_name: str, task_inputs: Dict[str, Shareable]) -> Dict[str, Task]:
        tasks = {}
        for client_name in task_inputs:
            task = Task(name=task_name, data=task_inputs[client_name], result_received_cb=self.results_cb)
            tasks[client_name] = task
        return tasks

    def update_result(self, client_name: str, dxo: DXO):
        try:
            self.lock.acquire()
            self.log_debug(self.fl_ctx, "Acquired a lock")
            self.results.update({client_name: dxo})
        finally:
            self.log_debug(self.fl_ctx, "Released a lock")
            self.lock.release()

    def results_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        client_name = client_task.client.name
        task_name = client_task.task.name
        print("task_name", task_name)
        self.log_info(fl_ctx, f"Processing {task_name}, {self.task} result from client {client_name}")
        result = client_task.result
        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"Received result from client:{client_name} for task {task_name} ")
            dxo = from_shareable(result)
            self.update_result(client_name, dxo)
        else:
            if rc in self.controller.abort_job_in_error.keys():
                self.controller.handle_client_errors(rc, client_task, fl_ctx)

        # Cleanup task result
        client_task.result = None
