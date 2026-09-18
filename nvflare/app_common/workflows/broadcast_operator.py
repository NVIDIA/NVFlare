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
import time
from typing import Dict, List, Optional, Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.workflows.error_handling_controller import ErrorHandlingController


class BroadcastAndWait(FLComponent):
    def __init__(self, fl_ctx: FLContext, controller: ErrorHandlingController, log_client_names: bool = True):
        super().__init__()
        self.lock = threading.Lock()
        self.fl_ctx = fl_ctx
        self.controller = controller
        self.log_client_names = log_client_names
        self.task = None

        # [target, DXO]
        self.results: Dict[str, DXO] = {}

    def broadcast_and_wait(
        self,
        task_name: str,
        task_input: Shareable,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        task_props: Optional[Dict] = None,
        min_responses: int = 1,
        abort_signal: Signal = None,
    ) -> Dict[str, DXO]:
        task = Task(name=task_name, data=task_input, result_received_cb=self.results_cb, props=task_props)
        self.controller.broadcast_and_wait(task, fl_ctx, targets, min_responses, 0, abort_signal)
        return self.results

    def multicasts_and_wait(
        self,
        task_name: str,
        task_inputs: Dict[str, Shareable],
        fl_ctx: FLContext,
        abort_signal: Signal = None,
        task_check_period: int = 0.5,
    ) -> Dict[str, DXO]:

        tasks: Dict[str, Task] = self.get_tasks(task_name, task_inputs)
        for client_name in tasks:
            self.controller.send(task=tasks[client_name], fl_ctx=fl_ctx, targets=[client_name])

        while self.controller.get_num_standing_tasks():
            if abort_signal.triggered:
                self.log_info(fl_ctx, "Abort signal triggered. Finishing multicasts_and_wait.")
                return
            self.log_debug(fl_ctx, "Checking standing tasks to see if multicasts_and_wait finished.")
            time.sleep(task_check_period)

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
        log_ctx = self._logging_context(fl_ctx)
        processing_client_detail = f" from client {client_name}" if self.log_client_names else ""
        self.log_info(log_ctx, f"Processing {task_name}, {self.task} result{processing_client_detail}")
        result = client_task.result
        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            received_client_detail = f" from client:{client_name}" if self.log_client_names else ""
            self.log_info(log_ctx, f"Received result{received_client_detail} for task {task_name} ")
            dxo = from_shareable(result)
            self.update_result(client_name, dxo)
        else:
            if rc in self.controller.abort_job_in_error.keys():
                self.controller.handle_client_errors(rc, client_task, log_ctx)

        # Cleanup task result
        client_task.result = None

    def _logging_context(self, fl_ctx: FLContext) -> FLContext:
        if self.log_client_names or not fl_ctx:
            return fl_ctx

        log_ctx = fl_ctx.clone()
        log_ctx.remove_prop(ReservedKey.PEER_CTX, force_removal=True)
        return log_ctx
