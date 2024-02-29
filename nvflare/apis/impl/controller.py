# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional, Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, ControllerSpec, SendOrder, Task, TaskCompletionStatus
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.signal import Signal
from nvflare.apis.wf_comm_spec import WFCommSpec


class Controller(FLComponent, ControllerSpec, ABC):
    def __init__(self, task_check_period=0.2):
        """Manage life cycles of tasks and their destinations.

        Args:
            task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.2.
        """
        super().__init__()
        self._task_check_period = task_check_period
        self.communicator = None

    def set_communicator(self, communicator: WFCommSpec, fl_ctx: FLContext):
        communicator.task_check_period = self._task_check_period
        self.communicator = communicator
        engine = fl_ctx.get_engine()
        if not engine:
            self.system_panic(f"Engine not found. {self.__class__.__name__} exiting.", fl_ctx)
            return

        self._engine = engine

    def broadcast(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 1,
        wait_time_after_min_received: int = 0,
    ):
        self.communicator.broadcast(task, fl_ctx, targets, min_responses, wait_time_after_min_received)

    def broadcast_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 1,
        wait_time_after_min_received: int = 0,
        abort_signal: Optional[Signal] = None,
    ):
        self.communicator.broadcast_and_wait(
            task, fl_ctx, targets, min_responses, wait_time_after_min_received, abort_signal
        )

    def broadcast_forever(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None):
        self.communicator.broadcast_forever(task, fl_ctx, targets)

    def send(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
    ):
        self.communicator.send(task, fl_ctx, targets, send_order, task_assignment_timeout)

    def send_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        abort_signal: Signal = None,
    ):
        self.communicator.send_and_wait(task, fl_ctx, targets, send_order, task_assignment_timeout, abort_signal)

    def relay(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        task_result_timeout: int = 0,
        dynamic_targets: bool = True,
    ):
        self.communicator.relay(
            task, fl_ctx, targets, send_order, task_assignment_timeout, task_result_timeout, dynamic_targets
        )

    def relay_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order=SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        task_result_timeout: int = 0,
        dynamic_targets: bool = True,
        abort_signal: Optional[Signal] = None,
    ):
        self.communicator.relay_and_wait(
            task,
            fl_ctx,
            targets,
            send_order,
            task_assignment_timeout,
            task_result_timeout,
            dynamic_targets,
            abort_signal,
        )

    def get_num_standing_tasks(self) -> int:
        if not isinstance(self.communicator, WFCommServer):
            raise NotImplementedError
        return self.communicator.get_num_standing_tasks()

    def cancel_task(
        self, task: Task, completion_status=TaskCompletionStatus.CANCELLED, fl_ctx: Optional[FLContext] = None
    ):
        if not isinstance(self.communicator, WFCommServer):
            raise NotImplementedError
        self.communicator.cancel_task(task, completion_status, fl_ctx)

    def cancel_all_tasks(self, completion_status=TaskCompletionStatus.CANCELLED, fl_ctx: Optional[FLContext] = None):
        if not isinstance(self.communicator, WFCommServer):
            raise NotImplementedError
        self.communicator.cancel_all_tasks(completion_status, fl_ctx)
