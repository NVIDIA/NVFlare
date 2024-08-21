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
from nvflare.apis.controller_spec import ControllerSpec, SendOrder, Task, TaskCompletionStatus
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.wf_comm_spec import WFCommSpec


class Controller(FLComponent, ControllerSpec, ABC):
    def __init__(self, task_check_period=0.2):
        """Controller logic for tasks and their destinations.

        Must set_communicator() to access communication related function implementations.

        Args:
            task_check_period (float, optional): interval for checking status of tasks. Applicable for WFCommServer. Defaults to 0.2.
        """
        super().__init__()
        self._task_check_period = task_check_period
        self.communicator = None

    def initialize(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if not engine:
            self.system_panic(f"Engine not found. {self.__class__.__name__} exiting.", fl_ctx)
            return

        self._engine = engine
        self.start_controller(fl_ctx)

    def set_communicator(self, communicator: WFCommSpec):
        if not isinstance(communicator, WFCommSpec):
            raise TypeError(f"communicator must be an instance of WFCommSpec, but got {type(communicator)}")

        self.communicator = communicator
        self.communicator.controller = self
        self.communicator.task_check_period = self._task_check_period

    def broadcast(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 1,
        wait_time_after_min_received: int = 0,
    ):
        return self.communicator.broadcast(task, fl_ctx, targets, min_responses, wait_time_after_min_received)

    def broadcast_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        min_responses: int = 1,
        wait_time_after_min_received: int = 0,
        abort_signal: Optional[Signal] = None,
    ):
        return self.communicator.broadcast_and_wait(
            task, fl_ctx, targets, min_responses, wait_time_after_min_received, abort_signal
        )

    def broadcast_forever(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None):
        return self.communicator.broadcast_forever(task, fl_ctx, targets)

    def send(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
    ):
        return self.communicator.send(task, fl_ctx, targets, send_order, task_assignment_timeout)

    def send_and_wait(
        self,
        task: Task,
        fl_ctx: FLContext,
        targets: Union[List[Client], List[str], None] = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        task_assignment_timeout: int = 0,
        abort_signal: Signal = None,
    ):
        return self.communicator.send_and_wait(task, fl_ctx, targets, send_order, task_assignment_timeout, abort_signal)

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
        return self.communicator.relay(
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
        return self.communicator.relay_and_wait(
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
        try:
            return self.communicator.get_num_standing_tasks()
        except Exception as e:
            self.logger.warning(f"get_num_standing_tasks() is not supported by {self.communicator}: {e}")
            return None

    def cancel_task(
        self, task: Task, completion_status=TaskCompletionStatus.CANCELLED, fl_ctx: Optional[FLContext] = None
    ):
        self.communicator.cancel_task(task, completion_status, fl_ctx)

    def cancel_all_tasks(self, completion_status=TaskCompletionStatus.CANCELLED, fl_ctx: Optional[FLContext] = None):
        try:
            self.communicator.cancel_all_tasks(completion_status, fl_ctx)
        except Exception as e:
            self.log_warning(fl_ctx, f"cancel_all_tasks() is not supported by {self.communicator}: {e}")

    def get_client_disconnect_time(self, client_name):
        """Get the time when the client is deemed disconnected.

        Args:
            client_name: the name of the client

        Returns: time at which the client was deemed disconnected; or None if the client is not disconnected.

        """
        if not self.communicator:
            return None

        try:
            return self.communicator.get_client_disconnect_time(client_name)
        except Exception as e:
            self.logger.warning(f"get_client_disconnect_time() is not supported by {self.communicator}: {e}")
            return None

    def add_to_fed_job(self, job, ctx, **kwargs):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:

        """
        job.check_kwargs(args_to_check=kwargs, args_expected={})
        job.add_controller(obj=self, ctx=ctx)
