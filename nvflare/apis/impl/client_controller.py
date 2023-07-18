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
from typing import Union, List

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ControllerSpec, Task, SendOrder
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.signal import Signal


class ClientController(Controller):
    def start_controller(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def broadcast(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None,
                  min_responses: int = 0, wait_time_after_min_received: int = 0):
        # super().broadcast(task, fl_ctx, targets, min_responses, wait_time_after_min_received)

        engine = fl_ctx.get_engine()
        request = task.data
        reply = engine.send_aux_request(
            targets=targets, topic="client_controller_task", request=request, timeout=task.timeout, fl_ctx=fl_ctx
        )

        return reply

    def broadcast_and_wait(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None,
                           min_responses: int = 0, wait_time_after_min_received: int = 0, abort_signal: Signal = None):
        engine = fl_ctx.get_engine()
        request = task.data
        reply = engine.send_aux_request(
            targets=targets, topic="client_controller_task", request=request, timeout=task.timeout, fl_ctx=fl_ctx
        )

        return reply

    def send(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None,
             send_order: SendOrder = SendOrder.SEQUENTIAL, task_assignment_timeout: int = 0):
        super().send(task, fl_ctx, targets, send_order, task_assignment_timeout)

    def send_and_wait(self, task: Task, fl_ctx: FLContext, targets: Union[List[Client], List[str], None] = None,
                      send_order: SendOrder = SendOrder.SEQUENTIAL, task_assignment_timeout: int = 0,
                      abort_signal: Signal = None):
        super().send_and_wait(task, fl_ctx, targets, send_order, task_assignment_timeout, abort_signal)