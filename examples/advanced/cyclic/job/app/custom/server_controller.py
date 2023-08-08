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

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import ReservedTopic, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType


class ServerCyclicController(Controller):
    def __init__(self, num_rounds: int = 5, task_name="train"):
        super().__init__()

        self.num_rounds = num_rounds
        self.task_name = task_name

        self.waiter = threading.Event()

    def start_controller(self, fl_ctx: FLContext):
        super().start_controller(fl_ctx)

        engine = fl_ctx.get_engine()
        engine.register_aux_message_handler(topic=ReservedTopic.DO_TASK, message_handle_func=self._handle_aux_message)

    def _handle_aux_message(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        final_rounds = request.get_header(AppConstants.CURRENT_ROUND)
        contributions = request.get_header(AppConstants.CONTRIBUTIONS)

        self.logger.info(f"got the final result from cyclic controller. total round: {final_rounds}")
        self.logger.info(f"Contribution rounds from the clients: {contributions}")

        self._persist_result(request, fl_ctx)
        self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)
        self.log_info(fl_ctx, "End persist model on server.")

        self.waiter.set()
        return make_reply(ReturnCode.OK)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # Task for one cyclic
        targets = fl_ctx.get_engine().get_clients()

        targets_names = [t.name for t in targets]
        self.log_debug(fl_ctx, f"Cyclic controller on targets: {targets_names}")
        start_client = targets_names[0]

        shareable = Shareable()
        shareable.set_header(AppConstants.NUM_ROUNDS, self.num_rounds)
        shareable.set_header(AppConstants.PARTICIPATING_CLIENTS, targets_names)

        task = Task(
            name=self.task_name,
            data=shareable,
            result_received_cb=self._process_result,
        )

        self.send(task=task, fl_ctx=fl_ctx, targets=[start_client])
        self.logger.info(
            f"Task {task.name} has been sent to client {start_client}, " f"wait for the cyclic controller to finish ..."
        )

        self.waiter.wait()
        self.logger.info("The cyclic controller completed.")

    def _process_result(self, client_task: ClientTask, fl_ctx: FLContext):
        # super()._process_result(client_task, fl_ctx)
        pass

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def _persist_result(self, request, fl_ctx):
        self.logger.info(f"Persist the cyclic final result: {request}")
