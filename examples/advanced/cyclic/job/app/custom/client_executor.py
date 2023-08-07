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

import logging
import time

from nvflare.apis.controller_spec import Task
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.client_controller import ClientController
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants


class ClientExecutor(Executor, ClientController):
    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super(ClientExecutor, self).handle_event(event_type, fl_ctx)
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        # engine = fl_ctx.get_engine()
        pass

    def execute(self, task_name: str, request: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.logger.info(f"ClientTrainer abort signal: {abort_signal.triggered}")
        if abort_signal.triggered:
            self.finalize(fl_ctx)
            shareable = Shareable()
            return shareable

        result = self._execute_task(task_name, request)

        client_name = fl_ctx.get_identity_name()

        # Keep track the contribution info from the client
        contributions = request.get_header(AppConstants.CONTRIBUTIONS, {})
        contribution_count = contributions.get(client_name, 0)
        contribution_count += 1
        contributions[client_name] = contribution_count
        request.set_header(AppConstants.CONTRIBUTIONS, contributions)

        next_target = self._get_next_target(request, client_name)
        if next_target:
            task = Task(name=task_name, data=result, props={}, timeout=30)
            self.send(task, fl_ctx, [next_target])
        return result

    def finalize(self, fl_ctx: FLContext):
        # super().finalize(fl_ctx)
        pass

    def _execute_task(self, task_name, shareable):
        self.logger.info(f"Executing task: {task_name} ...... ")
        time.sleep(1.0)
        self.logger.info("Perform the local training, generating the training result in shareable.")
        return shareable

    def _get_next_target(self, request: [], client_name):
        targets = request.get_header(AppConstants.PARTICIPATING_CLIENTS)
        index = 0
        for index in range(len(targets)):
            if targets[index] == client_name:
                break

        if index != len(targets)-1:
            next_target = targets[index+1]
        else:
            current_round = request.get_header(AppConstants.CURRENT_ROUND, 0)
            current_round += 1
            request.set_header(AppConstants.CURRENT_ROUND, current_round)
            num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
            if current_round >= num_rounds:
                next_target = SiteType.SERVER
            else:
                next_target = targets[0]
        return next_target

