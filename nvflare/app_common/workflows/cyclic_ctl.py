# Copyright (c) 2021, NVIDIA CORPORATION.
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

import random

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType


class CyclicController(Controller):
    """
    A sample implementation to demonstrate how to use relay method for Cyclic Federated Learning
    """

    def __init__(
        self,
        num_rounds: int = 5,
        task_assignment_timeout: int = 10,
        persistor_id="persistor",
        shareable_generator_id="shareable_generator",
        task_name="train",
    ):
        super().__init__()
        if not isinstance(num_rounds, int):
            raise TypeError("num_rounds must be int.")
        if not isinstance(task_assignment_timeout, int):
            raise TypeError("task_assignment_timeout must be int.")
        if not isinstance(persistor_id, str):
            raise TypeError("persistor_id must be a string.")
        if not isinstance(shareable_generator_id, str):
            raise TypeError("shareable_generator_id must be a string.")
        if not isinstance(task_name, str):
            raise TypeError("task_name must be a string.")
        self.num_rounds = num_rounds
        self.persistor_id = persistor_id
        self.shareable_generator_id = shareable_generator_id
        self.task_assignment_timeout = task_assignment_timeout
        self.task_name = task_name

    def start_controller(self, fl_ctx: FLContext):
        self.log_debug(fl_ctx, "starting controller")
        self.persistor = fl_ctx.get_engine().get_component(self.persistor_id)
        self.shareable_generator = fl_ctx.get_engine().get_component(self.shareable_generator_id)
        if not isinstance(self.persistor, LearnablePersistor):
            self.system_panic(f"Persistor {self.persistor_id} must be a Persistor instance", fl_ctx)
        if not isinstance(self.shareable_generator, ShareableGenerator):
            self.system_panic(
                f"Shareable generator {self.shareable_generator_id} must be a Shareable Generator instance", fl_ctx
            )
        self.last_learnable = self.persistor.load(fl_ctx)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self.last_learnable, private=True, sticky=True)
        self.fire_event(AppEventType.INITIAL_MODEL_LOADED, fl_ctx)

    def _process_result(self, client_task: ClientTask, fl_ctx: FLContext):
        # submitted shareable is stored in client_task.result
        # we need to update task.data with that shareable so the next target
        # will get the updated shareable
        task = client_task.task

        # update the global learnable with the received result (shareable)
        # e.g. the received result could be weight_diffs, the learnable could be full weights.
        self.last_learnable = self.shareable_generator.shareable_to_learnable(client_task.result, fl_ctx)

        # prepare task shareable data for next client
        task.data = self.shareable_generator.learnable_to_shareable(self.last_learnable, fl_ctx)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:
            engine = fl_ctx.get_engine()
            self.log_debug(fl_ctx, "Cyclic starting.")

            for current_round in range(self.num_rounds):
                if abort_signal.triggered:
                    return

                self.log_debug(fl_ctx, f"Starting {current_round=}.")
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round, private=True, sticky=False)

                # Task for one cyclic
                targets = engine.get_clients()
                random.shuffle(targets)
                targets_names = [t.name for t in targets]
                self.log_debug(fl_ctx, f"Relay on {targets_names}")

                shareable = self.shareable_generator.learnable_to_shareable(self.last_learnable, fl_ctx)
                shareable.set_header(AppConstants.CURRENT_ROUND, current_round)

                task = Task(
                    name=self.task_name,
                    data=shareable,
                    result_received_cb=self._process_result,
                )

                self.relay_and_wait(
                    task=task,
                    targets=targets,
                    task_assignment_timeout=self.task_assignment_timeout,
                    fl_ctx=fl_ctx,
                    dynamic_targets=False,
                    abort_signal=abort_signal,
                )
                self.persistor.save(self.last_learnable, fl_ctx)
                self.log_debug(fl_ctx, f"Ending {current_round=}.")

            self.log_debug(fl_ctx, "Cyclic ended.")
        except BaseException as e:
            error_msg = f"Cyclic control_flow exception {e}"
            self.log_error(fl_ctx, error_msg)
            self.system_panic(str(e), fl_ctx)

    def stop_controller(self, fl_ctx: FLContext):
        self.persistor.save(learnable=self.last_learnable, fl_ctx=fl_ctx)
        self.log_debug(fl_ctx, "controller stopped")

    def process_result_of_unknown_task(
        self,
        client: Client,
        task_name: str,
        client_task_id: str,
        result: Shareable,
        fl_ctx: FLContext,
    ):
        self.log_warning(fl_ctx, f"Dropped result of unknown task: {task_name} from client {client.name}.")
