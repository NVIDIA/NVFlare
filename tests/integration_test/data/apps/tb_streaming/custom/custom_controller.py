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

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants


def _prepare_training_ctx(client_task: ClientTask, fl_ctx: FLContext):
    task = client_task.task
    fl_ctx.set_prop("current_round", task.props["round"], private=False)
    fl_ctx.set_prop("total_rounds", task.props["total"], private=False)


def _process_training_result(client_task: ClientTask, fl_ctx: FLContext):
    task = client_task.task
    task.data = client_task.result


class CustomController(Controller):
    def __init__(
        self,
        min_clients: int,
        num_rounds: int,
        persistor_id="persistor",
        shareable_generator_id="shareable_generator",
    ):
        Controller.__init__(self)
        self.persistor_id = persistor_id
        self.shareable_generator_id = shareable_generator_id
        self.persistor = None
        self.shareable_gen = None

        # config data
        self._min_clients = min_clients
        self._num_rounds = num_rounds

        # workflow phases: init, train
        self._phase = "init"
        self._global_model = None

    def start_controller(self, fl_ctx: FLContext):
        self._phase = "init"
        engine = fl_ctx.get_engine()

        self.shareable_gen = engine.get_component(self.shareable_generator_id)
        if not isinstance(self.shareable_gen, ShareableGenerator):
            self.system_panic("shareable_gen should be an instance of ShareableGenerator.", fl_ctx)

        self.persistor = engine.get_component(self.persistor_id)
        if not isinstance(self.persistor, LearnablePersistor):
            self.system_panic("persistor should be an instance of LearnablePersistor.", fl_ctx)

        self._global_model = self.persistor.load(fl_ctx)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_model, private=True, sticky=True)

    def process_result_of_unknown_task(
        self,
        client: Client,
        task_name: str,
        client_task_id: str,
        result: Shareable,
        fl_ctx: FLContext,
    ):
        return None

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self._phase = "train"
        engine = fl_ctx.get_engine()

        # No rounds - will remove later
        for r in range(self._num_rounds):
            if not abort_signal:
                return
            task = Task(
                name="poc",
                data=self.shareable_gen.learnable_to_shareable(self._global_model, fl_ctx),
                props={"round": r, "total": self._num_rounds},
                timeout=0,
                before_task_sent_cb=_prepare_training_ctx,
                result_received_cb=_process_training_result,
            )

            client_list = engine.get_clients()
            for c in client_list:
                self.log_info(fl_ctx, f"@@@ client name {c.name}")

            self.log_info(fl_ctx, f"@@@ Broadcast and wait {task.name}")
            self.broadcast_and_wait(
                task=task,
                fl_ctx=fl_ctx,
                targets=None,
                min_responses=0,
                abort_signal=abort_signal,
            )
            self.log_info(fl_ctx, f"@@@ Broadcast and wait - end {task.name}")

            self._global_model = self.shareable_gen.shareable_to_learnable(task.data, fl_ctx)
            self.persistor.save(self._global_model, fl_ctx)

            self.logger.info("model saved")

    def stop_controller(self, fl_ctx: FLContext):
        self._phase = "finished"
