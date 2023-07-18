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

import gc
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
from nvflare.security.logging import secure_format_exception


class RelayOrder:
    FIXED = "FIXED"
    RANDOM = "RANDOM"
    RANDOM_WITHOUT_SAME_IN_A_ROW = "RANDOM_WITHOUT_SAME_IN_A_ROW"


SUPPORTED_ORDERS = (RelayOrder.FIXED, RelayOrder.RANDOM, RelayOrder.RANDOM_WITHOUT_SAME_IN_A_ROW)


class CyclicController(Controller):
    def __init__(
        self,
        num_rounds: int = 5,
        task_assignment_timeout: int = 10,
        persistor_id="persistor",
        shareable_generator_id="shareable_generator",
        task_name="train",
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
        snapshot_every_n_rounds: int = 1,
        order: str = RelayOrder.FIXED,
    ):
        """A sample implementation to demonstrate how to use relay method for Cyclic Federated Learning.

        Args:
            num_rounds (int, optional): number of rounds this controller should perform. Defaults to 5.
            task_assignment_timeout (int, optional): timeout (in sec) to determine if one client fails to
                request the task which it is assigned to . Defaults to 10.
            persistor_id (str, optional): id of the persistor so this controller can save a global model.
                Defaults to "persistor".
            shareable_generator_id (str, optional): id of shareable generator. Defaults to "shareable_generator".
            task_name (str, optional): the task name that clients know how to handle. Defaults to "train".
            task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.5.
            persist_every_n_rounds (int, optional): persist the global model every n rounds. Defaults to 1.
                If n is 0 then no persist.
            snapshot_every_n_rounds (int, optional): persist the server state every n rounds. Defaults to 1.
                If n is 0 then no persist.
            order (str, optional): the order of relay.
                If FIXED means the same order for every round.
                If RANDOM means random order for every round.
                If RANDOM_WITHOUT_SAME_IN_A_ROW means every round the order gets shuffled but a client will never be
                run twice in a row (in different round).

        Raises:
            TypeError: when any of input arguments does not have correct type

        """
        super().__init__(task_check_period=task_check_period)

        if not isinstance(num_rounds, int):
            raise TypeError("num_rounds must be int but got {}".format(type(num_rounds)))
        if not isinstance(task_assignment_timeout, int):
            raise TypeError("task_assignment_timeout must be int but got {}".format(type(task_assignment_timeout)))
        if not isinstance(persistor_id, str):
            raise TypeError("persistor_id must be a string but got {}".format(type(persistor_id)))
        if not isinstance(shareable_generator_id, str):
            raise TypeError("shareable_generator_id must be a string but got {}".format(type(shareable_generator_id)))
        if not isinstance(task_name, str):
            raise TypeError("task_name must be a string but got {}".format(type(task_name)))

        if order not in SUPPORTED_ORDERS:
            raise ValueError(f"order must be in {SUPPORTED_ORDERS}")

        self._num_rounds = num_rounds
        self._start_round = 0
        self._end_round = self._start_round + self._num_rounds
        self._current_round = 0
        self._last_learnable = None
        self.persistor_id = persistor_id
        self.shareable_generator_id = shareable_generator_id
        self.task_assignment_timeout = task_assignment_timeout
        self.task_name = task_name
        self.persistor = None
        self.shareable_generator = None
        self._persist_every_n_rounds = persist_every_n_rounds
        self._snapshot_every_n_rounds = snapshot_every_n_rounds
        self._participating_clients = None
        self._last_client = None
        self._order = order

    def start_controller(self, fl_ctx: FLContext):
        self.log_debug(fl_ctx, "starting controller")
        self.persistor = self._engine.get_component(self.persistor_id)
        self.shareable_generator = self._engine.get_component(self.shareable_generator_id)
        if not isinstance(self.persistor, LearnablePersistor):
            self.system_panic(
                f"Persistor {self.persistor_id} must be a Persistor instance, but got {type(self.persistor)}", fl_ctx
            )
        if not isinstance(self.shareable_generator, ShareableGenerator):
            self.system_panic(
                f"Shareable generator {self.shareable_generator_id} must be a Shareable Generator instance,"
                f"but got {type(self.shareable_generator)}",
                fl_ctx,
            )
        self._last_learnable = self.persistor.load(fl_ctx)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._last_learnable, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=True)
        self.fire_event(AppEventType.INITIAL_MODEL_LOADED, fl_ctx)

        self._participating_clients = self._engine.get_clients()
        if len(self._participating_clients) <= 1:
            self.system_panic("Not enough client sites.", fl_ctx)
        self._last_client = None

    def _get_relay_orders(self):
        targets = list(self._participating_clients)
        if self._order == RelayOrder.RANDOM:
            random.shuffle(targets)
        elif self._order == RelayOrder.RANDOM_WITHOUT_SAME_IN_A_ROW:
            random.shuffle(targets)
            if self._last_client == targets[0]:
                targets = targets.append(targets.pop(0))
        self._last_client = targets[-1]
        return targets

    def _process_result(self, client_task: ClientTask, fl_ctx: FLContext):
        # submitted shareable is stored in client_task.result
        # we need to update task.data with that shareable so the next target
        # will get the updated shareable
        task = client_task.task

        # update the global learnable with the received result (shareable)
        # e.g. the received result could be weight_diffs, the learnable could be full weights.
        self._last_learnable = self.shareable_generator.shareable_to_learnable(client_task.result, fl_ctx)

        # prepare task shareable data for next client
        task.data = self.shareable_generator.learnable_to_shareable(self._last_learnable, fl_ctx)
        task.data.set_header(AppConstants.CURRENT_ROUND, self._current_round)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:
            self.log_debug(fl_ctx, "Cyclic starting.")

            for self._current_round in range(self._start_round, self._end_round):
                if abort_signal.triggered:
                    return

                self.log_debug(fl_ctx, "Starting current round={}.".format(self._current_round))
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)

                # Task for one cyclic
                targets = self._get_relay_orders()
                targets_names = [t.name for t in targets]
                self.log_debug(fl_ctx, f"Relay on {targets_names}")

                shareable = self.shareable_generator.learnable_to_shareable(self._last_learnable, fl_ctx)
                shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)

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

                if self._persist_every_n_rounds != 0 and (self._current_round + 1) % self._persist_every_n_rounds == 0:
                    self.log_info(fl_ctx, "Start persist model on server.")
                    self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                    self.persistor.save(self._last_learnable, fl_ctx)
                    self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)
                    self.log_info(fl_ctx, "End persist model on server.")

                if (
                    self._snapshot_every_n_rounds != 0
                    and (self._current_round + 1) % self._snapshot_every_n_rounds == 0
                ):
                    # Call the self._engine to persist the snapshot of all the FLComponents
                    self._engine.persist_components(fl_ctx, completed=False)

                self.log_debug(fl_ctx, "Ending current round={}.".format(self._current_round))
                gc.collect()

            self.log_debug(fl_ctx, "Cyclic ended.")
        except Exception as e:
            error_msg = f"Cyclic control_flow exception: {secure_format_exception(e)}"
            self.log_error(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def stop_controller(self, fl_ctx: FLContext):
        self.persistor.save(learnable=self._last_learnable, fl_ctx=fl_ctx)
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

    def get_persist_state(self, fl_ctx: FLContext) -> dict:
        return {
            "current_round": self._current_round,
            "end_round": self._end_round,
            "last_learnable": self._last_learnable,
        }

    def restore(self, state_data: dict, fl_ctx: FLContext):
        try:
            self._current_round = state_data.get("current_round")
            self._end_round = state_data.get("end_round")
            self._last_learnable = state_data.get("last_learnable")
            self._start_round = self._current_round
        finally:
            pass
