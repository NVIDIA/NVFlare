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
from nvflare.apis.controller_spec import Task, TaskOperatorKey, OperatorMethod
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.client_controller import ClientController
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.security.logging import secure_format_exception


class ClientScatterAndGather(ClientController):

    def __init__(self,
                 min_clients: int = 1000,
                 num_rounds: int = 5,
                 start_round: int = 0,
                 aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID
                 ):
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._start_round = start_round
        self.aggregator_id = aggregator_id

        self._current_round = None
        self.aggregator = None

    def start_controller(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.aggregator = engine.get_component(self.aggregator_id)
        if not isinstance(self.aggregator, Aggregator):
            self.system_panic(
                f"aggregator {self.aggregator_id} must be an Aggregator type object but got {type(self.aggregator)}",
                fl_ctx,
            )
            return

    def stop_controller(self, fl_ctx: FLContext):
        super().stop_controller(fl_ctx)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:

            self.log_info(fl_ctx, "Beginning ScatterAndGather training phase.")
            self._phase = AppConstants.PHASE_TRAIN

            fl_ctx.set_prop(AppConstants.PHASE, self._phase, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            if self._current_round is None:
                self._current_round = self._start_round
            while self._current_round < self._start_round + self._num_rounds:

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, f"Round {self._current_round} started.")
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

                # Create train_task
                data_shareable: Shareable = self.shareable_gen.learnable_to_shareable(self._global_weights, fl_ctx)
                data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
                data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
                data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

                operator = {
                    TaskOperatorKey.OP_ID: self.train_task_name,
                    TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
                    TaskOperatorKey.TIMEOUT: self._train_timeout,
                    TaskOperatorKey.AGGREGATOR: self.aggregator_id,
                }

                train_task = Task(
                    name=self.train_task_name,
                    data=data_shareable,
                    operator=operator,
                    props={},
                    timeout=self._train_timeout,
                    before_task_sent_cb=self._prepare_train_task_data,
                    result_received_cb=self._process_train_result,
                )

                replies = self.broadcast_and_wait(
                    task=train_task,
                    min_responses=self._min_clients,
                    # wait_time_after_min_received=self._wait_time_after_min_received,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                aggregation_result = self.aggregator.aggregate(replies, fl_ctx)

                fl_ctx.set_prop(FLContextKey.TASK_RESULT, aggregation_result)

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(fl_ctx, f"Round {self._current_round} finished.")
                self._current_round += 1

            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, "Finished ScatterAndGather Training.")
        except Exception as e:
            error_msg = f"Exception in ScatterAndGather control_flow: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

