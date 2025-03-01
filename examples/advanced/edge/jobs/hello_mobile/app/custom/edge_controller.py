# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any

from nvflare.apis.controller_spec import Task, ClientTask
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.aggregators.edge_result_accumulator import EdgeResultAccumulator
from nvflare.security.logging import secure_format_exception


class SimpleEdgeController(Controller):

    def __init__(
        self,
        num_rounds: int,
        initial_weights: Any
    ):
        super().__init__()
        self.num_rounds = num_rounds
        self.current_round = None
        self.initial_weights = initial_weights
        self.aggregator = None

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Initializing Simple mobile workflow.")
        self.aggregator = EdgeResultAccumulator()

        # initialize global model
        fl_ctx.set_prop(AppConstants.START_ROUND, 1, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self.num_rounds, private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self.initial_weights, private=True, sticky=True)
        self.fire_event(AppEventType.INITIAL_MODEL_LOADED, fl_ctx)

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stopping Simple mobile workflow.")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:

            self.log_info(fl_ctx, "Beginning mobile training phase.")

            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self.num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            for i in range(self.num_rounds):

                self.current_round = i
                if abort_signal.triggered:
                    return

                self.log_info(fl_ctx, f"Round {self.current_round} started.")
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self.initial_weights, private=True, sticky=True)
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_round, private=True, sticky=True)
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

                # Create train_task
                task_data = Shareable()
                task_data["weights"] = self.initial_weights
                task_data["task_done"] = self.current_round >= (self.num_rounds - 1)
                task_data.set_header(AppConstants.CURRENT_ROUND, self.current_round)
                task_data.set_header(AppConstants.NUM_ROUNDS, self.num_rounds)
                task_data.add_cookie(AppConstants.CONTRIBUTION_ROUND, self.current_round)

                train_task = Task(
                    name="train",
                    data=task_data,
                    result_received_cb=self.process_train_result,
                )

                self.broadcast_and_wait(
                    task=train_task,
                    min_responses=1,
                    wait_time_after_min_received=30,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                if abort_signal.triggered:
                    return

                self.log_info(fl_ctx, "Start aggregation.")
                self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
                aggr_result = self.aggregator.aggregate(fl_ctx)
                self.log_info(fl_ctx, f"Aggregation result: {aggr_result}")
                fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
                self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
                self.log_info(fl_ctx, "End aggregation.")

                if abort_signal.triggered:
                    return

            final_weights = aggr_result.get("weights", None)
            self.log_info(fl_ctx, f"Finished Mobile Training. Final weights: {final_weights}")
        except Exception as e:
            error_msg = f"Exception in mobile control_flow: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def process_train_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        result = client_task.result
        client_name = client_task.client.name
        rc = result.get_return_code()

        # Raise errors if bad peer context or execution exception.
        if rc and rc != ReturnCode.OK:
            self.system_panic(
                f"Result from {client_name} is bad, error code: {rc}. "
                f"{self.__class__.__name__} exiting at round {self.current_round}.",
                fl_ctx=fl_ctx,
            )

            return

        self.log_info(fl_ctx, f"Weights: {result.get('weights', None)}")

        accepted = self.aggregator.accept(result, fl_ctx)
        accepted_msg = "ACCEPTED" if accepted else "REJECTED"
        self.log_info(
            fl_ctx, f"Contribution from {client_name} {accepted_msg} by the aggregator at round {self.current_round}."
        )
