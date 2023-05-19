# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import copy

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Task
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.app_constant import AlgorithmConstants, AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.security.logging import secure_format_exception


class ScatterAndGatherScaffold(ScatterAndGather):
    def __init__(
        self,
        min_clients: int = 1,
        num_rounds: int = 5,
        start_round: int = 0,
        wait_time_after_min_received: int = 10,
        aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID,
        persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
        shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
        train_task_name=AppConstants.TASK_TRAIN,
        train_timeout: int = 0,
        ignore_result_error: bool = False,
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
        snapshot_every_n_rounds: int = 1,
    ):
        """The controller for ScatterAndGatherScaffold workflow.

        The model persistor (persistor_id) is used to load the initial global model which is sent to all clients.
        Each client sends it's updated weights after local training which is aggregated (aggregator_id). The
        shareable generator is used to convert the aggregated weights to shareable and shareable back to weight.
        The model_persistor also saves the model after training.

        Args:
            min_clients (int, optional): Min number of clients in training. Defaults to 1.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            start_round (int, optional): Start round for training. Defaults to 0.
            wait_time_after_min_received (int, optional): Time to wait before beginning aggregation after
                contributions received. Defaults to 10.
            aggregator_id (str, optional): ID of the aggregator component. Defaults to "aggregator".
            persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".
            shareable_generator_id (str, optional): ID of the shareable generator. Defaults to "shareable_generator".
            train_task_name (str, optional): Name of the train task. Defaults to "train".
            train_timeout (int, optional): Time to wait for clients to do local training.
            ignore_result_error (bool, optional): whether this controller can proceed if client result has errors.
                Defaults to False.
            task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.5.
            persist_every_n_rounds (int, optional): persist the global model every n rounds. Defaults to 1.
                If n is 0 then no persist.
            snapshot_every_n_rounds (int, optional): persist the server state every n rounds. Defaults to 1.
                If n is 0 then no persist.
        """

        super().__init__(
            min_clients=min_clients,
            num_rounds=num_rounds,
            start_round=start_round,
            wait_time_after_min_received=wait_time_after_min_received,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            train_task_name=train_task_name,
            train_timeout=train_timeout,
            ignore_result_error=ignore_result_error,
            task_check_period=task_check_period,
            persist_every_n_rounds=persist_every_n_rounds,
            snapshot_every_n_rounds=snapshot_every_n_rounds,
        )

        # for SCAFFOLD
        self.aggregator_ctrl = None
        self._global_ctrl_weights = None

    def start_controller(self, fl_ctx: FLContext) -> None:
        super().start_controller(fl_ctx=fl_ctx)
        self.log_info(fl_ctx, "Initializing ScatterAndGatherScaffold workflow.")

        # for SCAFFOLD
        if not self._global_weights:
            self.system_panic("Global weights not available!", fl_ctx)
            return

        self._global_ctrl_weights = copy.deepcopy(self._global_weights["weights"])
        # Initialize correction term with zeros
        for k in self._global_ctrl_weights.keys():
            self._global_ctrl_weights[k] = np.zeros_like(self._global_ctrl_weights[k])
        # TODO: Print some stats of the correction magnitudes

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:

            self.log_info(fl_ctx, "Beginning ScatterAndGatherScaffold training phase.")
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
                # get DXO with global model weights
                dxo_global_weights = model_learnable_to_dxo(self._global_weights)

                # add global SCAFFOLD controls using a DXO collection
                dxo_global_ctrl = DXO(data_kind=DataKind.WEIGHT_DIFF, data=self._global_ctrl_weights)
                dxo_dict = {
                    AppConstants.MODEL_WEIGHTS: dxo_global_weights,
                    AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: dxo_global_ctrl,
                }
                dxo_collection = DXO(data_kind=DataKind.COLLECTION, data=dxo_dict)
                data_shareable = dxo_collection.to_shareable()

                # add meta information
                data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
                data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
                data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

                train_task = Task(
                    name=self.train_task_name,
                    data=data_shareable,
                    props={},
                    timeout=self._train_timeout,
                    before_task_sent_cb=self._prepare_train_task_data,
                    result_received_cb=self._process_train_result,
                )

                self.broadcast_and_wait(
                    task=train_task,
                    min_responses=self._min_clients,
                    wait_time_after_min_received=self._wait_time_after_min_received,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
                aggr_result = self.aggregator.aggregate(fl_ctx)

                # extract aggregated weights and controls
                collection_dxo = from_shareable(aggr_result)
                dxo_aggr_result = collection_dxo.data.get(AppConstants.MODEL_WEIGHTS)
                if not dxo_aggr_result:
                    self.log_error(fl_ctx, "Aggregated model weights are missing!")
                    return
                dxo_ctrl_aggr_result = collection_dxo.data.get(AlgorithmConstants.SCAFFOLD_CTRL_DIFF)
                if not dxo_ctrl_aggr_result:
                    self.log_error(fl_ctx, "Aggregated model weight controls are missing!")
                    return

                fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
                self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                # update global model using shareable generator
                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                self._global_weights = self.shareable_gen.shareable_to_learnable(dxo_aggr_result.to_shareable(), fl_ctx)

                # update SCAFFOLD global controls
                ctr_diff = dxo_ctrl_aggr_result.data
                for v_name, v_value in ctr_diff.items():
                    self._global_ctrl_weights[v_name] += v_value
                fl_ctx.set_prop(
                    AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL, self._global_ctrl_weights, private=True, sticky=True
                )

                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.sync_sticky()
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                if self._persist_every_n_rounds != 0 and (self._current_round + 1) % self._persist_every_n_rounds == 0:
                    self.log_info(fl_ctx, "Start persist model on server.")
                    self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                    self.persistor.save(self._global_weights, fl_ctx)
                    self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)
                    self.log_info(fl_ctx, "End persist model on server.")

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(fl_ctx, f"Round {self._current_round} finished.")
                self._current_round += 1

                # need to persist snapshot after round increased because the global weights should be set to
                # the last finished round's result
                if self._snapshot_every_n_rounds != 0 and self._current_round % self._snapshot_every_n_rounds == 0:
                    self._engine.persist_components(fl_ctx, completed=False)

            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, "Finished ScatterAndGatherScaffold Training.")
        except Exception as e:
            error_msg = f"Exception in ScatterAndGatherScaffold control_flow: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)
