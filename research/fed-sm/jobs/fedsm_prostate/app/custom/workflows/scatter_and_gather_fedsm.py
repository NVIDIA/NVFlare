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
import gc
import time
import traceback

from nvflare.apis.controller_spec import Task, TaskCompletionStatus
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather

_TASK_KEY_DONE = "___done"


class ScatterAndGatherFedSM(ScatterAndGather):
    def __init__(
        self,
        client_id_label_mapping,
        parallel_task: int = 1,
        min_clients: int = 1,
        num_rounds: int = 5,
        start_round: int = 0,
        wait_time_after_min_received: int = 10,
        aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID,
        persistor_id="persistor_fedsm",
        shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
        train_task_name=AppConstants.TASK_TRAIN,
        train_timeout: int = 0,
        ignore_result_error: bool = True,
    ):
        """FedSM Workflow. The ScatterAndGatherFedSM workflow defines federated training on all clients.
        FedSM involves training, aggregating, and persisting three types of models:
        - selector model: FL the same as regular FedAvg
        - global model: FL the same as regular FedAvg
        - personalized models: one for each candidate site, FL aggregation with SoftPull
        client_id_label_mapping is needed for training selector model
        All models are combined to be persisted by a single persistor (as required by NVFlare framework)
        in order to load/save the initial models sent to clients according to the model and client IDs.
        - persistor_fedsm is customized for persisting FedSM model set (a dict of models) for all clients
        Each client sends it's updated three weights after local training, to be aggregated accordingly by aggregator,
        we use one customized aggregator to handle all three models,
        global and selector models following standard weighted average, while personalized models following SoftPull
        The shareable generator is used to convert the aggregated weights to shareable, and shareable back to weights.

        Args:
            client_id_label_mapping: needed for training selector model, no Default.
            parallel_task (int, optional): sequential or parallel task. Defaults to 1 (parallel).
            min_clients (int, optional): Min number of clients in training. Defaults to 1.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            start_round (int, optional): Start round for training. Defaults to 0.
            wait_time_after_min_received (int, optional): Time to wait before beginning aggregation after contributions received. Defaults to 10.
            train_timeout (int, optional): Time to wait for clients to do local training.
            aggregator_id (str, optional): ID of the aggregator component for FedSM models. Defaults to "aggregator".
            persistor_id_fedsm (str, optional): ID of the persistor component for FedSM models. Defaults to "persistor_fedsm".
            shareable_generator_id (str, optional): ID of the shareable generator. Defaults to "shareable_generator".
            train_task_name (str, optional): Name of the train task. Defaults to "train".
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
        )
        # extras for FedSM
        # client_id to label mapping for selector
        self.parallel_task = parallel_task
        self.client_id_label_mapping = client_id_label_mapping
        self.participating_clients = None

    def start_controller(self, fl_ctx: FLContext) -> None:
        super().start_controller(fl_ctx=fl_ctx)
        self.log_info(fl_ctx, "Initializing FedSM-specific workflow components.")
        self.log_info(
            fl_ctx,
            f"Client_ID to selector label mapping: {self.client_id_label_mapping}",
        )
        # get engine
        engine = fl_ctx.get_engine()
        if not engine:
            self.system_panic("Engine not found. ScatterAndGather exiting.", fl_ctx)
            return

        # Get all clients
        clients = engine.get_clients()
        self.participating_clients = [c.name for c in clients]
        self.log_info(fl_ctx, f"Participating clients: {self.participating_clients}")

        # Validate client info
        for client_id in self.participating_clients:
            if client_id not in self.client_id_label_mapping.keys():
                self.system_panic(
                    f"Client {client_id} not found in the id_label mapping. Please double check. ScatterAndGatherFedSM exiting.",
                    fl_ctx,
                )
                return

    def _wait_for_task(self, task: Task, abort_signal: Signal):
        task.props[_TASK_KEY_DONE] = False
        task.task_done_cb = self._process_finished_task(task=task, func=task.task_done_cb)
        while True:
            if task.completion_status is not None:
                break
            if abort_signal and abort_signal.triggered:
                self.cancel_task(task, fl_ctx=None, completion_status=TaskCompletionStatus.ABORTED)
                break

            task_done = task.props[_TASK_KEY_DONE]
            if task_done:
                break
            time.sleep(self._task_check_period)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:
            self.log_info(fl_ctx, "Beginning ScatterAndGatherFedSM training phase.")
            self._phase = AppConstants.PHASE_TRAIN

            fl_ctx.set_prop(AppConstants.PHASE, self._phase, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            for self._current_round in range(self._start_round, self._start_round + self._num_rounds):
                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, f"Round {self._current_round} started.")
                self.log_info(fl_ctx, f"Models in fl_ctx: {self._global_weights['weights'].keys()}")

                fl_ctx.set_prop(
                    AppConstants.GLOBAL_MODEL,
                    self._global_weights,
                    private=True,
                    sticky=True,
                )
                fl_ctx.set_prop(
                    AppConstants.CURRENT_ROUND,
                    self._current_round,
                    private=True,
                    sticky=False,
                )
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

                # Create train_task for each participating clients, 3 models for each
                tasks_each_round = []
                for client_id in self.participating_clients:
                    # get the models for a client
                    select_weight = self._global_weights["weights"]["select_weights"]
                    global_weight = self._global_weights["weights"]["global_weights"]
                    client_weight = self._global_weights["weights"][client_id]
                    client_label = self.client_id_label_mapping[client_id]

                    # add all three models using a DXO collection
                    dxo_select_weights = DXO(data_kind=DataKind.WEIGHT_DIFF, data=select_weight)
                    dxo_global_weights = DXO(data_kind=DataKind.WEIGHT_DIFF, data=global_weight)
                    dxo_person_weights = DXO(data_kind=DataKind.WEIGHTS, data=client_weight)
                    # add Adam parameter sets
                    if self._current_round > 0:
                        select_exp_avg = self._global_weights["weights"]["select_exp_avg"]
                        select_exp_avg_sq = self._global_weights["weights"]["select_exp_avg_sq"]
                        dxo_select_exp_avg = DXO(data_kind=DataKind.WEIGHTS, data=select_exp_avg)
                        dxo_select_exp_avg_sq = DXO(data_kind=DataKind.WEIGHTS, data=select_exp_avg_sq)
                    else:
                        dxo_select_exp_avg = None
                        dxo_select_exp_avg_sq = None
                    # create dxo for client
                    dxo_dict = {
                        # add selector model weights and Adam parameters
                        "select_weights": dxo_select_weights,
                        "select_exp_avg": dxo_select_exp_avg,
                        "select_exp_avg_sq": dxo_select_exp_avg_sq,
                        # add global and personalized model weights
                        "global_weights": dxo_global_weights,
                        "person_weights": dxo_person_weights,
                        # add target id info for checking at client end
                        "target_id": client_id,
                        # add target label for client end selector training
                        "select_label": client_label,
                    }
                    dxo_collection = DXO(data_kind=DataKind.COLLECTION, data=dxo_dict)
                    data_shareable = dxo_collection.to_shareable()

                    # add meta information
                    data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
                    data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
                    data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

                    # create task
                    train_task = Task(
                        name=self.train_task_name,
                        data=data_shareable,
                        props={},
                        timeout=self._train_timeout,
                        before_task_sent_cb=self._prepare_train_task_data,
                        result_received_cb=self._process_train_result,
                    )

                    # send only to the target client
                    if self.parallel_task:
                        # tasks send in parallel
                        self.send(
                            task=train_task,
                            targets=[client_id],
                            fl_ctx=fl_ctx,
                        )
                        tasks_each_round.append(train_task)
                    else:
                        # tasks will be executed sequentially
                        self.send_and_wait(
                            task=train_task,
                            targets=[client_id],
                            fl_ctx=fl_ctx,
                        )
                    if self._check_abort_signal(fl_ctx, abort_signal):
                        return

                # wait for all tasks in this round to finish
                if self.parallel_task:
                    for task in tasks_each_round:
                        self._wait_for_task(task, abort_signal)

                # aggregate the returned results in shareable
                self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
                aggr_result = self.aggregator.aggregate(fl_ctx)
                collection_dxo = from_shareable(aggr_result)
                fl_ctx.set_prop(
                    AppConstants.AGGREGATION_RESULT,
                    aggr_result,
                    private=True,
                    sticky=False,
                )
                self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                # update all models using shareable generator for FedSM
                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                self._global_weights = self.shareable_gen.shareable_to_learnable(
                    shareable=collection_dxo.to_shareable(),
                    client_ids=self.participating_clients,
                    fl_ctx=fl_ctx,
                )

                # update models
                fl_ctx.set_prop(
                    AppConstants.GLOBAL_MODEL,
                    self._global_weights,
                    private=True,
                    sticky=True,
                )
                fl_ctx.sync_sticky()
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                self.persistor.save(self._global_weights["weights"], fl_ctx)
                self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(fl_ctx, f"Round {self._current_round} finished.")

                gc.collect()

            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, "Finished ScatterAndGatherFedSM Training.")
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Exception in ScatterAndGatherFedSM control_flow: {e}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(str(e), fl_ctx)
