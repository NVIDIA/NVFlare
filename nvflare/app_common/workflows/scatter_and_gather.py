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
import traceback

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


class ScatterAndGather(Controller):
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
    ):
        """The controller for ScatterAndGather Workflow.

        The ScatterAndGather workflow defines FederatedAveraging on all clients.
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

        Raises:
            TypeError: when any of input arguments does not have correct type
            ValueError: when any of input arguments is out of range
        """
        Controller.__init__(self)

        # Check arguments
        if not isinstance(min_clients, int):
            raise TypeError("min_clients must be int but got {}".format(type(min_clients)))
        if not isinstance(num_rounds, int):
            raise TypeError("num_rounds must be int but got {}".format(type(num_rounds)))
        if not isinstance(start_round, int):
            raise TypeError("start_round must be int but got {}".format(type(start_round)))
        if not isinstance(wait_time_after_min_received, int):
            raise TypeError(
                "wait_time_after_min_received must be int but got {}".format(type(wait_time_after_min_received))
            )
        if not isinstance(train_timeout, int):
            raise TypeError("train_timeout must be int but got {}".format(type(train_timeout)))
        if not isinstance(aggregator_id, str):
            raise TypeError("aggregator_id must be a string but got {}".format(type(aggregator_id)))
        if not isinstance(persistor_id, str):
            raise TypeError("persistor_id must be a string but got {}".format(type(persistor_id)))
        if not isinstance(shareable_generator_id, str):
            raise TypeError("shareable_generator_id must be a string but got {}".format(type(shareable_generator_id)))
        if not isinstance(train_task_name, str):
            raise TypeError("train_task_name must be a string but got {}".format(type(train_task_name)))
        if min_clients <= 0:
            raise ValueError("min_clients must be greater than 0.")
        if num_rounds < 0:
            raise ValueError("num_rounds must be greater than or equal to 0.")
        if start_round < 0:
            raise ValueError("start_round must be greater than or equal to 0.")
        if wait_time_after_min_received < 0:
            raise ValueError("wait_time_after_min_received must be greater than or equal to 0.")

        self.aggregator_id = aggregator_id
        self.persistor_id = persistor_id
        self.shareable_generator_id = shareable_generator_id
        self.train_task_name = train_task_name
        self.aggregator = None
        self.persistor = None
        self.shareable_gen = None

        # config data
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._wait_time_after_min_received = wait_time_after_min_received  # 5 minutes
        self._start_round = start_round
        self._train_timeout = train_timeout
        self.ignore_result_error = ignore_result_error

        # workflow phases: init, train, validate
        self._phase = AppConstants.PHASE_INIT
        self._global_weights = None
        self._current_round = None

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Initializing ScatterAndGather workflow.")
        self._phase = AppConstants.PHASE_INIT
        engine = fl_ctx.get_engine()
        if not engine:
            self.system_panic("Engine not found. ScatterAndGather exiting.", fl_ctx)
            return

        self.aggregator = engine.get_component(self.aggregator_id)
        if not isinstance(self.aggregator, Aggregator):
            self.system_panic(
                f"aggregator {self.aggregator_id} must be an Aggregator type object but got {type(self.aggregator)}",
                fl_ctx,
            )
            return

        self.shareable_gen = engine.get_component(self.shareable_generator_id)
        if not isinstance(self.shareable_gen, ShareableGenerator):
            self.system_panic(
                f"Shareable generator {self.shareable_generator_id} must be a ShareableGenerator type object, "
                f"but got {type(self.shareable_gen)}",
                fl_ctx,
            )
            return

        self.persistor = engine.get_component(self.persistor_id)
        if not isinstance(self.persistor, LearnablePersistor):
            self.system_panic(
                f"Model Persistor {self.persistor_id} must be a LearnablePersistor type object, "
                f"but got {type(self.persistor)}",
                fl_ctx,
            )
            return

        # initialize global model
        fl_ctx.set_prop(AppConstants.START_ROUND, self._start_round, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
        self._global_weights = self.persistor.load(fl_ctx)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
        self.fire_event(AppEventType.INITIAL_MODEL_LOADED, fl_ctx)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:

            self.log_info(fl_ctx, "Beginning ScatterAndGather training phase.")
            self._phase = AppConstants.PHASE_TRAIN

            fl_ctx.set_prop(AppConstants.PHASE, self._phase, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            # for self._current_round in range(self._start_round, self._start_round + self._num_rounds):
            if self._current_round is None:
                self._current_round = self._start_round
            while self._current_round < self._start_round + self._num_rounds:

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, f"Round {self._current_round} started.")
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=False)
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

                # Create train_task
                data_shareable: Shareable = self.shareable_gen.learnable_to_shareable(self._global_weights, fl_ctx)
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
                fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
                self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                self._global_weights = self.shareable_gen.shareable_to_learnable(aggr_result, fl_ctx)
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.sync_sticky()
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                self.persistor.save(self._global_weights, fl_ctx)
                self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(fl_ctx, f"Round {self._current_round} finished.")

                self._current_round += 1

                # Call the engine to persist the snapshot of all the FLComponents
                engine = fl_ctx.get_engine()
                engine.persist_components(fl_ctx, completed=False)

            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, "Finished ScatterAndGather Training.")
        except BaseException as e:
            traceback.print_exc()
            error_msg = f"Exception in ScatterAndGather control_flow: {e}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(str(e), fl_ctx)

    def stop_controller(self, fl_ctx: FLContext) -> None:
        self._phase = AppConstants.PHASE_FINISHED
        self.cancel_all_tasks()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == InfoCollector.EVENT_TYPE_GET_STATS:
            collector = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, None)
            if collector:
                if not isinstance(collector, GroupInfoCollector):
                    raise TypeError("collector must be GroupInfoCollector but got {}".format(type(collector)))

                collector.add_info(
                    group_name=self._name,
                    info={"phase": self._phase, "current_round": self._current_round, "num_rounds": self._num_rounds},
                )

    def _prepare_train_task_data(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        fl_ctx.set_prop(AppConstants.TRAIN_SHAREABLE, client_task.task.data, private=True, sticky=False)
        self.fire_event(AppEventType.BEFORE_TRAIN_TASK, fl_ctx)

    def _process_train_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        result = client_task.result
        client_name = client_task.client.name

        self._accept_train_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

        # Cleanup task result
        client_task.result = None

    def process_result_of_unknown_task(
        self, client: Client, task_name, client_task_id, result: Shareable, fl_ctx: FLContext
    ) -> None:
        if self._phase == AppConstants.PHASE_TRAIN and task_name == self.train_task_name:
            self._accept_train_result(client_name=client.name, result=result, fl_ctx=fl_ctx)
            self.log_info(fl_ctx, f"Result of unknown task {task_name} sent to aggregator.")
        else:
            self.log_error(fl_ctx, "Ignoring result from unknown task.")

    def _accept_train_result(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> bool:

        rc = result.get_return_code()
        contribution_round = result.get_cookie(AppConstants.CONTRIBUTION_ROUND)
        result.set_header(AppConstants.CONTRIBUTION_ROUND, contribution_round)

        # Raise errors if bad peer context or execution exception.
        if rc and rc != ReturnCode.OK:
            if self.ignore_result_error:
                self.log_error(fl_ctx, f"Ignore the client train result. Train result error code: {rc}")
                return False
            else:
                if rc in [ReturnCode.MISSING_PEER_CONTEXT, ReturnCode.BAD_PEER_CONTEXT]:
                    self.system_panic("Peer context is bad or missing. ScatterAndGather exiting.", fl_ctx=fl_ctx)
                    return False
                elif rc in [ReturnCode.EXECUTION_EXCEPTION, ReturnCode.TASK_UNKNOWN]:
                    self.system_panic(
                        "Execution Exception in client training. ScatterAndGather exiting.", fl_ctx=fl_ctx
                    )
                    return False
                elif rc in [
                    ReturnCode.EXECUTION_RESULT_ERROR,
                    ReturnCode.TASK_DATA_FILTER_ERROR,
                    ReturnCode.TASK_RESULT_FILTER_ERROR,
                ]:
                    self.system_panic("Execution result is not a shareable. ScatterAndGather exiting.", fl_ctx=fl_ctx)
                    return False

        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.CONTRIBUTION_ROUND, contribution_round, private=True, sticky=False)
        self.fire_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)

        accepted = self.aggregator.accept(result, fl_ctx)
        accepted_msg = "ACCEPTED" if accepted else "REJECTED"
        self.log_info(fl_ctx, f"Contribution from {client_name} {accepted_msg} by the aggregator.")

        fl_ctx.set_prop(AppConstants.AGGREGATION_ACCEPTED, accepted, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_CONTRIBUTION_ACCEPT, fl_ctx)

        return accepted

    def _check_abort_signal(self, fl_ctx, abort_signal: Signal):
        if abort_signal.triggered:
            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, f"Abort signal received. Exiting at round {self._current_round}.")
            return True
        return False

    def get_persist_state(self, fl_ctx: FLContext) -> dict:
        return {
            "current_round": self._current_round,
            "start_round": self._start_round,
            "num_rounds": self._num_rounds,
            "global_weights": self._global_weights,
        }

    def restore(self, state_data: dict, fl_ctx: FLContext):
        try:
            self._current_round = state_data.get("current_round")
            self._start_round = state_data.get("start_round")
            self._num_rounds = state_data.get("num_rounds")
            self._global_weights = state_data.get("global_weights")
        finally:
            pass
