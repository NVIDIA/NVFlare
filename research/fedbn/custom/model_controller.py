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
from abc import abstractmethod
from typing import List, Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import OperatorMethod, TaskOperatorKey
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_component_wrapper import FLComponentWrapper
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils.experimental import experimental
from nvflare.fuel.utils.validation_utils import check_non_negative_int, check_positive_int, check_str
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


@experimental
class ModelController(Controller, FLComponentWrapper):
    def __init__(
        self,
        min_clients: int = 1000,
        num_rounds: int = 5,
        persistor_id="",
        ignore_result_error: bool = False,
        allow_empty_global_weights: bool = False,
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
    ):
        """FLModel based controller.

        Args:
            min_clients (int, optional): The minimum number of clients responses before
                Workflow starts to wait for `wait_time_after_min_received`. Note that the workflow will move forward
                when all available clients have responded regardless of this value. Defaults to 1000.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".
            ignore_result_error (bool, optional): whether this controller can proceed if client result has errors.
                Defaults to False.
            allow_empty_global_weights (bool, optional): whether to allow empty global weights. Some pipelines can have
                empty global weights at first round, such that clients start training from scratch without any global info.
                Defaults to False.
            task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.5.
            persist_every_n_rounds (int, optional): persist the global model every n rounds. Defaults to 1.
                If n is 0 then no persist.
        """
        super().__init__(task_check_period=task_check_period)

        # Check arguments
        check_positive_int("min_clients", min_clients)
        check_non_negative_int("num_rounds", num_rounds)
        check_non_negative_int("persist_every_n_rounds", persist_every_n_rounds)
        check_str("persistor_id", persistor_id)
        if not isinstance(task_check_period, (int, float)):
            raise TypeError(f"task_check_period must be an int or float but got {type(task_check_period)}")
        elif task_check_period <= 0:
            raise ValueError("task_check_period must be greater than 0.")
        self._task_check_period = task_check_period
        self.persistor_id = persistor_id
        self.persistor = None

        # config data
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._persist_every_n_rounds = persist_every_n_rounds
        self.ignore_result_error = ignore_result_error
        self.allow_empty_global_weights = allow_empty_global_weights

        # workflow phases: init, train, validate
        self._phase = AppConstants.PHASE_INIT
        self._current_round = None

        # model related
        self.model = None
        self._results = []

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        self.info("Initializing ModelController workflow.")

        if self.persistor_id:
            self.persistor = self._engine.get_component(self.persistor_id)
            if not isinstance(self.persistor, LearnablePersistor):
                self.panic(
                    f"Model Persistor {self.persistor_id} must be a LearnablePersistor type object, "
                    f"but got {type(self.persistor)}"
                )
                return

        # initialize global model
        if self.persistor:
            global_weights = self.persistor.load(self.fl_ctx)

            if not isinstance(global_weights, ModelLearnable):
                self.panic(
                    f"Expected global weights to be of type `ModelLearnable` but received {type(global_weights)}"
                )
                return

            if global_weights.is_empty():
                if not self.allow_empty_global_weights:
                    # if empty not allowed, further check whether it is available from fl_ctx
                    global_weights = self.fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)

            if not global_weights.is_empty():
                self.model = FLModel(
                    params_type=ParamsType.FULL,
                    params=global_weights[ModelLearnableKey.WEIGHTS],
                    meta=global_weights[ModelLearnableKey.META],
                )
            elif self.allow_empty_global_weights:
                self.model = FLModel(params_type=ParamsType.FULL, params={})
            else:
                self.panic(
                    f"Neither `persistor` {self.persistor_id} or `fl_ctx` returned a global model! If this was intended, set `self.allow_empty_global_weights` to `True`."
                )
                return
        else:
            self.model = FLModel(params_type=ParamsType.FULL, params={})

        # persistor uses Learnable format to save model
        ml = make_model_learnable(weights=self.model.params, meta_props=self.model.meta)
        self.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, ml, private=True, sticky=True)
        self.event(AppEventType.INITIAL_MODEL_LOADED)

        self.engine = self.fl_ctx.get_engine()
        self.initialize()

    def _build_shareable(self, data: FLModel = None) -> Shareable:
        if not data:  # if no data is given, send self.model
            data = self.model

        data_shareable: Shareable = FLModelUtils.to_shareable(data)
        data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
        data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
        data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

        return data_shareable

    def send_model_and_wait(
        self,
        targets: Union[List[Client], List[str], None] = None,
        data: FLModel = None,
        task_name=AppConstants.TASK_TRAIN,
        timeout: int = 0,
        wait_time_after_min_received: int = 10,
    ) -> List:
        """Send the current global model or given data to a list of targets

        The task is scheduled into a task list.  Clients can request tasks and controller will dispatch the task to eligible clients.

        Args:
            targets: the list of eligible clients or client names or None (all clients). Defaults to None.
            data: FLModel to be sent to clients. If no data is given, send `self.model`.
            task_name (str, optional): Name of the train task. Defaults to "train".
            timeout (int, optional): Time to wait for clients to do local training. Defaults to 0, i.e., never time out.
            wait_time_after_min_received (int, optional): Time to wait before beginning aggregation after
                minimum number of clients responses has been received. Defaults to 10.
        """

        if not isinstance(task_name, str):
            raise TypeError("train_task_name must be a string but got {}".format(type(task_name)))
        check_non_negative_int("timeout", timeout)
        check_non_negative_int("wait_time_after_min_received", wait_time_after_min_received)

        # Create train_task
        data_shareable = self._build_shareable(data)

        operator = {
            TaskOperatorKey.OP_ID: task_name,
            TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
            TaskOperatorKey.TIMEOUT: timeout,
        }

        train_task = Task(
            name=task_name,
            data=data_shareable,
            operator=operator,
            props={},
            timeout=timeout,
            before_task_sent_cb=self._prepare_task_data,
            result_received_cb=self._process_result,
        )

        self._results = []  # reset results list
        self.info(f"Sending task {task_name} to {[client.name for client in targets]}")
        self.broadcast_and_wait(
            task=train_task,
            targets=targets,
            min_responses=self._min_clients,
            wait_time_after_min_received=wait_time_after_min_received,
            fl_ctx=self.fl_ctx,
            abort_signal=self.abort_signal,
        )

        if targets is not None:
            if len(self._results) != self._min_clients:
                self.warning(
                    f"Number of results ({len(self._results)}) is different from min_clients ({self._min_clients})."
                )

        # de-refernce the internel results before returning
        results = self._results
        self._results = []
        return results

    def _prepare_task_data(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        fl_ctx.set_prop(AppConstants.TRAIN_SHAREABLE, client_task.task.data, private=True, sticky=False)
        self.event(AppEventType.BEFORE_TRAIN_TASK)

    def _process_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        result = client_task.result
        client_name = client_task.client.name

        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)

        self.event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT)
        self._accept_train_result(client_name=client_name, result=result, fl_ctx=fl_ctx)
        self.event(AppEventType.AFTER_CONTRIBUTION_ACCEPT)

        # Turn result into FLModel
        result_model = FLModelUtils.from_shareable(result)
        result_model.meta["client_name"] = client_name
        result_model.meta["current_round"] = self._current_round
        result_model.meta["total_rounds"] = self._num_rounds

        self._results.append(result_model)

        # Cleanup task result
        client_task.result = None

        gc.collect()

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ) -> None:
        if self._phase == AppConstants.PHASE_TRAIN and task_name == task_name:
            self._accept_train_result(client_name=client.name, result=result, fl_ctx=fl_ctx)
            self.info(f"Result of unknown task {task_name} sent to aggregator.")
        else:
            self.error("Ignoring result from unknown task.")

    def _accept_train_result(self, client_name: str, result: Shareable, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        rc = result.get_return_code()

        # Raise panic if bad peer context or execution exception.
        if rc and rc != ReturnCode.OK:
            if self.ignore_result_error:
                self.warning(
                    f"Ignore the train result from {client_name} at round {self._current_round}. Train result error code: {rc}",
                )
            else:
                self.panic(
                    f"Result from {client_name} is bad, error code: {rc}. "
                    f"{self.__class__.__name__} exiting at round {self._current_round}."
                )
                return

        self.fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)

    @abstractmethod
    def run(self):
        """Main `run` routine called by the Controller's `control_flow` to execute the workflow.

        Returns: None.

        """
        raise NotImplementedError

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        self._phase = AppConstants.PHASE_TRAIN
        fl_ctx.set_prop(AppConstants.PHASE, self._phase, private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
        self.fl_ctx = fl_ctx
        self.abort_signal = abort_signal
        try:
            self.info("Beginning model controller run.")
            self.event(AppEventType.TRAINING_STARTED)
            self._phase = AppConstants.PHASE_TRAIN

            self.run()
            self._phase = AppConstants.PHASE_FINISHED
        except Exception as e:
            error_msg = f"Exception in model controller run: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)

    def save_model(self):
        if self.persistor:
            if (
                self._persist_every_n_rounds != 0 and (self._current_round + 1) % self._persist_every_n_rounds == 0
            ) or self._current_round == self._num_rounds - 1:
                self.info("Start persist model on server.")
                self.event(AppEventType.BEFORE_LEARNABLE_PERSIST)
                # persistor uses Learnable format to save model
                ml = make_model_learnable(weights=self.model.params, meta_props=self.model.meta)
                self.persistor.save(ml, self.fl_ctx)
                self.event(AppEventType.AFTER_LEARNABLE_PERSIST)
                self.info("End persist model on server.")

    def stop_controller(self, fl_ctx: FLContext):
        self._phase = AppConstants.PHASE_FINISHED
        self.fl_ctx = fl_ctx
        self.finalize()

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
