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

import random
from abc import ABC, abstractmethod
from typing import List, Union

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import OperatorMethod, TaskOperatorKey
from nvflare.apis.fl_component import FLComponentHelper
from nvflare.apis.fl_constant import FLMetaKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector

from .scatter_and_gather import _check_non_neg_int


class FedAvgModelController(Controller, FLComponentHelper, ABC):
    def __init__(
        self,
        min_clients: int = 1000,
        num_rounds: int = 5,
        wait_time_after_min_received: int = 10,
        persistor_id="",
        train_task_name=AppConstants.TASK_TRAIN,
        train_timeout: int = 0,
        ignore_result_error: bool = False,
        allow_empty_global_weights: bool = False,
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
    ):
        """The base controller for FedAvg Workflow. *Note*: This class is experimental.

        Implements [FederatedAveraging](https://arxiv.org/abs/1602.05629).
        The model persistor (persistor_id) is used to load the initial global model which is sent to a list of clients.
        Each client sends it's updated weights after local training which is aggregated.
        Next, the global model is updated.
        The model_persistor also saves the model after training.

        The below abstract routines need to be implemented by the derived classes.

            - def sample_clients(self, min_clients)
            - def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel
            - def update_model(self, aggr_result)
            - def run(self)

        Args:
            min_clients (int, optional): The minimum number of clients responses before
                Workflow starts to wait for `wait_time_after_min_received`. Note that the workflow will move forward
                when all available clients have responded regardless of this value. Defaults to 1000.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            wait_time_after_min_received (int, optional): Time to wait before beginning aggregation after
                minimum number of clients responses has been received. Defaults to 10.
            persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".
            train_task_name (str, optional): Name of the train task. Defaults to "train".
            train_timeout (int, optional): Time to wait for clients to do local training.
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
        if not isinstance(min_clients, int):
            raise TypeError("min_clients must be int but got {}".format(type(min_clients)))
        elif min_clients <= 0:
            raise ValueError("min_clients must be greater than 0.")

        _check_non_neg_int(num_rounds, "num_rounds")
        _check_non_neg_int(wait_time_after_min_received, "wait_time_after_min_received")
        _check_non_neg_int(train_timeout, "train_timeout")
        _check_non_neg_int(persist_every_n_rounds, "persist_every_n_rounds")

        if not isinstance(persistor_id, str):
            raise TypeError("persistor_id must be a string but got {}".format(type(persistor_id)))
        if not isinstance(train_task_name, str):
            raise TypeError("train_task_name must be a string but got {}".format(type(train_task_name)))

        if not isinstance(task_check_period, (int, float)):
            raise TypeError(f"task_check_period must be an int or float but got {type(task_check_period)}")
        elif task_check_period <= 0:
            raise ValueError("task_check_period must be greater than 0.")

        self.persistor_id = persistor_id
        self.train_task_name = train_task_name
        self.persistor = None

        # config data
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._wait_time_after_min_received = wait_time_after_min_received
        self._train_timeout = train_timeout
        self._persist_every_n_rounds = persist_every_n_rounds
        self.ignore_result_error = ignore_result_error
        self.allow_empty_global_weights = allow_empty_global_weights

        # workflow phases: init, train, validate
        self._phase = AppConstants.PHASE_INIT
        self._current_round = None

        self.model = None
        self._results = []

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        self.info("Initializing ScatterAndGather workflow.")
        self._phase = AppConstants.PHASE_INIT

        if self.persistor_id:
            self.persistor = self._engine.get_component(self.persistor_id)
            if not isinstance(self.persistor, LearnablePersistor):
                self.panic(
                    f"Model Persistor {self.persistor_id} must be a LearnablePersistor type object, "
                    f"but got {type(self.persistor)}"
                )
                return

        # initialize global model
        self.fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
        if self.persistor:
            global_weights = self.persistor.load(self.fl_ctx)
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
        else:
            self.model = FLModel(params_type=ParamsType.FULL, params={})

        self.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self.model, private=True, sticky=True)
        self.event(AppEventType.INITIAL_MODEL_LOADED)

        self.engine = self.fl_ctx.get_engine()
        self.initialize()

    def send_model_and_wait(self, targets: Union[List[Client], List[str], None] = None, data: FLModel = None) -> List:
        # Create train_task
        data_shareable: Shareable = FLModelUtils.to_shareable(data)
        data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
        data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
        data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

        operator = {
            TaskOperatorKey.OP_ID: self.train_task_name,
            TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
            TaskOperatorKey.TIMEOUT: self._train_timeout,
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

        self.info(f"Sending train task to {[client.name for client in targets]}")
        self.broadcast_and_wait(
            task=train_task,
            targets=targets,
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=self.fl_ctx,
            abort_signal=self.abort_signal,
        )

        if len(self._results) != self._min_clients:
            self.warning(
                f"Number of results ({len(self._results)}) is different from min_clients ({self._min_clients})."
            )

        return self._results

    def _prepare_train_task_data(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        fl_ctx.set_prop(AppConstants.TRAIN_SHAREABLE, client_task.task.data, private=True, sticky=False)
        self.event(AppEventType.BEFORE_TRAIN_TASK)

    def _process_train_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        result = client_task.result
        client_name = client_task.client.name

        self._accept_train_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

        # Turn result into FLModel
        result_model = FLModelUtils.from_shareable(result)
        result_model.meta["client_name"] = client_name
        result_model.meta["current_round"] = self._current_round
        result_model.meta["total_rounds"] = self._num_rounds

        self._results.append(result_model)

        # Cleanup task result
        client_task.result = None

    def process_result_of_unknown_task(
        self, client: Client, task_name, client_task_id, result: Shareable, fl_ctx: FLContext
    ) -> None:
        if self._phase == AppConstants.PHASE_TRAIN and task_name == self.train_task_name:
            self._accept_train_result(client_name=client.name, result=result, fl_ctx=fl_ctx)
            self.info(f"Result of unknown task {task_name} sent to aggregator.")
        else:
            self.error("Ignoring result from unknown task.")

    def _accept_train_result(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> bool:
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

        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)
        self.fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)

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
                # Replace: self.persistor.save(global_weights, fl_ctx)
                self.persistor.save(self.model, self.fl_ctx)
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

    # To be implemented by derived classes
    @abstractmethod
    def sample_clients(self, min_clients):
        """Called by the `run` routine to get a list of available clients.

        Args:
            min_clients: number of clients to return.

        Returns: list of clients.

        """
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel:
        """Called by the `run` routine to aggregate the training results of clients.

        Args:
            results: a list of FLModel containing training results of the clients.
            aggregate_fn: a function that turns the list of FLModel into one resulting (aggregated) FLModel.

        Returns: aggregated FLModel.

        """
        raise NotImplementedError

    @abstractmethod
    def update_model(self, aggr_result):
        """Called by the `run` routine to update the current global model (self.model) given the aggregated result.

        Args:
            aggr_result: aggregated FLModel.

        Returns: None.

        """
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """Main `run` routine called by the Controller's `control_flow` to execute the workflow.

        Returns: None.

        """
        raise NotImplementedError


class BaseFedAvg(FedAvgModelController, ABC):
    """Controller for FedAvg Workflow. *Note*: This class is experimental.
    Implements [FederatedAveraging](https://arxiv.org/abs/1602.05629).

    Provides the default implementations for the follow routines:
        - def sample_clients(self, min_clients)
        - def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel
        - def update_model(self, aggr_result)

    The `run` routine still needs to be implemented by the derived class:

        - def run(self)
    """

    def sample_clients(self, min_clients):
        self._min_clients = min_clients

        clients = self.engine.get_clients()
        random.shuffle(clients)

        if len(clients) < self._min_clients:
            self._min_clients = len(clients)

        clients = clients[0 : self._min_clients]

        return clients

    @staticmethod
    def _check_results(results: List[FLModel]):
        empty_clients = []
        for _result in results:
            if not _result.params:
                empty_clients.append(_result.meta.get("client_name", "unkown"))

        if len(empty_clients) > 0:
            raise ValueError(f"Result from client(s) {empty_clients} is empty!")

    @staticmethod
    def _aggregate_fn(results: List[FLModel]) -> FLModel:
        aggregation_helper = WeightedAggregationHelper()
        for _result in results:
            aggregation_helper.add(
                data=_result.params,
                weight=_result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
                contributor_name=_result.meta.get("client_name", "unkown"),
                contribution_round=_result.meta.get("current_round", None),
            )

        aggregated_dict = aggregation_helper.get_result()

        aggr_result = FLModel(
            params=aggregated_dict,
            params_type=results[0].params_type,
            meta={"nr_aggregated": len(results), "current_round": results[0].meta["current_round"]},
        )
        return aggr_result

    def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel:
        self.debug("Start aggregation.")
        self.event(AppEventType.BEFORE_AGGREGATION)
        self._check_results(results)

        if not aggregate_fn:
            aggregate_fn = self._aggregate_fn

        self.info(f"aggregating {len(results)} update(s) at round {self._current_round}")
        aggr_result = aggregate_fn(results)
        self._results = []

        self.fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
        self.event(AppEventType.AFTER_AGGREGATION)
        self.debug("End aggregation.")

        return aggr_result

    def update_model(self, aggr_result):
        self.event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE)

        self.model.meta = aggr_result.meta
        if aggr_result.params_type == ParamsType.FULL:
            self.model.params = aggr_result.params
        elif aggr_result.params_type == ParamsType.DIFF:
            for v_name, v_value in aggr_result.params.items():
                self.model.params[v_name] = self.model.params[v_name] + v_value
        else:
            raise RuntimeError(f"params_type {aggr_result.params_type} not supported!")

        self.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self.model, private=True, sticky=True)
        self.fl_ctx.sync_sticky()
        self.event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE)
