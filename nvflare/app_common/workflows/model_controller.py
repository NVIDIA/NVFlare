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
from abc import abstractmethod, ABC
from typing import Union, List

from nvflare.apis.fl_component import FLComponentHelper
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.apis.client import Client
from nvflare.apis.controller_spec import OperatorMethod, TaskOperatorKey
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.apis.impl.controller import ClientTask
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.security.logging import secure_format_exception
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.apis.fl_constant import FLMetaKey, ReturnCode
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

from .scatter_and_gather import ScatterAndGather


class BaseModelController(ScatterAndGather, FLComponentHelper, ABC):
    def __init__(self, **kwargs):
        self.model = None
        self.aggregator = None
        self.shareable_gen = None
        self.results = []

        super().__init__(**kwargs)

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
        self.fl_ctx.set_prop(AppConstants.START_ROUND, self._start_round, private=True, sticky=True)
        self.fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
        if self.persistor:
            self._global_weights = self.persistor.load(self.fl_ctx)
            if self._global_weights.is_empty():
                if not self.allow_empty_global_weights:
                    # if empty not allowed, further check whether it is available from fl_ctx
                    self._global_weights = self.fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)

            if not self._global_weights.is_empty():
                self.model = FLModel(
                    params_type=ParamsType.FULL,
                    params=self._global_weights[ModelLearnableKey.WEIGHTS],
                    meta=self._global_weights[ModelLearnableKey.META]
                )
            else:
                self.model = FLModel(
                    params_type=ParamsType.FULL,
                    params={}
                )

            self.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
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

        self.info(f"Sending train task to {[client.name for client in targets]}")
        self.broadcast_and_wait(
            task=train_task,
            targets=targets,
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            fl_ctx=self.fl_ctx,
            abort_signal=self.abort_signal,
        )

        if len(self.results) != self._min_clients:
            self.warning(f"Number of results ({len(self.results)}) is different from min_clients ({self._min_clients}).")

        return self.results

    def _process_train_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        result = client_task.result
        client_name = client_task.client.name

        self._accept_train_result(client_name=client_name, result=result, fl_ctx=fl_ctx)

        # Cleanup task result
        client_task.result = None

        # Turn result into FLModel
        result_model = FLModelUtils.from_shareable(result)
        result_model.meta["client_name"] = client_name
        result_model.meta["current_round"] = self._current_round
        result_model.meta["total_rounds"] = self._num_rounds

        self.results.append(result_model)

    def _accept_train_result(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> bool:
        self.fl_ctx = fl_ctx
        rc = result.get_return_code()

        # Raise errors if bad peer context or execution exception.
        if rc and rc != ReturnCode.OK:
            if self.ignore_result_error:
                self.warning(
                    f"Ignore the train result from {client_name} at round {self._current_round}. Train result error code: {rc}",
                )
                return False
            else:
                self.panic(
                    f"Result from {client_name} is bad, error code: {rc}. "
                    f"{self.__class__.__name__} exiting at round {self._current_round}."
                )
                return False

        self.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)
        self.fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)

        return True

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        self.fl_ctx = fl_ctx
        self.abort_signal = abort_signal
        try:
            self.info("Beginning model controller run.")
            self._phase = AppConstants.PHASE_TRAIN

            self.run()
        except Exception as e:
            error_msg = f"Exception in model controller run: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)

    def save_model(self):
        if self.persistor:
            if (
                    self._persist_every_n_rounds != 0
                    and (self._current_round + 1) % self._persist_every_n_rounds == 0
            ) or self._current_round == self._start_round + self._num_rounds - 1:
                self.info("Start persist model on server.")
                self.event(AppEventType.BEFORE_LEARNABLE_PERSIST)
                # Replace: self.persistor.save(self._global_weights, fl_ctx)
                self.persistor.save(self.model, self.fl_ctx)
                self.event(AppEventType.AFTER_LEARNABLE_PERSIST)
                self.info("End persist model on server.")

    def stop_controller(self, fl_ctx: FLContext):
        super().stop_controller(fl_ctx)
        self.fl_ctx = fl_ctx
        self.finalize()

    # To be implemented by derived classes
    @abstractmethod
    def sample_clients(self, min_clients):
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel:
        raise NotImplementedError

    @abstractmethod
    def update_model(self, aggr_result):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError


class ModelController(BaseModelController, ABC):

    def sample_clients(self, min_clients):
        self._min_clients = min_clients

        clients = self.engine.get_clients()
        random.shuffle(clients)

        if len(clients) < self._min_clients:
            self._min_clients = len(clients)

        clients = clients[0:self._min_clients]

        return clients

    @staticmethod
    def _aggregate_fn(results: List[FLModel]) -> FLModel:
        aggregation_helper = WeightedAggregationHelper()
        for _result in results:
            aggregation_helper.add(data=_result.params,
                                   weight=_result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
                                   contributor_name=_result.meta.get("client_name", "unkown"),
                                   contribution_round=_result.meta.get("current_round", None))

        aggregated_dict = aggregation_helper.get_result()
        aggregation_helper.reset_stats()

        aggr_result = FLModel(
            params=aggregated_dict,
            params_type=results[0].params_type,
            meta={"nr_aggregated": len(results), "current_round": results[0].meta["current_round"]}
        )
        return aggr_result

    def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel:
        # TODO: write separate FLModel aggregator? Check params_type for each result: add _check_results()
        self.info("Start aggregation.")
        self.event(AppEventType.BEFORE_AGGREGATION)
        # Replaces: aggr_result = self.aggregator.aggregate(self.fl_ctx)

        # TODO: self._check_results(results), params_type, current_round

        if not aggregate_fn:
            aggregate_fn = self._aggregate_fn

        self.info(f"aggregating {len(results)} update(s) at round {self._current_round}")
        aggr_result = aggregate_fn(results)
        self.results = []

        self.fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
        self.event(AppEventType.AFTER_AGGREGATION)
        self.info("End aggregation.")

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
