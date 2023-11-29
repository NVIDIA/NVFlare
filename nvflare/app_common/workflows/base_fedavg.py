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
from abc import abstractmethod
from typing import List

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.security.logging import secure_format_exception
from nvflare.fuel.utils.experimental import experimental

from .model_controller import ModelController


@experimental
class FedAvgModelControllerSpec(ModelController):
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
    """
    def __int__(self):
        super().__int__()

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


@experimental
class BaseFedAvg(FedAvgModelControllerSpec):
    """Controller for FedAvg Workflow. *Note*: This class is experimental.
    Implements [FederatedAveraging](https://arxiv.org/abs/1602.05629).

    Provides the default implementations for the follow routines:
        - def sample_clients(self, min_clients)
        - def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel
        - def update_model(self, aggr_result)

    The `run` routine needs to be implemented by the derived class:

        - def run(self)
    """
    def __int__(self):
        super().__int__()

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
        try:
            aggr_result = aggregate_fn(results)
        except Exception as e:
            error_msg = f"Exception in aggregate call: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return FLModel()
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

    def run(self):
        raise NotImplementedError
