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

from typing import List

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.security.logging import secure_format_exception

from .model_controller import ModelController


class BaseFedAvg(ModelController):
    def __init__(
        self,
        *args,
        num_clients: int = 3,
        num_rounds: int = 5,
        start_round: int = 0,
        **kwargs,
    ):
        """The base controller for FedAvg Workflow. *Note*: This class is based on the `ModelController`.

        Implements [FederatedAveraging](https://arxiv.org/abs/1602.05629).

        A model persistor can be configured via the `persistor_id` argument of the `ModelController`.
        The model persistor is used to load the initial global model which is sent to a list of clients.
        Each client sends it's updated weights after local training which is aggregated.
        Next, the global model is updated.
        The model_persistor will also save the model after training.

        Provides the default implementations for the follow routines:
            - def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel
            - def update_model(self, aggr_result)

        The `run` routine needs to be implemented by the derived class:

            - def run(self)

        Args:
            num_clients (int, optional): The number of clients. Defaults to 3.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            start_round (int, optional): The starting round number.
        """
        super().__init__(*args, **kwargs)

        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.start_round = start_round

        self.current_round = None

    @staticmethod
    def _check_results(results: List[FLModel]):
        empty_clients = []
        for _result in results:
            if not _result.params:
                empty_clients.append(_result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN))

        if len(empty_clients) > 0:
            raise ValueError(f"Result from client(s) {empty_clients} is empty!")

    @staticmethod
    def aggregate_fn(results: List[FLModel]) -> FLModel:
        if not results:
            raise ValueError("received empty results for aggregation.")

        aggr_helper = WeightedAggregationHelper()
        aggr_metrics_helper = WeightedAggregationHelper()
        all_metrics = True
        for _result in results:
            aggr_helper.add(
                data=_result.params,
                weight=_result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
                contributor_name=_result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
                contribution_round=_result.current_round,
            )
            if not _result.metrics:
                all_metrics = False
            if all_metrics:
                aggr_metrics_helper.add(
                    data=_result.metrics,
                    weight=_result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
                    contributor_name=_result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
                    contribution_round=_result.current_round,
                )

        aggr_params = aggr_helper.get_result()
        aggr_metrics = aggr_metrics_helper.get_result() if all_metrics else None

        aggr_result = FLModel(
            params=aggr_params,
            params_type=results[0].params_type,
            metrics=aggr_metrics,
            meta={"nr_aggregated": len(results), "current_round": results[0].current_round},
        )
        return aggr_result

    def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel:
        """Called by the `run` routine to aggregate the training results of clients.

        Args:
            results: a list of FLModel containing training results of the clients.
            aggregate_fn: a function that turns the list of FLModel into one resulting (aggregated) FLModel.

        Returns: aggregated FLModel.

        """
        self.debug("Start aggregation.")
        self.event(AppEventType.BEFORE_AGGREGATION)
        self._check_results(results)

        if not aggregate_fn:
            aggregate_fn = self.aggregate_fn

        self.info(f"aggregating {len(results)} update(s) at round {self.current_round}")
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

    def update_model(self, model, aggr_result):
        """Called by the `run` routine to update the current global model (self.model) given the aggregated result.

        Args:
            model: FLModel to be updated.
            aggr_result: aggregated FLModel.

        Returns: None.

        """
        self.event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE)

        model = FLModelUtils.update_model(model, aggr_result)

        # persistor uses Learnable format to save model
        ml = make_model_learnable(weights=model.params, meta_props=model.meta)
        self.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, ml, private=True, sticky=True)

        self.event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE)

        return model
