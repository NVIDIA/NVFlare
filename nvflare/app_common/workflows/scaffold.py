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

import copy
from typing import List

import numpy as np

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AlgorithmConstants, AppConstants

from .base_fedavg import BaseFedAvg


class Scaffold(BaseFedAvg):
    """Controller for Scaffold Workflow. *Note*: This class is based on the experimental `ModelController`.
    Implements [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html).

    Provides the implementations for the `run` routine, controlling the main workflow:
        - def run(self)

    The parent classes provide the default implementations for other routines.

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

    def initialize(self):
        super().initialize()
        self._global_ctrl_weights = copy.deepcopy(self.model.params)
        # Initialize correction term with zeros
        for k in self._global_ctrl_weights.keys():
            self._global_ctrl_weights[k] = np.zeros_like(self._global_ctrl_weights[k])

    def run(self) -> None:
        self.info("Start FedAvg.")

        for self._current_round in range(self._num_rounds):
            self.info(f"Round {self._current_round} started.")

            clients = self.sample_clients(self._min_clients)

            # Add SCAFFOLD global control terms to global model meta
            global_model = self.model
            global_model.meta[AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL] = self._global_ctrl_weights

            results = self.send_model_and_wait(targets=clients, data=global_model)

            aggregate_results = self.aggregate(results, aggregate_fn=scaffold_aggregate_fn)

            self.update_model(aggregate_results)

            # update SCAFFOLD global controls
            ctr_diff = aggregate_results.meta[AlgorithmConstants.SCAFFOLD_CTRL_DIFF]
            for v_name, v_value in ctr_diff.items():
                self._global_ctrl_weights[v_name] += v_value

            self.save_model()

        self.info("Finished FedAvg.")


def scaffold_aggregate_fn(results: List[FLModel]) -> FLModel:
    # aggregates both the model weights and the SCAFFOLD control terms

    aggregation_helper = WeightedAggregationHelper()
    crtl_aggregation_helper = WeightedAggregationHelper()
    for _result in results:
        aggregation_helper.add(
            data=_result.params,
            weight=_result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
            contributor_name=_result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
            contribution_round=_result.meta.get("current_round", None),
        )
        crtl_aggregation_helper.add(
            data=_result.meta[AlgorithmConstants.SCAFFOLD_CTRL_DIFF],
            weight=_result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
            contributor_name=_result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
            contribution_round=_result.meta.get("current_round", None),
        )

    aggregated_dict = aggregation_helper.get_result()

    aggr_result = FLModel(
        params=aggregated_dict,
        params_type=results[0].params_type,
        meta={
            AlgorithmConstants.SCAFFOLD_CTRL_DIFF: crtl_aggregation_helper.get_result(),
            "nr_aggregated": len(results),
            "current_round": results[0].meta["current_round"],
        },
    )

    return aggr_result
