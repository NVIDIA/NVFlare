# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Callable, Dict, Optional

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.math_utils import parse_compare_criteria

from .base_fedavg import BaseFedAvg


class FedAvgEarlyStopping(BaseFedAvg):
    """Controller for FedAvg Workflow with Early Stopping and Model Selection.

    Args:
        stop_cond (str, optional): early stopping condition based on metric.
        string literal in the format of "<key> <op> <value>" (e.g. "accuracy >= 80")
    """

    def __init__(
        self,
        *args,
        stop_cond: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if stop_cond:
            self.stop_condition = parse_compare_criteria(stop_cond)
        else:
            self.stop_condition = None
        self.best_model: Optional[FLModel] = None

    def run(self) -> None:
        self.info("Start FedAvg.")

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            clients = self.sample_clients(self.min_clients)

            results = self.send_model_and_wait(targets=clients, data=model)

            aggregate_results = self.aggregate(
                results, aggregate_fn=self.aggregate_fn
            )  # using default aggregate_fn with `WeightedAggregationHelper`. Can overwrite self.aggregate_fn with signature Callable[List[FLModel], FLModel]

            model = self.update_model(model, aggregate_results)

            self.info(f"Round {self.current_round} global metrics: {model.metrics}")

            self.select_best_model(model)

            if self.should_stop(model.metrics, self.stop_condition):
                self.info(
                    f"Stopping at round={self.current_round} out of total_rounds={self.num_rounds}. Early stop condition satisfied: {self.stop_condition}"
                )
                break

        save_path = os.path.join(self.get_app_dir(), "FL_global_model.pt")
        self.save_flmodel(self.best_model, save_path)

        self.info("Finished FedAvg.")

    def should_stop(self, metrics: Optional[Dict] = None, stop_condition: Optional[str] = None):
        if stop_condition is None or metrics is None:
            return False

        key, target, op_fn = stop_condition
        value = metrics.get(key, None)

        if value is None:
            raise RuntimeError(f"stop criteria key '{key}' doesn't exists in metrics")

        return op_fn(value, target)

    def select_best_model(self, curr_model: FLModel):
        if self.best_model is None:
            self.best_model = curr_model
            return

        if self.stop_condition:
            metric, _, op_fn = self.stop_condition
            if self.is_curr_model_better(self.best_model, curr_model, metric, op_fn):
                self.info("Current model is new best model.")
                self.best_model = curr_model
        else:
            self.best_model = curr_model

    def is_curr_model_better(
        self, best_model: FLModel, curr_model: FLModel, target_metric: str, op_fn: Callable
    ) -> bool:
        curr_metrics = curr_model.metrics
        if curr_metrics is None:
            return False
        if target_metric not in curr_metrics:
            return False

        best_metrics = best_model.metrics
        return op_fn(curr_metrics.get(target_metric), best_metrics.get(target_metric))
