# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Any, Dict, Optional

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.math_utils import parse_compare_criteria
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs


class FedAvgV1(BaseFedAvg):
    def __init__(
        self,
        *args,
        stop_cond: Optional[str] = None,
        patience: Optional[int] = None,
        task_to_optimize: Optional[str] = "train",
        initial_model: Optional[FLModel] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.patience = patience
        self.task_to_optimize = task_to_optimize
        self.num_fl_rounds_without_improvement: int = 0
        self.stop_cond = stop_cond
        if stop_cond:
            self.stop_condition = parse_compare_criteria(stop_cond)
        else:
            self.stop_condition = None

        self.initial_model: FLModel = initial_model
        self.best_target_metric_value: Any = None
        fobs.register(TensorDecomposer)

    def run(self) -> None:
        if self.initial_model:
            initial_weights = self.initial_model.state_dict()
        else:
            initial_weights = {}

        model = FLModel(params=initial_weights)

        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):

            self.info(f"Round {self.current_round} started.")

            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            results = self.send_model_and_wait(task_name=self.task_to_optimize, targets=clients, data=model)

            # using default aggregate_fn with `WeightedAggregationHelper`.
            # Can overwrite self.aggregate_fn with signature Callable[List[FLModel], FLModel]
            aggregate_results = self.aggregate(results, aggregate_fn=self.aggregate_fn)
            model = self.update_model(model, aggregate_results)

            self.info(f"Round {self.current_round} global metrics: {model.metrics}")
            self.is_curr_model_better(model)
            if self.should_stop(model.metrics):
                self.info(f"Stopping at round={self.current_round} out of total_rounds={self.num_rounds}.")
                break

        self.info("Finished FedAvg.")

    def is_curr_model_better(self, curr_model: FLModel) -> bool:
        """Checks if the new model is better than the current best model.

        Args:
            curr_model (FLModel): the new model to evaluate.

        Returns:
            True if the new model is better than the current best model, False otherwise
        """
        if self.stop_condition is None:
            return True

        curr_metrics = curr_model.metrics
        if curr_metrics is None:
            return False

        target_metric, _, op_fn = self.stop_condition
        curr_target_metric = curr_metrics.get(target_metric, None)
        if curr_target_metric is None:
            return False

        if self.best_target_metric_value is None or op_fn(curr_target_metric, self.best_target_metric_value):
            if self.patience and self.best_target_metric_value == curr_target_metric:
                self.num_fl_rounds_without_improvement += 1
                return False
            else:
                self.best_target_metric_value = curr_target_metric
                self.num_fl_rounds_without_improvement = 0
                return True

        self.num_fl_rounds_without_improvement += 1
        return False

    def should_stop(self, metrics: Optional[Dict] = None) -> bool:
        """Checks whether the current FL experiment should stop.

        Args:
            metrics (Dict, optional): experiment metrics.

        Returns:
            True if the experiment should stop, False otherwise.
        """
        if self.stop_condition is None or metrics is None:
            return False

        if self.patience and (self.patience <= self.num_fl_rounds_without_improvement):
            self.info(f"Exceeded the number of FL rounds ({self.patience}) without improvements")
            return True

        key, target, op_fn = self.stop_condition
        value = metrics.get(key, None)

        if value is None:
            raise RuntimeError(f"stop criteria key '{key}' doesn't exists in metrics")

        if op_fn(value, target):
            self.info(f"Early stop condition satisfied: {self.stop_condition}")
            return True

        return False
