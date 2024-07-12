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

import torch

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.utils.math_utils import parse_compare_criteria
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs


class PTFedAvgEarlyStopping(BaseFedAvg):
    """Controller for FedAvg Workflow with Early Stopping and Model Selection.

    Args:
        num_clients (int, optional): The number of clients. Defaults to 3.
        num_rounds (int, optional): The total number of training rounds. Defaults to 5.
        stop_cond (str, optional): early stopping condition based on metric.
            string literal in the format of "<key> <op> <value>" (e.g. "accuracy >= 80")
        save_filename (str, optional): filename for saving model
        initial_model (nn.Module, optional): initial PyTorch model
    """

    def __init__(
        self,
        *args,
        stop_cond: str = None,
        save_filename: str = "FL_global_model.pt",
        initial_model=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.stop_cond = stop_cond
        if stop_cond:
            self.stop_condition = parse_compare_criteria(stop_cond)
        else:
            self.stop_condition = None
        self.save_filename = save_filename
        self.initial_model = initial_model
        self.best_model: Optional[FLModel] = None

    def run(self) -> None:
        self.info("Start FedAvg.")

        if self.initial_model:
            # Use FOBS for serializing/deserializing PyTorch tensors (self.initial_model)
            fobs.register(TensorDecomposer)
            # PyTorch weights
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

            results = self.send_model_and_wait(targets=clients, data=model)

            aggregate_results = self.aggregate(
                results, aggregate_fn=self.aggregate_fn
            )  # using default aggregate_fn with `WeightedAggregationHelper`. Can overwrite self.aggregate_fn with signature Callable[List[FLModel], FLModel]

            model = self.update_model(model, aggregate_results)

            self.info(f"Round {self.current_round} global metrics: {model.metrics}")

            self.select_best_model(model)

            self.save_model(self.best_model, os.path.join(os.getcwd(), self.save_filename))

            if self.should_stop(model.metrics, self.stop_condition):
                self.info(
                    f"Stopping at round={self.current_round} out of total_rounds={self.num_rounds}. Early stop condition satisfied: {self.stop_condition}"
                )
                break

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

    def save_model(self, model, filepath=""):
        params = model.params
        # PyTorch save
        torch.save(params, filepath)

        # save FLModel metadata
        model.params = {}
        fobs.dumpf(model, filepath + ".metadata")
        model.params = params

    def load_model(self, filepath=""):
        # PyTorch load
        params = torch.load(filepath)

        # load FLModel metadata
        model = fobs.loadf(filepath + ".metadata")
        model.params = params
        return model
