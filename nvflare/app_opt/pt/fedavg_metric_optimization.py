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

import operator
import os
from typing import Any, List, Literal, Optional

import torch
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs

"""
The FedAvgMetricOptimization class combines two techniques—early stopping and metric
optimization (minimization or maximization)—to identify the best model during FL
learning. It uses a patience parameter, which specifies how many FL rounds to wait
without any metric improvement before stopping the training.
"""


class PTFedAvgMetricOptimization(BaseFedAvg):
    def __init__(
        self,
        *args,
        target_metric_name: str,
        optimization_mode: Literal["min", "max"] = "min",
        task_train_name: Optional[str] = AppConstants.TASK_TRAIN,
        task_validation_name: Optional[str] = AppConstants.TASK_VALIDATION,
        task_to_optimize: Literal["train", "validation"] = "validation",
        patience: Optional[int] = None,
        save_filename: Optional[str] = "FL_global_model.pt",
        initial_model=None,
        **kwargs,
    ) -> None:
        """Controller for FedAvg Workflow with Metric Optimization and Model Selection

        Args:
            target_metric_name (str): The name of the metric to track.
            optimization_mode (str, optional): one of `min` or `max`. In `min` mode,
                training will stop when quantity monitored has stopped decreasing and
                in `max` mode it will stop when the quantity monitored has stopped
                increasing.
            task_train_name (str, optional): Name of the train task. Defaults to
                AppConstants.TASK_TRAIN.
            task_validation_name (str, optional): Name of the validation task. Defaults
                to AppConstants.TASK_VALIDATION.
            task_to_optimize (str, optional): Specifies Which task to optimize the
                target metric for.
            patience (int, optional): The number of checks with no improvement after
                which the FL will be stopped. If set to `None`, the training would
                not stop.
            save_filename (str, optional): The filename for saving model.
            initial_model (nn.Module, optional): The initial PyTorch model.
        """
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.optimization_mode = optimization_mode
        self.num_rounds_without_improvement = 0
        self.save_filename = save_filename
        self.initial_model = initial_model

        self.task_train_name = task_train_name
        self.task_validation_name = task_validation_name
        self.task_to_optimize = task_to_optimize

        if not self.optimization_mode.lower() in ["min", "max"]:
            raise RuntimeError(
                f"The optimization mode '{self.optimization_mode}' is invalid, should "
                " be either: 'min' or 'max'"
            )
        if not self.task_to_optimize.lower() in ["train", "validation"]:
            raise RuntimeError(
                f"The task to optimize '{self.optimization_mode}' is invalid, should "
                " be either: 'train' or 'validation'"
            )

        if self.optimization_mode.lower() == "min":
            self.optimization_func = operator.lt
        else:
            self.optimization_func = operator.gt

        self.target_metric_name = target_metric_name
        self.best_model: Optional[FLModel] = None
        self.target_metric_value: Optional[Any] = None

    def aggregate_target_metrics(self, models: List[FLModel]) -> Any:
        """Aggregates the target metric from the client's model.

        Args:
            models (List[FLModel]): List of client's model

        Returns:
            The aggregated target metric
        """
        if not models:
            raise RuntimeError("The list of models is empty")

        target_metrics = None

        for model in models:
            target_metric_value = model.metrics.get(self.target_metric_name, None)
            if target_metric_value is None:
                raise RuntimeError(
                    f"Stop criteria key '{self.target_metric_name}' doesn't exists in "
                    "metrics."
                )

            if target_metrics is None:
                target_metrics = target_metric_value
            else:
                target_metrics += target_metric_value

        return target_metrics / len(models)

    def run(self) -> None:
        """Main FL running loop"""
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

        for self.current_round in range(
            self.start_round, self.start_round + self.num_rounds
        ):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            train_results = self.send_model_and_wait(
                task_name=self.task_train_name, targets=clients, data=model
            )
            aggregate_training_data = self.aggregate(
                train_results, aggregate_fn=self.aggregate_fn
            )
            model = self.update_model(model, aggregate_training_data)

            self.info(f"Round {self.current_round} global metrics: {model.metrics}")

            evaluate_results = self.send_model_and_wait(
                task_name=self.task_validation_name, targets=clients, data=model
            )

            if self.task_to_optimize.lower() == "validation":
                aggregated_target_metric = self.aggregate_target_metrics(
                    evaluate_results
                )
            else:
                aggregated_target_metric = self.aggregate_target_metrics(train_results)

            self.update_target_metric_value(aggregated_target_metric)

            if self.should_stop():
                self.info(
                    f"Stopping at round={self.current_round} out of "
                    f"total_rounds={self.num_rounds}. Metric Optimization stop "
                    f"condition satisfied. Target metric {self.target_metric_name} "
                    f"did not improve for ({self.patience}) consecutive FL rounds."
                )
                break

            if self.should_save_model(model):
                self.info("Current model is new best model.")
                self.save_model(model, os.path.join(os.getcwd(), self.save_filename))

        self.info("Finished FedAvg with metric optimization.")

    def is_aggregated_target_metric_better(self, aggregated_target_metric: Any) -> bool:
        """Checks if the aggregated target metric are better thatn the previous best.

        Args:
            aggregated_target_metric (Any): The new aggregated target metrics to compare
            with

        Returns:
            True if the aggregated target metric is better than previous best,
            otherwise, False.
        """
        if self.target_metric_value is None:
            return True

        return self.optimization_func(
            aggregated_target_metric, self.target_metric_value
        )

    def update_target_metric_value(self, aggregated_target_metric: Any) -> None:
        """Updates the target metric value if the new aggregated target metric is
        better.

        Args:
            aggregated_target_metric (Any): The new aggregated metrics to compare with
        """
        if self.is_aggregated_target_metric_better(aggregated_target_metric):
            self.info(
                f"Target metric ({self.target_metric_name}) improved with value: "
                f"from {self.target_metric_value} to {aggregated_target_metric}."
            )
            self.target_metric_value = aggregated_target_metric
            self.num_rounds_without_improvement = 0
        else:
            self.num_rounds_without_improvement += 1

    def should_stop(self) -> bool:
        """Checks if the stop criteria is met.

        Returns:
            True if the stop criter is met, otherwise, False. If patience is None,
            always returns False.
        """
        if self.patience is None:
            return False

        if self.num_rounds_without_improvement >= self.patience:
            return True
        else:
            return False

    def should_save_model(self, model: FLModel) -> bool:
        """Checks if the current model is better than previous best model.

        Args:
            model (FLModel): model to save

        Returns:
            True if the new model is better than previous model, otherwise, False.
        """
        if self.num_rounds_without_improvement == 0:
            self.best_model = model
            return True
        else:
            return False

    def save_model(self, model: FLModel, filepath: Optional[str] = "") -> None:
        """Saves an object to a file.

        Args:
            model (FLModel): model to save
            filepath (str, optional): location of the saved model
        """
        params = model.params
        # PyTorch save
        torch.save(params, filepath)

        # save FLModel metadata
        model.params = {}
        fobs.dumpf(model, filepath + ".metadata")
        model.params = params

    def load_model(self, filepath: Optional[str] = "") -> None:
        """Loads an object saved from a file.

        Args:
            filepath (str, optional): location of the saved model
        """
        # PyTorch load
        params = torch.load(filepath)

        # load FLModel metadata
        model = fobs.loadf(filepath + ".metadata")
        model.params = params
        return model
