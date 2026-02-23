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

import os
import time
from typing import Any, Dict, Optional, Union

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.math_utils import parse_compare_criteria
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.log_utils import center_message

from .base_fedavg import BaseFedAvg


class FedAvg(BaseFedAvg):
    """Controller for FedAvg Workflow with optional Early Stopping and Model Selection.

    *Note*: This class is based on the `ModelController`.
    Implements [FederatedAveraging](https://arxiv.org/abs/1602.05629).

    Uses InTime (streaming) aggregation for memory efficiency - each client result is
    aggregated immediately upon receipt rather than collecting all results first.

    Supports custom aggregators via the ModelAggregator interface.

    Provides the implementations for the `run` routine, controlling the main workflow:
        - def run(self)

    The parent classes provide the default implementations for other routines.

    For simple model persistence without complex ModelPersistor setup, you can:
    1. Pass `model` (dict of params) and `save_filename`
    2. Override `save_model()` and `load_model()` for framework-specific serialization

    Args:
        num_clients (int, optional): The number of clients. Defaults to 3.
        num_rounds (int, optional): The total number of training rounds. Defaults to 5.
        start_round (int, optional): The starting round number.
        persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".
            If empty and model is provided, uses simple save_model/load_model methods.
        model (dict or FLModel, optional): Initial model parameters. If provided,
            this is used instead of loading from persistor. Defaults to None.
        save_filename (str, optional): Filename for saving the best model. Defaults to
            "FL_global_model.pt". Only used when persistor_id is empty.
        aggregator (ModelAggregator, optional): Custom aggregator for combining client
            model updates. Must implement accept_model(), aggregate_model(), reset_stats().
            If None, uses built-in weighted averaging (memory-efficient). Defaults to None.
        stop_cond (str, optional): Early stopping condition based on metric. String
            literal in the format of '<key> <op> <value>' (e.g. "accuracy >= 80").
            If None, early stopping is disabled. Defaults to None.
        patience (int, optional): The number of rounds with no improvement after which
            FL will be stopped. Only applies if stop_cond is set. Defaults to None.
        task_name (str, optional): Task name for training. Defaults to "train".
        exclude_vars (str, optional): Regex pattern for variables to exclude from
            aggregation. Defaults to None. Only used when no custom aggregator is provided.
        aggregation_weights (dict, optional): Per-client aggregation weights.
            Defaults to None (equal weights). Only used when no custom aggregator is provided.
    """

    def __init__(
        self,
        *args,
        model: Optional[Union[Dict, FLModel]] = None,
        save_filename: Optional[str] = "FL_global_model.pt",
        aggregator: Optional[ModelAggregator] = None,
        stop_cond: Optional[str] = None,
        patience: Optional[int] = None,
        task_name: Optional[str] = "train",
        exclude_vars: Optional[str] = None,
        aggregation_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Simple model persistence (alternative to persistor)
        self.model = model
        self.save_filename = save_filename

        # Custom aggregator (optional)
        self.aggregator = aggregator

        # Early stopping configuration
        self.stop_cond = stop_cond
        self.patience = patience
        self.task_name = task_name

        # Aggregation configuration (used only when no custom aggregator)
        self.exclude_vars = exclude_vars
        self.aggregation_weights = aggregation_weights or {}

        # Parse stop condition
        if self.stop_cond:
            self.stop_condition = parse_compare_criteria(stop_cond)
        else:
            self.stop_condition = None

        # Early stopping state
        self.num_fl_rounds_without_improvement: int = 0
        self.best_target_metric_value: Any = None

        # InTime aggregation helpers (reset each round, used only when no custom aggregator)
        self._aggr_helper: Optional[WeightedAggregationHelper] = None
        self._aggr_metrics_helper: Optional[WeightedAggregationHelper] = None
        self._all_metrics: bool = True
        self._received_count: int = 0
        self._expected_count: int = 0
        self._params_type = None  # Only store params_type, not full result

    def run(self) -> None:
        self.info(center_message("Start FedAvg."))

        # Set NUM_ROUNDS in FL context for persistor and other components
        self.fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self.num_rounds, private=True, sticky=False)

        # Load initial model - prefer model if provided, else use persistor
        if self.model is not None:
            if isinstance(self.model, FLModel):
                model = self.model
            else:
                # Assume dict of params
                model = FLModel(params=self.model)
            self.info("Using provided model")
        else:
            model = self.load_model()

        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(center_message(message=f"Round {self.current_round} started.", boarder_str="-"))

            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            # Reset aggregation state for this round
            if self.aggregator:
                # Use custom aggregator
                self.aggregator.reset_stats()
            else:
                # Use built-in InTime aggregation
                self._aggr_helper = WeightedAggregationHelper(exclude_vars=self.exclude_vars)
                self._aggr_metrics_helper = WeightedAggregationHelper()
                self._all_metrics = True  # Only used by built-in aggregation
            # Shared state for both aggregator types
            self._received_count = 0
            self._expected_count = len(clients)
            self._params_type = None

            # Non-blocking send with callback for streaming aggregation
            self.send_model(
                task_name=self.task_name,
                targets=clients,
                data=model,
                callback=self._aggregate_one_result,
            )

            # Wait for all results to be processed
            while self.get_num_standing_tasks():
                if self.abort_signal.triggered:
                    self.info("Abort signal triggered. Finishing FedAvg.")
                    return
                time.sleep(self._task_check_period)

            # Fire BEFORE_AGGREGATION so widgets (e.g. IntimeModelSelector) can run and fire GLOBAL_BEST_MODEL_AVAILABLE
            self.fire_event(AppEventType.BEFORE_AGGREGATION, self.fl_ctx)

            # Get final aggregated result
            aggregate_results = self._get_aggregated_result()

            model = self.update_model(model, aggregate_results)

            # Early stopping: check if current model is better
            if self.stop_condition:
                self.info(f"Round {self.current_round} global metrics: {model.metrics}")

                if self.is_curr_model_better(model):
                    self.info("New best model found.")
                    self.save_model(model)
                else:
                    if self.patience:
                        self.info(
                            f"No metric improvement, num of FL rounds without improvement: "
                            f"{self.num_fl_rounds_without_improvement}"
                        )

                # Check if we should stop early
                if self.should_stop(model.metrics):
                    self.info(f"Stopping at round={self.current_round} out of total_rounds={self.num_rounds}.")
                    break
            else:
                # No early stopping: save model every round
                self.save_model(model)

            # Memory cleanup at end of round (if configured)
            self._maybe_cleanup_memory()

        self.info(center_message("Finished FedAvg."))

    def _aggregate_one_result(self, result: FLModel) -> None:
        """Callback: aggregate ONE client result immediately (InTime aggregation)."""
        if not result.params:
            client_name = result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN)
            self.warning(f"Empty result from client {client_name}, skipping.")
            return

        # Store only params_type from first result (not the full model)
        if self._params_type is None:
            self._params_type = result.params_type

        client_name = result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN)

        if self.aggregator:
            # Use custom aggregator
            self.aggregator.accept_model(result)
        else:
            # Use built-in InTime aggregation with weighted averaging
            # Get weight: use aggregation_weights if specified, else use NUM_STEPS
            if self.aggregation_weights and client_name in self.aggregation_weights:
                aggregation_weight = self.aggregation_weights[client_name]
            else:
                aggregation_weight = 1.0

            n_iter = result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, None)
            # Handle None case (e.g., first round of some algorithms like K-Means)
            if n_iter is None:
                n_iter = 1.0
            weight = aggregation_weight * float(n_iter)

            self._aggr_helper.add(
                data=result.params,
                weight=weight,
                contributor_name=client_name,
                contribution_round=self.current_round,
            )

            # Add to metrics aggregation if available
            if not result.metrics:
                self._all_metrics = False
            if self._all_metrics and result.metrics:
                self._aggr_metrics_helper.add(
                    data=result.metrics,
                    weight=weight,
                    contributor_name=client_name,
                    contribution_round=self.current_round,
                )

        self._received_count += 1
        self.info(f"Aggregated {self._received_count}/{self._expected_count} results")

    def _get_aggregated_result(self) -> FLModel:
        """Get the final aggregated result after all clients have responded."""
        if self.aggregator:
            # Use custom aggregator
            result: FLModel = self.aggregator.aggregate_model()
            result.meta = result.meta or {}
            result.meta["nr_aggregated"] = self._received_count
            result.meta["current_round"] = self.current_round
            return result
        else:
            # Use built-in InTime aggregation
            aggr_params = self._aggr_helper.get_result()
            aggr_metrics = self._aggr_metrics_helper.get_result() if self._all_metrics else None

            return FLModel(
                params=aggr_params,
                params_type=self._params_type,
                metrics=aggr_metrics,
                meta={"nr_aggregated": self._received_count, "current_round": self.current_round},
            )

    def should_stop(self, metrics: Optional[Dict] = None) -> bool:
        """Checks whether the current FL experiment should stop.

        Args:
            metrics (Dict, optional): experiment metrics.

        Returns:
            True if the experiment should stop, False otherwise.
        """
        if self.stop_condition is None or metrics is None:
            return False

        # Check patience
        if self.patience and (self.patience <= self.num_fl_rounds_without_improvement):
            self.info(f"Exceeded the number of FL rounds ({self.patience}) without improvements")
            return True

        # Check stop condition
        key, target, op_fn = self.stop_condition
        value = metrics.get(key, None)

        if value is None:
            self.warning(f"Stop criteria key '{key}' doesn't exist in metrics: {list(metrics.keys())}")
            return False

        if op_fn(value, target):
            self.info(f"Early stop condition satisfied: {self.stop_cond}")
            return True

        return False

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

    def load_model(self) -> FLModel:
        """Load model. Uses persistor if available, otherwise uses load_model_file.

        Override `load_model_file` for framework-specific deserialization (e.g., torch.load).

        Returns:
            FLModel: loaded model, or None if loading fails
        """
        if self.persistor:
            # Use persistor (parent class behavior)
            return super().load_model()
        elif self.save_filename:
            # Try to load from file
            filepath = os.path.join(self.get_run_dir(), self.save_filename)
            if os.path.exists(filepath):
                self.info(f"Loading model from {filepath}")
                return self.load_model_file(filepath)
            else:
                self.info(f"No saved model found at {filepath}, starting fresh")
                return FLModel(params={})
        else:
            self.warning("No persistor or save_filename configured")
            return FLModel(params={})

    def save_model(self, model: FLModel) -> None:
        """Save model. Uses persistor if available, otherwise uses save_model_file.

        Override `save_model_file` for framework-specific serialization (e.g., torch.save).

        Args:
            model (FLModel): model to save
        """
        if self.persistor:
            # Use persistor (parent class behavior)
            super().save_model(model)
        elif self.save_filename:
            # Use simple file-based saving
            filepath = os.path.join(self.get_run_dir(), self.save_filename)
            self.save_model_file(model, filepath)
            self.info(f"Model saved to {filepath}")
        else:
            self.warning("No persistor or save_filename configured, model not saved")

    def save_model_file(self, model: FLModel, filepath: str) -> None:
        """Save model to file. Override this for framework-specific serialization.

        Default implementation uses FOBS (pickle-compatible).
        For PyTorch, override with: torch.save(model.params, filepath)

        Args:
            model (FLModel): model to save
            filepath (str): path to save the model
        """
        # Default: use FOBS to save entire FLModel
        fobs.dumpf(model, filepath)

    def load_model_file(self, filepath: str) -> FLModel:
        """Load model from file. Override this for framework-specific deserialization.

        Default implementation uses FOBS (pickle-compatible).
        For PyTorch, override with: FLModel(params=torch.load(filepath))

        Args:
            filepath (str): path to load the model from

        Returns:
            FLModel: loaded model
        """
        # Default: use FOBS to load entire FLModel
        return fobs.loadf(filepath)
