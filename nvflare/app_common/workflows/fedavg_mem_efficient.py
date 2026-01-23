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

import gc

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants

from .fedavg import FedAvg


class FedAvgMemEfficient(FedAvg):
    """Memory-efficient FedAvg that aggregates in-place to minimize memory usage.

    This controller uses an optimized aggregation approach that:
    - Aggregates directly into the global model (no intermediate buffer)
    - Frees client results parameter-by-parameter during aggregation
    - Uses in-place operations (PyTorch tensors or NumPy arrays)

    Memory usage: ~(num_clients + 1) * model_size at start,
                  dropping to ~1 * model_size as aggregation proceeds
    vs standard: (num_clients + 2) * model_size throughout

    Works with both PyTorch tensors and NumPy arrays.
    """

    def run(self) -> None:
        """Run FedAvg with memory-efficient in-place aggregation."""
        from nvflare.fuel.utils.log_utils import center_message

        self.info(center_message("Start FedAvg (Memory-Efficient)."))

        # Load initial model
        if self.initial_model is not None:
            if isinstance(self.initial_model, FLModel):
                model = self.initial_model
            else:
                model = FLModel(params=self.initial_model)
            self.info("Using provided initial_model")
        else:
            model = self.load_model()

        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(center_message(message=f"Round {self.current_round} started.", boarder_str="-"))
            model.current_round = self.current_round
            clients = self.sample_clients(self.num_clients)

            # Collect all results first
            results = self.send_model_and_wait(targets=clients, data=model)

            # Aggregate in-place directly into model
            self.info(f"Aggregating {len(results)} results in-place (memory-efficient)")
            model = self._aggregate_mem_efficient(model, results)

            # Free results immediately
            results.clear()
            del results
            gc.collect()

            # Early stopping logic (inherited from FedAvg)
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
                if self.should_stop(model.metrics):
                    self.info(f"Stopping at round={self.current_round} out of total_rounds={self.num_rounds}.")
                    break
            else:
                self.save_model(model)

        self.info(center_message("Finished FedAvg (Memory-Efficient)."))

    def _aggregate_mem_efficient(self, target_model: FLModel, results: list) -> FLModel:
        """Memory-efficient aggregation that modifies target_model in-place.

        Handles both PyTorch tensors and NumPy arrays.

        Args:
            target_model: Global model to aggregate into (modified in-place)
            results: List of client results

        Returns:
            target_model (same reference, modified in-place)
        """
        if not results or not results[0].params:
            return target_model

        # Calculate total weights per parameter
        total_weights = {}
        for result in results:
            weight = result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
            for k in result.params.keys():
                total_weights[k] = total_weights.get(k, 0.0) + weight

        # Determine if this is DIFF aggregation
        is_diff = results[0].params_type == ParamsType.DIFF

        # Get all parameter keys
        param_keys = list(results[0].params.keys())

        # Process parameter-by-parameter to minimize memory
        for k in param_keys:
            if k not in target_model.params:
                continue

            target_param = target_model.params[k]

            # Check if it's a PyTorch tensor by checking for tensor-specific methods
            is_torch_tensor = hasattr(target_param, "add_") and hasattr(target_param, "zero_")

            # For FULL params: zero out target first
            # For DIFF params: keep existing values (differential update)
            if not is_diff:
                if is_torch_tensor:
                    target_param.zero_()
                else:
                    # NumPy array
                    target_model.params[k][:] = 0

            # Accumulate weighted contributions from all clients
            for result in results:
                if k not in result.params:
                    continue

                weight = result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
                normalized_weight = weight / total_weights[k]

                # In-place accumulation
                if is_torch_tensor:
                    # PyTorch tensor - use .add_() method
                    if hasattr(target_param.dtype, "is_floating_point") and target_param.dtype.is_floating_point:
                        target_param.add_(result.params[k], alpha=normalized_weight)
                    else:
                        # Integer/bool tensors: add without scaling
                        target_param.add_(result.params[k])
                else:
                    # NumPy array - use += operator
                    target_model.params[k] += result.params[k] * normalized_weight

                # Free client parameter immediately
                del result.params[k]

            # Force GC after each parameter
            gc.collect()

        # Clear all result param dicts
        for result in results:
            result.params.clear()

        gc.collect()

        # Aggregate metrics
        aggr_metrics = None
        all_metrics = all(r.metrics for r in results)
        if all_metrics:
            aggr_metrics_helper = WeightedAggregationHelper()
            for result in results:
                aggr_metrics_helper.add(
                    data=result.metrics,
                    weight=result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
                    contributor_name=result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
                    contribution_round=result.current_round,
                )
            aggr_metrics = aggr_metrics_helper.get_result()

        # Update target_model metadata
        target_model.metrics = aggr_metrics
        target_model.meta = {"nr_aggregated": len(results), "current_round": results[0].current_round}
        target_model.params_type = results[0].params_type

        return target_model
