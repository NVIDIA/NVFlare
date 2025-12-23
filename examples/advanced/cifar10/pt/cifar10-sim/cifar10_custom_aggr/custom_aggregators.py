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
"""
Custom aggregator implementations for FedAvg recipe.
This module provides two example aggregators:
1. WeightedAggregator: Aggregates based on client data size (num_steps)
2. MedianAggregator: Uses median aggregation for Byzantine robustness
"""

import numpy as np

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class WeightedAggregator(ModelAggregator):
    """
    Weighted aggregation based on client data size.

    This aggregator weights each client's contribution by the number of training steps
    (or samples) they performed, which is more fair when clients have different dataset sizes.
    """

    def __init__(self):
        super().__init__()
        self.weighted_sum = {}
        self.total_weight = 0
        self.client_weights = []  # Track individual client weights for debugging
        self.params_type = None  # Track params_type from accepted models

    def accept_model(self, model: FLModel):
        """Accept submitted model and add to the weighted sum."""
        # Get client's data size from metadata (NUM_STEPS_CURRENT_ROUND is sent by client)
        weight = model.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0)
        self.client_weights.append(weight)

        # Track and validate params_type from all models
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(
                f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}. "
                "All client models must have the same params_type."
            )

        for key, value in model.params.items():
            if key not in self.weighted_sum:
                self.weighted_sum[key] = value * weight
            else:
                self.weighted_sum[key] += value * weight
        self.total_weight += weight

    def aggregate_model(self) -> FLModel:
        """Perform weighted aggregation and return result as FLModel."""
        if self.total_weight == 0:
            self.error("Total weight is zero, cannot aggregate!")
            return FLModel(params={})

        aggregated_params = {key: val / self.total_weight for key, val in self.weighted_sum.items()}

        # Return with the same params_type as the accepted models
        return FLModel(params=aggregated_params, params_type=self.params_type)

    def reset_stats(self):
        """Reset the aggregator state for next round."""
        self.weighted_sum = {}
        self.total_weight = 0
        self.client_weights = []
        self.params_type = None


class MedianAggregator(ModelAggregator):
    """
    Median aggregation for Byzantine robustness.

    Instead of averaging, this aggregator computes the median of each parameter
    across all clients. This provides robustness against Byzantine (malicious) clients
    who might send adversarial model updates.
    """

    def __init__(self):
        super().__init__()
        self.client_models = []
        self.params_type = None  # Track params_type from accepted models

    def accept_model(self, model: FLModel):
        """Accept submitted model and add to collection."""
        # Track and validate params_type from all models
        if self.params_type is None:
            self.params_type = model.params_type
        elif self.params_type != model.params_type:
            raise ValueError(
                f"ParamsType mismatch: expected {self.params_type}, got {model.params_type}. "
                "All client models must have the same params_type."
            )

        self.client_models.append(model.params)

    def aggregate_model(self) -> FLModel:
        """Perform median aggregation and return result as FLModel."""
        if len(self.client_models) == 0:
            self.error("No client models to aggregate!")
            return FLModel(params={})

        # Stack all client parameters and compute median using numpy
        aggregated_params = {}
        param_keys = self.client_models[0].keys()

        for key in param_keys:
            # Stack arrays from all clients along axis 0 (note, for large models, this can be memory intensive)
            stacked = np.stack([m[key] for m in self.client_models], axis=0)
            # Compute median along the client dimension (axis=0)
            aggregated_params[key] = np.median(stacked, axis=0)

        # Return with the same params_type as the accepted models
        return FLModel(params=aggregated_params, params_type=self.params_type)

    def reset_stats(self):
        """Reset the aggregator state for next round."""
        self.client_models = []
        self.params_type = None
