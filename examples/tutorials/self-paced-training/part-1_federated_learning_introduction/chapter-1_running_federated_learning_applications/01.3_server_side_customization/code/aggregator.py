# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""A custom server aggregator for this lesson's full NumPy model weights."""

import numpy as np

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class MedianAggregator(ModelAggregator):
    """Take the median of each parameter coordinate, giving every site one vote.

    This small-model example keeps all client results for the current round in
    memory. Clients must send matching parameter arrays and an accuracy metric.
    """

    def __init__(self):
        super().__init__()
        self.reset_stats()

    def accept_model(self, model: FLModel):
        """Collect one client's trained model for the current round."""
        if model.params_type != ParamsType.FULL:
            raise ValueError("This example expects full model weights.")
        if not model.metrics or "accuracy" not in model.metrics:
            raise ValueError("This example expects an 'accuracy' metric from every client.")
        self.models.append(model)

    def aggregate_model(self) -> FLModel:
        """Combine weights by median and report the mean received-model accuracy."""
        if not self.models:
            raise ValueError("No client models to aggregate.")

        params = {}
        for key in self.models[0].params:
            # Axis 0 selects clients; all remaining dimensions belong to the parameter.
            stacked = np.stack([model.params[key] for model in self.models], axis=0)
            params[key] = np.median(stacked, axis=0)

        # Metrics need their own aggregation rule: each site has 100 evaluation samples.
        accuracy = float(np.mean([model.metrics["accuracy"] for model in self.models]))
        result = FLModel(params=params, params_type=ParamsType.FULL, metrics={"accuracy": accuracy})
        self.reset_stats()
        return result

    def reset_stats(self):
        """Discard the previous round's contributions before collecting new ones."""
        self.models = []
