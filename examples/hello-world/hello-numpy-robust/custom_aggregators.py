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

"""Custom aggregators for the hello-numpy-robust example."""

import numpy as np

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class MedianAggregator(ModelAggregator):
    """Element-wise median aggregator to reduce outlier impact."""

    def __init__(self):
        super().__init__()
        self._client_models = []
        self._params_type = None

    def accept_model(self, model: FLModel):
        if self._params_type is None:
            self._params_type = model.params_type
        elif self._params_type != model.params_type:
            raise ValueError(
                f"ParamsType mismatch: expected {self._params_type}, got {model.params_type}. "
                "All client models must have the same params_type."
            )

        self._client_models.append(model.params)

    def aggregate_model(self) -> FLModel:
        if not self._client_models:
            self.error("No client models to aggregate.")
            return FLModel(params={})

        aggregated_params = {}
        for key in self._client_models[0].keys():
            stacked = np.stack([m[key] for m in self._client_models], axis=0)
            aggregated_params[key] = np.median(stacked, axis=0)

        return FLModel(params=aggregated_params, params_type=self._params_type)

    def reset_stats(self):
        self._client_models = []
        self._params_type = None
