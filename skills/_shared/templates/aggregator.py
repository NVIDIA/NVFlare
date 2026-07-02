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

"""Packaged custom-aggregation template: a step-weighted ``ModelAggregator``.

Copy and adapt this into a generated ``aggregators.py`` when the conversion
needs custom aggregation. Wire it through the recipe ``aggregator=`` parameter
in ``job.py`` with the matching ``aggregator_data_kind`` and parameter transfer
settings. This uses the product extension point rather than a skill-owned
algorithm table, and it fits the standard ``FLModel`` exchange contract, so it
needs no client-side change beyond sending step-count metadata.
"""

from nvflare.apis.dxo import MetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class WeightedAggregator(ModelAggregator):
    """Average client updates weighted by each client's local step count."""

    def __init__(self):
        super().__init__()
        self.reset_stats()

    def reset_stats(self):
        self._weighted_sum = None
        self._total_weight = 0.0
        self._params_type = None

    def accept_model(self, model: FLModel):
        # Coerce missing/non-positive step counts to weight 1.0, matching the
        # product's own base_fedavg._get_num_steps_weight behavior.
        weight = float(model.meta.get(MetaKey.NUM_STEPS_CURRENT_ROUND, 1) or 1)
        self._params_type = model.params_type
        if self._weighted_sum is None:
            self._weighted_sum = {key: value * weight for key, value in model.params.items()}
        else:
            for key, value in model.params.items():
                self._weighted_sum[key] = self._weighted_sum[key] + value * weight
        self._total_weight += weight

    def aggregate_model(self) -> FLModel:
        if not self._total_weight:
            raise RuntimeError("no client models accepted this round")
        averaged = {key: value / self._total_weight for key, value in self._weighted_sum.items()}
        result = FLModel(params=averaged, params_type=self._params_type)
        self.reset_stats()
        return result
