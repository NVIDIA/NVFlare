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

import math

from nvflare.apis.dxo import MetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


def _step_weight(model: FLModel) -> float:
    # Mirror the product's base_fedavg._get_num_steps_weight: use the client's
    # step count only when it is a finite positive number; otherwise fall back
    # to 1.0. Reject bool (a bool is an int in Python) and non-numeric/None, and
    # never let negative or non-finite metadata corrupt the weighted average.
    value = (model.meta or {}).get(MetaKey.NUM_STEPS_CURRENT_ROUND)
    if isinstance(value, bool) or value is None:
        return 1.0
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    return weight if math.isfinite(weight) and weight > 0 else 1.0


class WeightedAggregator(ModelAggregator):
    """Average client updates weighted by each client's local step count."""

    def __init__(self):
        super().__init__()
        self.reset_stats()

    def reset_stats(self):
        self._weighted_sum = {}
        # Per-key weight so a parameter present in only some clients is averaged
        # over just those clients (not diluted by the full round weight), and a
        # key missing from the first client does not raise KeyError.
        self._key_weight = {}
        self._params_type = None

    def accept_model(self, model: FLModel):
        weight = _step_weight(model)
        self._params_type = model.params_type
        for key, value in model.params.items():
            if key in self._weighted_sum:
                self._weighted_sum[key] = self._weighted_sum[key] + value * weight
                self._key_weight[key] += weight
            else:
                self._weighted_sum[key] = value * weight
                self._key_weight[key] = weight

    def aggregate_model(self) -> FLModel:
        if not self._weighted_sum:
            raise RuntimeError("no client models accepted this round")
        averaged = {key: self._weighted_sum[key] / self._key_weight[key] for key in self._weighted_sum}
        result = FLModel(params=averaged, params_type=self._params_type)
        self.reset_stats()
        return result
