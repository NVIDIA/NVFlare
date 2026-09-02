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

"""Helpers for valid-supervision client participation."""

from typing import Any

import torch

from nvflare.apis.fl_constant import FLMetaKey


def state_dict_for_update(model: torch.nn.Module, paired_examples: int) -> dict[str, Any]:
    """Return empty parameters when a client has no paired target supervision."""

    if paired_examples <= 0:
        return {}
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def aggregation_meta(paired_examples: int) -> dict[str, float]:
    """Use valid paired examples as the FedAvg aggregation weight."""

    return {FLMetaKey.NUM_STEPS_CURRENT_ROUND: float(max(0, paired_examples))}
