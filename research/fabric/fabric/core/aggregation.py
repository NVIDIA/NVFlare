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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Model-state copying and patient-count-weighted FedAvg."""

from __future__ import annotations

from typing import Sequence

import torch


def copy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def fedavg_state_dict(states: Sequence[dict[str, torch.Tensor]], weights: Sequence[int]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("FedAvg received zero total client weight")
    averaged: dict[str, torch.Tensor] = {}
    for key in states[0]:
        first = states[0][key]
        if not torch.is_floating_point(first):
            averaged[key] = first.clone()
            continue
        value = torch.zeros_like(first, dtype=torch.float32)
        for state, weight in zip(states, weights):  # noqa: B905
            value += state[key].float() * (float(weight) / total)
        averaged[key] = value.to(dtype=first.dtype)
    return averaged
