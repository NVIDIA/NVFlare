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

"""Shared type aliases used across the framework. No runtime logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

# A `Batch` is whatever the backend's dataloader yields. The framework never
# introspects this — it only does `next(loader)` then `backend.compute_loss(batch)`.
# lerobot passes dict[str, Tensor], groot passes its own structure. Permissive on purpose.
Batch = dict[str, Any]

# A `LossDict` must contain "loss" (the backward target). Other keys are
# metrics (numeric tensors) and are logged. Strategies modify "loss" inside
# Strategy.modify_loss.
LossDict = dict[str, torch.Tensor]

# `ParamScope` tells get_state / set_state which params to handle.
#   "shared"   — params that participate in FL aggregation
#   "personal" — per-client private params (e.g. Ditto's local head)
#   "all"      — both, for checkpointing
ParamScope = Literal["shared", "personal", "all"]


@dataclass
class CheckpointSpec:
    """Tells the backend what to save and how."""

    include: ParamScope | Literal["lora_only"] = "all"
    extra_state: dict[str, Any] = field(default_factory=dict)
    # Backends may use this to record optimizer state, scheduler state, etc.
