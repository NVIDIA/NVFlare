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

"""Shared LR-schedule helpers for adapters.

`build_warmup_cosine` reproduces lerobot's schedule
(lerobot's `get_scheduler` with the cosine-decay-with-warmup
preset): linear warmup to the peak LR, cosine decay from the peak down
to a floor (`decay_lr`) over `decay_steps`, then constant at the floor.

It is expressed as a multiplicative `LambdaLR` factor on the optimizer's
base LR, so the optimizer's `lr` *is* the peak LR.

"""

from __future__ import annotations

import math

import torch


def warmup_cosine_factor(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    floor: float,
) -> float:
    """Multiplicative LR factor at `step` (0-based, incremented per optimizer step).

    * step < warmup_steps           -> linear ramp ``(step + 1) / warmup_steps``
    * warmup_steps <= step < decay  -> cosine from 1.0 down to ``floor``
    * step >= decay_steps           -> ``floor`` (constant)

    ``floor`` is ``decay_lr / peak_lr`` (0.0 to decay all the way to zero).
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if step >= decay_steps or decay_steps <= warmup_steps:
        return floor
    progress = (step - warmup_steps) / (decay_steps - warmup_steps)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    lr: float,
    warmup_steps: int,
    decay_steps: int,
    decay_lr: float | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup + cosine-decay-to-floor scheduler (lerobot recipe parity).

    `lr` is the peak LR (must equal the optimizer's base lr); `decay_lr`
    is the constant floor reached at `decay_steps` (None -> decay to 0).
    `warmup_steps=0` skips the ramp (factor is 1.0 at step 0).
    """
    if lr <= 0:
        raise ValueError(f"peak lr must be positive, got {lr}")
    floor = 0.0 if decay_lr is None else decay_lr / lr

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_factor(step, warmup_steps, decay_steps, floor),
    )
