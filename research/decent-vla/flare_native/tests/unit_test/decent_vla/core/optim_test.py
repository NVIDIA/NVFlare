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

"""Tests for `decent_vla.core.optim.build_warmup_cosine`.

This schedule must match lerobot's get_scheduler cosine-decay-with-warmup
preset: linear warmup to peak lr, cosine decay to `decay_lr` over
`decay_steps`, constant floor after. A wrong schedule here is a known
parity killer (documented 15pp regression from an LR-schedule bug).
"""

from __future__ import annotations

import pytest
import torch

from decent_vla.core.optim import build_warmup_cosine, warmup_cosine_factor

PEAK_LR = 1.0e-4
DECAY_LR = 2.5e-5
WARMUP = 1000
DECAY_STEPS = 30000


def _make_opt(lr: float = PEAK_LR) -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _lr_at_step(scheduler_steps: int, **kwargs) -> float:
    """LR seen by the optimizer after `scheduler_steps` scheduler.step() calls."""
    opt = _make_opt(kwargs.get("lr", PEAK_LR))
    sched = build_warmup_cosine(
        opt,
        lr=kwargs.get("lr", PEAK_LR),
        warmup_steps=kwargs.get("warmup_steps", WARMUP),
        decay_steps=kwargs.get("decay_steps", DECAY_STEPS),
        decay_lr=kwargs.get("decay_lr", DECAY_LR),
    )
    for _ in range(scheduler_steps):
        sched.step()
    return opt.param_groups[0]["lr"]


def test_step0_factor_is_one_over_warmup():
    assert _lr_at_step(0) == pytest.approx(PEAK_LR / WARMUP)


def test_warmup_ramp_is_linear():
    assert _lr_at_step(499) == pytest.approx(PEAK_LR * 500 / WARMUP)


def test_peak_at_end_of_warmup():
    assert _lr_at_step(WARMUP) == pytest.approx(PEAK_LR)


def test_floor_at_decay_steps():
    assert _lr_at_step(DECAY_STEPS) == pytest.approx(DECAY_LR)


def test_constant_floor_after_decay_steps():
    assert _lr_at_step(DECAY_STEPS + 100) == pytest.approx(DECAY_LR)
    assert _lr_at_step(DECAY_STEPS + 100) == _lr_at_step(DECAY_STEPS)


def test_decay_lr_none_decays_to_zero():
    assert _lr_at_step(DECAY_STEPS, decay_lr=None) == pytest.approx(0.0)


def test_midpoint_is_cosine_halfway():
    mid = (WARMUP + DECAY_STEPS) // 2
    expected = DECAY_LR + (PEAK_LR - DECAY_LR) * 0.5
    assert _lr_at_step(mid) == pytest.approx(expected, rel=1e-3)


def test_zero_warmup_skips_ramp():
    assert _lr_at_step(0, warmup_steps=0) == pytest.approx(PEAK_LR)


def test_monotone_decay_after_warmup():
    factors = [
        warmup_cosine_factor(s, WARMUP, DECAY_STEPS, DECAY_LR / PEAK_LR) for s in range(WARMUP, DECAY_STEPS + 1, 1000)
    ]
    assert all(a >= b for a, b in zip(factors, factors[1:], strict=False))


def test_rejects_nonpositive_peak_lr():
    with pytest.raises(ValueError):
        build_warmup_cosine(_make_opt(), lr=0.0, warmup_steps=10, decay_steps=100)
