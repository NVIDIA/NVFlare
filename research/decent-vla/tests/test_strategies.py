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

"""Algorithmic-correctness tests for FedAvg, FedAdam, FedProx using MockBackend."""
from __future__ import annotations

import torch

from decent_vla.core.strategy import ClientContext, ClientUpdate
from decent_vla.strategies import FedAdam, FedAvg, FedProx

# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def _make_global_state():
    return {
        "w": torch.zeros(4),
        "b": torch.zeros(4),
    }


def _make_client_update(delta_w: float, delta_b: float, n: int):
    return ClientUpdate(
        delta={
            "w": torch.full((4,), float(delta_w)),
            "b": torch.full((4,), float(delta_b)),
        },
        num_samples=n,
        metrics={"loss": 0.1},
    )


def test_fedavg_aggregate_weighted_mean():
    fedavg = FedAvg()
    gs = _make_global_state()
    updates = [
        _make_client_update(1.0, -1.0, n=10),
        _make_client_update(3.0, -3.0, n=30),
    ]
    new = fedavg.aggregate(updates, gs)
    # weighted mean delta: w = (10*1 + 30*3)/40 = 2.5
    assert torch.allclose(new["w"], torch.full((4,), 2.5))
    assert torch.allclose(new["b"], torch.full((4,), -2.5))


def test_fedavg_aggregate_no_updates_returns_global():
    fedavg = FedAvg()
    gs = _make_global_state()
    new = fedavg.aggregate([], gs)
    assert new is gs


def test_fedadam_aggregate_takes_adam_step():
    """First call should produce a sign-matching update of magnitude ≈ server_lr."""
    fedadam = FedAdam(server_lr=0.1, betas=(0.9, 0.999), eps=1e-3)
    gs = _make_global_state()
    updates = [_make_client_update(1.0, 0.0, n=1)]
    new = fedadam.aggregate(updates, gs)
    # First step: m_hat = d, v_hat = d² → step ≈ lr · d / (|d| + eps) ≈ lr · sign(d)
    expected_step_w = 0.1 * 1.0 / (1.0 + 1e-3)
    assert torch.allclose(new["w"], torch.full((4,), expected_step_w), atol=1e-4)
    assert torch.allclose(new["b"], torch.full((4,), 0.0), atol=1e-4)


def test_fedadam_state_accumulates_across_rounds():
    fedadam = FedAdam(server_lr=0.1)
    gs = _make_global_state()
    for _ in range(3):
        fedadam.aggregate([_make_client_update(1.0, 0.0, n=1)], gs)
    # m and v should have accumulated values for `w`
    assert "w" in fedadam._m
    assert "w" in fedadam._v
    assert fedadam._t == 3


def test_fedadam_state_dict_roundtrip_survives_resume():
    """Persisting + restoring the strategy state must reproduce the exact next
    step — the regression guard for cold-restarted server moments on resume."""
    gs = _make_global_state()
    warm = FedAdam(server_lr=0.1)
    # Large warm-up deltas so the accumulated 2nd moment v is far from 1: this
    # makes the next step depend on the restored moments (a constant delta would
    # saturate Adam to lr·sign(d) and hide whether the state was restored).
    for _ in range(3):
        warm.aggregate([_make_client_update(5.0, 0.0, n=1)], gs)

    # Simulate a resume: serialize, build a fresh instance, restore.
    blob = warm.state_dict()
    assert blob["t"] == 3 and "w" in blob["m"] and "w" in blob["v"]
    resumed = FedAdam(server_lr=0.1)
    resumed.load_state_dict(blob)
    assert resumed._t == 3

    # The next aggregation step must match what the warm (never-reset) instance does.
    upd = [_make_client_update(1.0, 0.0, n=1)]
    warm_next = warm.aggregate(upd, gs)
    resumed_next = resumed.aggregate(upd, gs)
    assert torch.allclose(warm_next["w"], resumed_next["w"])
    # A cold instance (moments zeroed, t=0) takes a visibly different step —
    # this is exactly the resume discontinuity the checkpointing fixes.
    cold_next = FedAdam(server_lr=0.1).aggregate(upd, gs)
    assert not torch.allclose(cold_next["w"], resumed_next["w"])


# ---------------------------------------------------------------------------
# FedProx modify_loss
# ---------------------------------------------------------------------------


class _ProxModel(torch.nn.Module):
    """A tiny module whose LIVE parameter is what FedProx reads (with gradient).

    ``FedProx.modify_loss`` now reads ``ctx.model.named_parameters()`` directly
    (not the detached ``backend.get_state``), so the test model's parameter IS
    the ``current`` value and the proximal term flows gradient into it.
    """

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.w = torch.nn.Parameter(w.clone())


def _make_ctx(model, global_state, scratch=None):
    return ClientContext(
        client_id="c0",
        round_num=0,
        step_in_round=0,
        backend=None,  # type: ignore[arg-type]  # modify_loss no longer touches the backend
        model=model,
        optimizer=None,  # type: ignore[arg-type]
        scheduler=None,
        global_state=global_state,
        broadcast={},
        persistent_state={},
        scratch=scratch if scratch is not None else {},
    )


def test_fedprox_zero_mu_is_identity():
    fedprox = FedProx(mu=0.0)
    gs = {"w": torch.zeros(4)}
    model = _ProxModel(torch.ones(4))
    ctx = _make_ctx(model, gs)
    fedprox.on_round_start(ctx)
    loss_in = torch.tensor(2.0, requires_grad=True)
    out = fedprox.modify_loss({"loss": loss_in}, ctx)
    assert torch.allclose(out["loss"], loss_in)


def test_fedprox_adds_proximal_term_mean_normalized():
    """L_new = L_base + (μ/2) · mean(‖θ − θ_global‖²) — mean by default."""
    fedprox = FedProx(mu=0.5)
    gs = {"w": torch.zeros(4)}                 # global = zeros
    model = _ProxModel(torch.ones(4))          # current = ones → sq diff = 4, count = 4, mean = 1
    ctx = _make_ctx(model, gs)
    fedprox.on_round_start(ctx)
    loss_in = torch.tensor(1.0, requires_grad=True)
    out = fedprox.modify_loss({"loss": loss_in}, ctx)
    # Expected: 1.0 + 0.5 · 0.5 · 1.0 = 1.25
    assert torch.allclose(out["loss"], torch.tensor(1.25))
    assert "prox_term" in out


def test_fedprox_proximal_term_flows_gradient():
    """The proximal term MUST produce a nonzero gradient on the live params.

    This is the regression guard for the historic no-op: reading detached
    ``get_state`` tensors made ``prox_term`` a constant w.r.t. the parameters,
    so ``backward()`` yielded exactly the FedAvg gradient. Here the base loss is
    a constant (no grad path), so any gradient on ``w`` comes solely from the
    proximal term: d/dw [ (μ/2)·mean((w−0)²) ] = μ·w/N = 0.5·1/4 = 0.125.
    """
    fedprox = FedProx(mu=0.5)
    gs = {"w": torch.zeros(4)}
    model = _ProxModel(torch.ones(4))
    ctx = _make_ctx(model, gs)
    fedprox.on_round_start(ctx)
    out = fedprox.modify_loss({"loss": torch.tensor(1.0)}, ctx)  # constant base loss
    assert out["loss"].requires_grad
    out["loss"].backward()
    assert model.w.grad is not None
    assert torch.count_nonzero(model.w.grad) == 4
    assert torch.allclose(model.w.grad, torch.full((4,), 0.125))


def test_fedprox_ignores_frozen_and_unshared_params():
    """Frozen params and params absent from the global snapshot don't contribute."""
    fedprox = FedProx(mu=0.5, normalize=False)
    gs = {"w": torch.zeros(4)}  # only 'w' is shared; 'frozen' is not in the snapshot
    model = _ProxModel(torch.ones(4))
    model.frozen = torch.nn.Parameter(torch.full((4,), 5.0), requires_grad=False)
    ctx = _make_ctx(model, gs)
    fedprox.on_round_start(ctx)
    out = fedprox.modify_loss({"loss": torch.tensor(1.0)}, ctx)
    # Only 'w' counts: sum sq diff = 4 → 1.0 + 0.5·0.5·4 = 2.0 (frozen's 5² ignored)
    assert torch.allclose(out["loss"], torch.tensor(2.0))


def test_fedprox_sum_mode_when_normalize_false():
    """Setting normalize=False reproduces the paper-canonical SUM formulation."""
    fedprox = FedProx(mu=0.5, normalize=False)
    gs = {"w": torch.zeros(4)}
    model = _ProxModel(torch.ones(4))  # sq diff sum = 4
    ctx = _make_ctx(model, gs)
    fedprox.on_round_start(ctx)
    out = fedprox.modify_loss({"loss": torch.tensor(1.0)}, ctx)
    # Expected: 1.0 + 0.5 · 0.5 · 4 = 2.0
    assert torch.allclose(out["loss"], torch.tensor(2.0))


def test_fedprox_scales_linearly_with_mu():
    """Doubling μ doubles the added term (with the same param distance)."""
    gs = {"w": torch.zeros(4)}

    fp1 = FedProx(mu=0.1)
    ctx1 = _make_ctx(_ProxModel(torch.ones(4)), gs)
    fp1.on_round_start(ctx1)
    out1 = fp1.modify_loss({"loss": torch.tensor(1.0)}, ctx1)

    fp2 = FedProx(mu=0.2)
    ctx2 = _make_ctx(_ProxModel(torch.ones(4)), gs)
    fp2.on_round_start(ctx2)
    out2 = fp2.modify_loss({"loss": torch.tensor(1.0)}, ctx2)

    added1 = (out1["loss"] - 1.0).item()
    added2 = (out2["loss"] - 1.0).item()
    assert abs(added2 - 2 * added1) < 1e-6, f"{added1=}  {added2=}"
