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

"""FedAdam streaming/in-place aggregate == reference, identical m/v.

`_reference_aggregate` is a verbatim copy of the CURRENT (pre-optimization)
`FedAdam.aggregate` method body, adapted only to be callable standalone
against a strategy instance (it reads/writes `strat._m`, `strat._v`,
`strat._t`, `strat.server_lr`, `strat.beta1`, `strat.beta2`, `strat.eps`
exactly as the real method does internally). It proves the optimized
in-place/streaming version in `fedadam.py` produces numerically identical
results, including the server-side Adam moment buffers (m/v) across
multiple rounds.

`aggregate` is side-effect-free by default (`consume=False`): the caller's
`ClientUpdate.delta` dicts are left intact. Only `consume=True` drains them
(streaming free-as-you-go), and doing so must not change the numbers.
"""
from __future__ import annotations

import torch

from decent_vla.core.strategy import ClientUpdate
from decent_vla.strategies.fedadam import FedAdam


def _mk_global():
    return {"a": torch.randn(32, 16), "b": torch.randn(48)}


def _mk_updates(seed):
    g = torch.Generator().manual_seed(seed)
    return [
        ClientUpdate(
            delta={"a": torch.randn(32, 16, generator=g), "b": torch.randn(48, generator=g)},
            num_samples=97840,
            client_id="c0",
        ),
        ClientUpdate(
            delta={"a": torch.randn(32, 16, generator=g), "b": torch.randn(48, generator=g)},
            num_samples=84257,
            client_id="c1",
        ),
    ]


def _reference_aggregate(strat, client_updates, global_state):
    """EXACT copy of the CURRENT fedadam.aggregate body BEFORE the in-place
    streaming optimization (kept verbatim so the identity test has a fixed
    ground truth independent of the file under test)."""
    if not client_updates:
        return global_state

    # 1) Weighted-mean of client deltas (same as FedAvg's aggregate).
    total = sum(u.num_samples for u in client_updates)
    if total == 0:
        return global_state
    delta: dict[str, torch.Tensor] = {}
    for update in client_updates:
        w = update.num_samples / total
        for k, d in update.delta.items():
            if k not in global_state:
                continue
            accum = delta.get(k)
            # Upcast to the fp32 master dtype/device first, *then* weight —
            # multiplying in the client dtype would round before the upcast.
            contrib = w * d.to(global_state[k].dtype).to(global_state[k].device)
            delta[k] = contrib if accum is None else accum + contrib

    # 2) Adam-style server update.
    strat._t += 1
    new_state = {k: v.detach().clone() for k, v in global_state.items()}
    for k, d in delta.items():
        m_prev = strat._m.get(k)
        m_prev = torch.zeros_like(d) if m_prev is None else m_prev.to(device=d.device, dtype=d.dtype)
        v_prev = strat._v.get(k)
        v_prev = torch.zeros_like(d) if v_prev is None else v_prev.to(device=d.device, dtype=d.dtype)
        m = strat.beta1 * m_prev + (1.0 - strat.beta1) * d
        v = strat.beta2 * v_prev + (1.0 - strat.beta2) * d.pow(2)
        strat._m[k] = m
        strat._v[k] = v
        m_hat = m / (1.0 - strat.beta1**strat._t)
        v_hat = v / (1.0 - strat.beta2**strat._t)
        new_state[k] = new_state[k] + strat.server_lr * m_hat / (v_hat.sqrt() + strat.eps)
    return new_state


def test_fedadam_lean_identical_across_rounds():
    """Numerical identity vs. the pre-optimization reference across 2 rounds,
    including server moment buffers (m/v) and step counter. Called with
    consume=True since the reference is written as a single-shot drain-style
    accumulation (its `delta` dict is fully local/temporary); consume-ness
    doesn't affect the math, only whether the caller's `ClientUpdate.delta`
    survives — see test_fedadam_lean_consume_flag_drains_without_changing_values."""
    torch.manual_seed(0)
    g = _mk_global()
    strat_ref = FedAdam(server_lr=0.1, betas=(0.9, 0.999), eps=1e-3)
    strat_new = FedAdam(server_lr=0.1, betas=(0.9, 0.999), eps=1e-3)

    for rnd in range(2):
        ups_ref = _mk_updates(rnd)
        ups_new = [
            ClientUpdate(
                delta={k: v.clone() for k, v in u.delta.items()},
                num_samples=u.num_samples,
                client_id=u.client_id,
            )
            for u in ups_ref
        ]
        g_ref = {k: v.clone() for k, v in g.items()}
        g_new = {k: v.clone() for k, v in g.items()}

        expected = _reference_aggregate(strat_ref, ups_ref, g_ref)
        got = strat_new.aggregate(ups_new, g_new, consume=True)

        for k in expected:
            assert torch.allclose(got[k], expected[k], rtol=1e-5, atol=1e-6), (rnd, k)

        # Deltas must be drained/freed by the streaming implementation when
        # consume=True.
        assert all(len(u.delta) == 0 for u in ups_new), f"round {rnd}: deltas not drained"

        # m/v server state must match across rounds too.
        assert set(strat_ref._m.keys()) == set(strat_new._m.keys())
        for k in strat_ref._m:
            assert torch.allclose(strat_ref._m[k], strat_new._m[k], rtol=1e-5, atol=1e-6), (rnd, "m", k)
            assert torch.allclose(strat_ref._v[k], strat_new._v[k], rtol=1e-5, atol=1e-6), (rnd, "v", k)
        assert strat_ref._t == strat_new._t

        # Feed round-1 output back in as round-2's global_state (cloned
        # independently for ref vs new to avoid aliasing across the loop).
        g = expected


def test_fedadam_lean_consume_flag_drains_without_changing_values():
    """consume=True drains each client's delta dict; consume=False (the
    default) leaves it intact. Both must produce identical aggregation
    results and identical server moment-buffer updates."""
    torch.manual_seed(7)
    g = _mk_global()

    ups_a = _mk_updates(0)  # consume=False path
    ups_b = [
        ClientUpdate(
            delta={k: v.clone() for k, v in u.delta.items()},
            num_samples=u.num_samples,
            client_id=u.client_id,
        )
        for u in ups_a
    ]  # consume=True path

    snap = [{k: d.clone() for k, d in u.delta.items()} for u in ups_a]

    g_a = {k: v.clone() for k, v in g.items()}
    g_b = {k: v.clone() for k, v in g.items()}

    strat_a = FedAdam(server_lr=0.1, betas=(0.9, 0.999), eps=1e-3)
    strat_b = FedAdam(server_lr=0.1, betas=(0.9, 0.999), eps=1e-3)

    got_a = strat_a.aggregate(ups_a, g_a)               # default: consume=False
    got_b = strat_b.aggregate(ups_b, g_b, consume=True)  # explicit opt-in

    # Numerically identical regardless of consume.
    for k in got_a:
        assert torch.allclose(got_a[k], got_b[k], rtol=1e-5, atol=1e-6), k
    for k in strat_a._m:
        assert torch.allclose(strat_a._m[k], strat_b._m[k], rtol=1e-5, atol=1e-6), ("m", k)
        assert torch.allclose(strat_a._v[k], strat_b._v[k], rtol=1e-5, atol=1e-6), ("v", k)
    assert strat_a._t == strat_b._t

    # consume=False must not touch the caller's deltas.
    for i, u in enumerate(ups_a):
        assert set(u.delta.keys()) == set(snap[i].keys())
        for k in snap[i]:
            assert torch.equal(u.delta[k], snap[i][k]), f"update[{i}].delta[{k}] mutated under consume=False"

    # consume=True must drain the caller's deltas.
    assert all(len(u.delta) == 0 for u in ups_b), "consume=True did not drain deltas"
