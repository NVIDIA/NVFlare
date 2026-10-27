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

"""Streaming in-place FedAvg aggregate == old out-of-place, and drains deltas."""
from __future__ import annotations

import torch

from decent_vla.core.strategy import ClientUpdate, StrategyBase


def _old_aggregate(client_updates, global_state):
    total = sum(u.num_samples for u in client_updates)
    new_state = {k: v.detach().clone() for k, v in global_state.items()}
    for u in client_updates:
        w = u.num_samples / total
        for k, d in u.delta.items():
            if k in new_state:
                new_state[k] = new_state[k] + w * d.to(new_state[k].dtype).to(new_state[k].device)
    return new_state


def _mk():
    g = {"a": torch.randn(64, 64), "b": torch.randn(128)}
    ups = [
        ClientUpdate(delta={"a": torch.randn(64, 64), "b": torch.randn(128)}, num_samples=97840, client_id="c0"),
        ClientUpdate(delta={"a": torch.randn(64, 64), "b": torch.randn(128)}, num_samples=84257, client_id="c1"),
    ]
    return g, ups


def test_lean_aggregate_matches_old():
    torch.manual_seed(0)
    g, ups = _mk()
    g2 = {k: v.clone() for k, v in g.items()}
    ups_old = [ClientUpdate(delta={k: d.clone() for k, d in u.delta.items()}, num_samples=u.num_samples, client_id=u.client_id) for u in ups]
    expected = _old_aggregate(ups_old, g2)
    got = StrategyBase().aggregate(ups, g, consume=True)
    for k in expected:
        assert torch.allclose(got[k], expected[k], atol=1e-6), k


def test_lean_aggregate_drains_deltas():
    torch.manual_seed(1)
    g, ups = _mk()
    StrategyBase().aggregate(ups, g, consume=True)
    assert all(len(u.delta) == 0 for u in ups)


def test_lean_aggregate_default_does_not_drain():
    """`aggregate` must be side-effect-free by default: callers that don't
    opt into `consume=True` keep their `ClientUpdate.delta` dicts intact, and
    the numerical result is identical to the consuming path."""
    torch.manual_seed(3)
    g, ups = _mk()
    snap = [{k: d.clone() for k, d in u.delta.items()} for u in ups]

    g_consume = {k: v.clone() for k, v in g.items()}
    ups_consume = [ClientUpdate(delta={k: d.clone() for k, d in u.delta.items()}, num_samples=u.num_samples, client_id=u.client_id) for u in ups]
    expected = StrategyBase().aggregate(ups_consume, g_consume, consume=True)

    got = StrategyBase().aggregate(ups, g)  # default consume=False

    for i, u in enumerate(ups):
        assert set(u.delta.keys()) == set(snap[i].keys()), "delta dict was mutated"
        for k in snap[i]:
            assert torch.equal(u.delta[k], snap[i][k]), f"update[{i}].delta[{k}] was mutated"

    for k in expected:
        assert torch.allclose(got[k], expected[k], atol=1e-6), k


def test_inplace_aggregate_matches_copying_and_returns_same_dict():
    """`inplace=True` is the `fl.lean_host_memory` path. It must be numerically
    identical to the copying path while returning the CALLER's dict, mutated —
    that identity is precisely what removes a full fp32 master copy (13.5 GB at
    pi0.5's 3.62B params, the difference between fitting in a 125 GB host and
    not). Note the copying contract is still the method default: see
    test_lean_aggregate_does_not_mutate_global below, which pins it."""
    torch.manual_seed(4)
    g, ups = _mk()
    g_copy = {k: v.clone() for k, v in g.items()}
    ups_copy = [
        ClientUpdate(delta={k: d.clone() for k, d in u.delta.items()},
                     num_samples=u.num_samples, client_id=u.client_id)
        for u in ups
    ]

    expected = StrategyBase().aggregate(ups_copy, g_copy, consume=True)
    got = StrategyBase().aggregate(ups, g, consume=True, inplace=True)

    assert got is g, "inplace=True must return the caller's dict, not a copy"
    for k in expected:
        assert torch.equal(got[k], expected[k]), k


def test_lean_aggregate_does_not_mutate_global():
    torch.manual_seed(2)
    g, ups = _mk()
    snap = {k: v.clone() for k, v in g.items()}
    StrategyBase().aggregate(ups, g, consume=True)
    for k in g:
        assert torch.equal(g[k], snap[k]), f"global_state[{k}] was mutated"
