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

"""Codec round-trips against the fake nvflare FLModel (CPU, pure).

The nvflare modules are imported INSIDE
each test — the autouse fixture evicts them from sys.modules so they bind to
the fake nvflare namespace.
"""

from __future__ import annotations

import pytest
import torch

from decent_vla.core.strategy import ClientUpdate, RoundPlan


@pytest.fixture(autouse=True)
def _reset_wire_dtype():
    """The wire dtype is a PROCESS-global in the codec, so a test that sets it
    would otherwise leak its precision into every test that runs after it.
    Restore the documented default after each test."""
    yield
    from decent_vla.runtime.nvflare import codec

    codec.set_wire_dtype("fp16")


def test_state_roundtrip_fp32_is_lossless():
    from decent_vla.runtime.nvflare import codec

    codec.set_wire_dtype("fp32")
    state = {"w": torch.randn(3, 4), "b": torch.arange(5)}
    model = codec.round_flmodel(state, RoundPlan(selected_clients=["c0"], local_steps=1))
    back = codec.np_to_state(model.params)
    assert set(back) == {"w", "b"}
    assert torch.equal(back["w"], state["w"])
    assert torch.equal(back["b"], state["b"])   # int tensors are never cast


def test_broadcast_honours_wire_dtype_at_fp16_default():
    """The BROADCAST now honours fl.wire_dtype (it used to ship the fp32 master
    unconditionally); precision resolves through the runtime/wire.py table."""
    from decent_vla.runtime.nvflare import codec

    codec.set_wire_dtype("fp16")
    state = {"w": torch.randn(3, 4)}
    model = codec.round_flmodel(state, RoundPlan(selected_clients=["c0"], local_steps=1))
    assert model.params["w"].dtype.name == "float16"
    back = codec.np_to_state(model.params)
    assert torch.allclose(back["w"].float(), state["w"], rtol=2 ** -10, atol=1e-6)


def test_wire_dtype_per_direction_mapping():
    from decent_vla.runtime.nvflare import codec

    codec.set_wire_dtype({"broadcast": "fp32", "delta": "fp16"})
    state = {"w": torch.randn(4)}
    model = codec.round_flmodel(state, RoundPlan(selected_clients=["c0"], local_steps=1))
    assert model.params["w"].dtype.name == "float32"
    up = ClientUpdate(delta={"w": torch.randn(4)}, num_samples=1, client_id="c0")
    assert codec.update_to_flmodel(up).params["w"].dtype.name == "float16"


def test_consume_drains_source_and_matches_pure_path():
    from decent_vla.runtime.nvflare import codec

    codec.set_wire_dtype("fp16")
    src = {"w": torch.randn(8), "b": torch.randn(3)}
    pure = codec.state_to_np(dict(src))
    drainable = dict(src)
    drained = codec.state_to_np(drainable, consume=True)
    assert drainable == {}, "consume=True must drain the source dict"
    for k in src:
        assert (pure[k] == drained[k]).all()


def test_update_roundtrip_wire_dtype_fp16_default():
    from decent_vla.runtime.nvflare import codec

    up = ClientUpdate(
        delta={"w": torch.randn(2, 2)},
        num_samples=7,
        client_id="c1",
        metrics={"loss": 0.5, "note": "hi"},
    )
    model = codec.update_to_flmodel(up)
    # deltas travel at fl.wire_dtype (default fp16)
    assert model.params["w"].dtype.name == "float16"
    back = codec.flmodel_to_update(model)
    assert back.num_samples == 7
    assert back.client_id == "c1"
    assert back.metrics["loss"] == 0.5
    assert back.metrics["note"] == "hi"
    assert torch.allclose(back.delta["w"].float(), up.delta["w"].to(torch.float16).float())


def test_update_roundtrip_wire_dtype_fp32_lossless():
    from decent_vla.runtime.nvflare import codec

    codec.set_wire_dtype("fp32")
    up = ClientUpdate(delta={"w": torch.randn(2, 2)}, num_samples=1, client_id="c0")
    model = codec.update_to_flmodel(up)
    assert model.params["w"].dtype.name == "float32"
    back = codec.flmodel_to_update(model)
    assert torch.equal(back.delta["w"], up.delta["w"])


def test_int_deltas_not_cast():
    from decent_vla.runtime.nvflare import codec

    up = ClientUpdate(delta={"steps": torch.arange(4)}, num_samples=1, client_id="c0")
    model = codec.update_to_flmodel(up)
    back = codec.flmodel_to_update(model)
    assert torch.equal(back.delta["steps"], up.delta["steps"])


def test_plan_roundtrip():
    from decent_vla.runtime.nvflare import codec

    plan = RoundPlan(
        selected_clients=["c0", "c2"],
        local_steps=9,
        broadcast={"mu": 0.01},
        round_num=3,
    )
    state = {"w": torch.zeros(1)}
    model = codec.round_flmodel(state, plan)
    assert model.current_round == 3
    _, back = codec.flmodel_to_round(model)
    assert back.selected_clients == ["c0", "c2"]
    assert back.local_steps == 9
    assert back.broadcast == {"mu": 0.01}
    assert back.round_num == 3


def test_bf16_state_downcast_to_fp16():
    # numpy has no bfloat16 — the codec must force-downcast.
    from decent_vla.runtime.nvflare import codec

    state = {"w": torch.randn(2, 2, dtype=torch.bfloat16)}
    arr = codec.state_to_np(state)
    assert arr["w"].dtype.name == "float16"


def test_wire_dtype_rejects_bf16():
    from decent_vla.runtime.nvflare import codec

    with pytest.raises(ValueError, match="wire_dtype"):
        codec.set_wire_dtype("bf16")
