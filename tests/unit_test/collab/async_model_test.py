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

import pytest

from nvflare.collab.async_engine import AsyncEngine, AsyncResult

torch = pytest.importorskip("torch")
from examples.advanced.collab.pt_async_cifar10.model import PTModelAdapter


def test_detached_cpu_snapshots_and_nonfloating_buffers():
    initial = {"w": torch.tensor([10.0], requires_grad=True), "counter": torch.tensor(5)}
    engine = AsyncEngine(initial, PTModelAdapter(), 1, 2, buffer_size=2, weight_metric="num_steps")
    initial["w"].detach().fill_(99)
    a, b = engine.dispatch(["a", "b"])
    first = {"w": torch.tensor([12.0]), "counter": torch.tensor(99)}
    second = {"w": torch.tensor([16.0]), "counter": torch.tensor(100)}
    assert not engine.accept(a, AsyncResult(a.assignment_id, 0, first, {"num_steps": 1}))
    first["w"].fill_(999)
    assert engine.accept(b, AsyncResult(b.assignment_id, 0, second, {"num_steps": 3}))
    assert engine.model["w"].item() == 15
    assert engine.model["counter"].item() == 5
    assert not engine.model["w"].requires_grad
    assert engine.model["w"].device.type == "cpu"


def test_low_precision_weighted_updates_accumulate_in_float32():
    dtype, buffer_size, is_delta = torch.float16, 2, True
    initial = {"w": torch.zeros(1, dtype=dtype), "counter": torch.tensor(5)}
    engine = AsyncEngine(initial, PTModelAdapter(), 1, buffer_size, buffer_size=buffer_size, weight_metric="num_steps")
    assignments = engine.dispatch([f"site-{i}" for i in range(buffer_size)])
    for index, assignment in enumerate(assignments):
        payload = {"w": torch.tensor([100.0 * (index + 1)], dtype=dtype)}
        if not is_delta:
            payload["counter"] = torch.tensor(99)
        published = engine.accept(
            assignment,
            AsyncResult(
                assignment.assignment_id,
                0,
                payload,
                {"num_steps": 1000},
                is_delta=is_delta,
            ),
        )
        if not published:
            assert engine._buffer.total["w"].dtype == torch.float32
            assert torch.isfinite(engine._buffer.total["w"]).all()
        assert payload["w"].dtype == dtype
    assert engine.model["w"].item() == 100.0 * (buffer_size + 1) / 2
    assert engine.model["w"].dtype == dtype
    assert engine.model["counter"].item() == 5
