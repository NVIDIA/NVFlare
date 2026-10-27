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

from nvflare.collab.async_workflow import AsyncEngine, AsyncResult

torch = pytest.importorskip("torch")
from nvflare.collab.async_models.pytorch import PTAsyncModelAdapter


def test_detached_cpu_snapshots_and_nonfloating_buffers():
    initial = {"w": torch.tensor([10.0], requires_grad=True), "counter": torch.tensor(5)}
    engine = AsyncEngine(initial, PTAsyncModelAdapter(), 1, 2, buffer_size=2, weight_metric="num_steps")
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
