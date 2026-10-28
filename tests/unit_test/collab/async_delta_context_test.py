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

"""Optional update context must not change the arithmetic of a delta upload."""

import numpy as np
import pytest

from nvflare.collab import collab
from nvflare.collab.async_models.numpy import NumpyAsyncModelAdapter
from nvflare.collab.async_workflow import AsyncEngine, AsyncResult

from .async_recipe_test import _Application, _execute


@pytest.mark.parametrize("backend", ["numpy", "pytorch"])
@pytest.mark.parametrize("consumer", ["observer", "transform", "both"])
def test_delta_only_consumers_can_publish_when_unscaled_full_model_would_overflow(backend, consumer):
    if backend == "numpy":
        adapter = NumpyAsyncModelAdapter()
    else:
        torch = pytest.importorskip("torch")
        from nvflare.collab.async_models.pytorch import PTAsyncModelAdapter

        adapter = PTAsyncModelAdapter()

    def make(value):
        if backend == "numpy":
            return np.array([value], dtype=np.float16)
        return torch.tensor([value], dtype=torch.float16)

    seen = []

    def transform(update):
        # This transform only uses the client delta, not a teacher model.
        scale = 0.1 if consumer == "transform" else 1.0
        return {key: value * scale for key, value in update.delta.items()}

    engine = AsyncEngine(
        {"w": make(60000)},
        adapter,
        num_updates=1,
        num_active_jobs=1,
        server_lr=1.0 if consumer == "transform" else 0.1,
        update_transform=transform if consumer != "observer" else None,
        update_observer=(lambda update: seen.append(update.staleness)) if consumer != "transform" else None,
    )
    try:
        job = engine.dispatch(["client"])[0]
        assert engine.accept(job, AsyncResult(job.assignment_id, 0, {"w": make(10000)}, is_delta=True))
        assert engine.model["w"].item() == float(np.float16(61000))
        assert engine.model["w"].dtype == make(0).dtype
        assert seen == ([] if consumer == "transform" else [0])
    finally:
        engine.close()


class _CountingAdapter(NumpyAsyncModelAdapter):
    def __init__(self):
        self.applies = 0

    def apply(self, model, delta, scale):
        self.applies += 1
        return super().apply(model, delta, scale)


@pytest.mark.parametrize("is_delta", [True, False])
def test_full_model_is_requested_once_and_uses_original_client_delta(is_delta):
    adapter = _CountingAdapter()
    seen = []

    def transform(update):
        assert adapter.applies == 0
        return {name: value * 0.5 for name, value in update.delta.items()}

    def observe(update):
        first = update.returned_model
        assert update.returned_model is first
        assert first["w"].item() == 12  # Original +2, before the transform halves it.
        assert adapter.applies == int(is_delta)
        seen.append(first)

    engine = AsyncEngine(
        {"w": np.array([10], dtype=np.float32)}, adapter, 1, 1, update_transform=transform, update_observer=observe
    )
    try:
        job = engine.dispatch(["client"])[0]
        payload = {"w": np.array([2 if is_delta else 12], dtype=np.float32)}
        assert engine.accept(job, AsyncResult(job.assignment_id, 0, payload, is_delta=is_delta))
        assert engine.model["w"].item() == 11
        assert seen[0]["w"].item() == 12
        assert adapter.applies == 1 + int(is_delta)
    finally:
        engine.close()


def test_explicit_unrepresentable_full_model_request_still_raises_without_accepting():
    def teacher_transform(update):
        _ = update.returned_model
        return update.delta

    engine = AsyncEngine(
        {"w": np.array([60000], dtype=np.float16)},
        NumpyAsyncModelAdapter(),
        1,
        1,
        server_lr=0.1,
        update_transform=teacher_transform,
    )
    try:
        job = engine.dispatch(["client"])[0]
        with np.errstate(over="ignore"), pytest.raises(ValueError, match="non-finite"):
            engine.accept(
                job, AsyncResult(job.assignment_id, 0, {"w": np.array([10000], dtype=np.float16)}, is_delta=True)
            )
        assert engine.accepted_count == engine.buffered_count == engine.version == 0
        assert engine.model["w"].item() == 60000
    finally:
        engine.close()


class _DeltaApplication(_Application):
    def __init__(self, observe):
        super().__init__()
        self.observed = []
        if observe:
            self.on_update = lambda update: self.observed.append(update.staleness)

    def initialize(self):
        return {"w": np.array([60000], dtype=np.float16)}


class _DeltaClient:
    @collab.publish
    def train(self, assignment_ids, model_version, global_model, logical_assignments):
        site = collab.site_name
        return (
            {"w": np.array([10000], dtype=np.float16)},
            {},
            {
                "assignment_id": assignment_ids[site],
                "model_version": model_version,
                "logical_name": logical_assignments[site],
                "is_delta": True,
            },
        )


@pytest.mark.parametrize("observe", [False, True])
def test_collab_delta_upload_with_optional_observer_does_not_reconstruct_full_model(observe):
    application = _DeltaApplication(observe)
    result = _execute(application, [_DeltaClient()], num_updates=1, server_lr=0.1)
    assert result["w"].item() == float(np.float16(61000))
    assert application.observed == ([0] if observe else [])
    assert application.closed and application.finished
