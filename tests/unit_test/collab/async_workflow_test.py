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

"""Numerical and lifecycle contracts for the framework-independent async engine."""

import numpy as np
import pytest

from nvflare.collab.async_models.numpy import NumpyAsyncModelAdapter
from nvflare.collab.async_workflow import AsyncEngine, AsyncResult


def model(value):
    return {"w": np.array([value], dtype=np.float32), "counter": np.array(7, dtype=np.int64)}


def engine(**kwargs):
    options = dict(
        initial_model=model(10),
        model_adapter=NumpyAsyncModelAdapter(),
        num_updates=3,
        num_active_jobs=2,
        buffer_size=1,
        min_open_slots=1,
    )
    options.update(kwargs)
    return AsyncEngine(**options)


def result(job, value, steps=1):
    return AsyncResult(job.assignment_id, job.base_version, model(value), {"num_steps": steps})


def test_stale_return_uses_assignment_base_and_publication_precedes_refill():
    run = engine()
    fast, slow = run.dispatch(["fast", "slow"])
    assert fast.base_model is slow.base_model
    assert run.accept(fast, result(fast, 12))
    refill = run.dispatch(["fast"])[0]
    assert refill.base_version == 1
    np.testing.assert_array_equal(refill.base_model["w"], [12])
    assert run.accept(slow, result(slow, 13))  # +3 against 10, not +1 against 12
    np.testing.assert_array_equal(run.model["w"], [15])
    assert run.accept(refill, result(refill, 16))  # +4 against 12
    np.testing.assert_array_equal(run.model["w"], [19])
    assert run.finished and run.dispatch(["fast", "slow"]) == []
    assert run.model["counter"] == 7


def test_step_weighted_buffer_matches_cifar_formula_and_does_not_retain_models():
    run = engine(num_updates=1, buffer_size=2, weight_metric="num_steps", server_lr=0.5)
    a, b = run.dispatch(["a", "b"])
    returned = result(a, 12, steps=1)
    assert not run.accept(a, returned)
    returned.model["w"][:] = 1000  # accumulation owns its weighted copy
    np.testing.assert_array_equal(run.model["w"], [10])
    assert run.accept(b, result(b, 16, steps=3))
    np.testing.assert_array_equal(run.model["w"], [12.5])  # 10 + .5*(2+18)/4
    assert run.buffered_count == 0 and not run.active
    assert run.accepted_count == 2


def test_k_b_o_sync_boundary_and_exact_budget():
    run = engine(num_updates=2, buffer_size=2, min_open_slots=2)
    a, b = run.dispatch(["a", "b"])
    assert not run.accept(a, result(a, 11))
    assert run.dispatch(["a"]) == []
    assert run.accept(b, result(b, 13))
    c, d = run.dispatch(["a", "b"])
    assert c.base_version == d.base_version == 1
    assert not run.accept(c, result(c, 13))
    assert run.accept(d, result(d, 15))
    np.testing.assert_array_equal(run.model["w"], [14])
    assert run.dispatch(["a", "b"]) == []


def test_b_can_exceed_k_and_tail_does_not_overdispatch():
    run = engine(num_updates=1, buffer_size=3)
    a, b = run.dispatch(["a", "b"])
    run.accept(a, result(a, 11))
    c = run.dispatch(["a"])[0]
    run.accept(b, result(b, 12))
    assert run.dispatch(["b"]) == []  # only one more contribution is needed
    assert run.accept(c, result(c, 13))
    np.testing.assert_allclose(run.model["w"], [12])


def test_late_duplicate_cannot_retire_new_assignment():
    run = engine()
    a, b = run.dispatch(["a", "b"])
    run.accept(a, result(a, 11))
    current = run.dispatch(["a"])[0]
    assert not run.accept(a, result(a, 100))
    assert run.active[current.assignment_id] is current and run.accepted_count == 1
    run.close()
    assert not run.accept(b, result(b, 100))
    assert not run.active and run.buffered_count == 0


def test_transform_observes_staleness_and_can_be_omitted():
    seen = []

    def transform(update):
        seen.append((update.staleness, float(update.current_model["w"][0]), update.weight))
        return {key: value * 0.5 for key, value in update.delta.items()}

    run = engine(num_updates=2, update_transform=transform)
    a, b = run.dispatch(["a", "b"])
    run.accept(a, result(a, 12))
    run.accept(b, result(b, 14))
    assert seen == [(0, 10, 1), (1, 11, 1)]
    np.testing.assert_array_equal(run.model["w"], [13])


def test_explicit_delta_does_not_require_server_to_difference_it():
    run = engine(num_updates=1)
    job = run.dispatch(["a"])[0]
    value = AsyncResult(job.assignment_id, job.base_version, {"w": np.array([3], dtype=np.float32)}, is_delta=True)
    assert run.accept(job, value)
    np.testing.assert_array_equal(run.model["w"], [13])


@pytest.mark.parametrize("is_delta", [False, True])
def test_weighted_scalar_arrays_preserve_shape_and_dtype(is_delta):
    run = engine(
        initial_model={"scalar": np.array(10, dtype=np.float32)},
        num_updates=1,
        buffer_size=2,
        weight_metric="num_steps",
    )
    a, b = run.dispatch(["a", "b"])
    for assignment, value, steps in ((a, 12, 1), (b, 16, 3)):
        payload = np.array(value - 10 if is_delta else value, dtype=np.float32)
        run.accept(
            assignment,
            AsyncResult(assignment.assignment_id, 0, {"scalar": payload}, {"num_steps": steps}, is_delta=is_delta),
        )
    assert run.model["scalar"].shape == ()
    assert run.model["scalar"].dtype == np.float32
    assert run.model["scalar"].item() == 15


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_weights_do_not_count(weight):
    run = engine(weight_metric="num_steps")
    job = run.dispatch(["a"])[0]
    with pytest.raises(ValueError, match="weight"):
        run.accept(job, result(job, 11, weight))
    assert run.accepted_count == run.buffered_count == 0
    assert run.active[job.assignment_id] is job


@pytest.mark.parametrize("case", ["metadata", "keys", "shape", "dtype", "nan", "none", "failure"])
def test_invalid_result_is_fail_fast_without_partial_acceptance(case):
    run = engine()
    job = run.dispatch(["a"])[0]
    value = result(job, 11)
    if case == "metadata":
        value = AsyncResult(99, 0, model(11))
    elif case == "keys":
        value.model.pop("counter")
    elif case == "shape":
        value.model["w"] = np.ones(2, dtype=np.float32)
    elif case == "dtype":
        value.model["w"] = np.ones(1, dtype=np.float64)
    elif case == "nan":
        value.model["w"][:] = np.nan
    elif case == "none":
        value = None
    with pytest.raises((ValueError, RuntimeError)):
        run.accept(job, value, error="transport timeout" if case == "failure" else None)
    assert run.accepted_count == run.buffered_count == run.version == 0
    run.close()
    assert not run.active


def test_close_discards_partial_buffer_and_releases_snapshots():
    run = engine(buffer_size=2)
    a, _ = run.dispatch(["a", "b"])
    run.accept(a, result(a, 12))
    run.close()
    assert not run.active and run.buffered_count == 0
    np.testing.assert_array_equal(run.model["w"], [10])


@pytest.mark.parametrize(
    "options",
    [
        {"num_updates": 0},
        {"num_active_jobs": True},
        {"buffer_size": 1.5},
        {"min_open_slots": 3},
        {"server_lr": float("inf")},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        engine(**options)


@pytest.mark.parametrize("is_delta", [False, True])
def test_numpy_float16_weighted_updates_do_not_overflow(is_delta):
    run = engine(
        initial_model={"w": np.zeros(1, dtype=np.float16)}, num_updates=1, buffer_size=2, weight_metric="num_steps"
    )
    for index, assignment in enumerate(run.dispatch(["a", "b"])):
        payload = {"w": np.array([100.0 * (index + 1)], dtype=np.float16)}
        published = run.accept(
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
            assert run._buffer.total["w"].dtype == np.float32
            assert np.isfinite(run._buffer.total["w"]).all()
        assert payload["w"].dtype == np.float16
    assert run.model["w"].item() == 150
    assert run.model["w"].dtype == np.float16
