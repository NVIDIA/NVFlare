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

"""Exercise scheduling/storage extension contracts with real Collab groups."""

import json
import pickle
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from nvflare.collab import AsyncRecipe, collab
from nvflare.collab.async_models.numpy import NumpyAsyncModelAdapter
from nvflare.collab.async_scheduling import ReplayScheduler
from nvflare.collab.async_storage import MemoryResultStore, MemorySnapshotStore
from nvflare.collab.async_workflow import AsyncEngine, AsyncResult

from .async_recipe_test import _Application, _Client, _execute, _numpy_module


class _TraceApplication(_Application):
    def __init__(self):
        super().__init__()
        self.events = []
        self.updates = []

    def on_event(self, event):
        self.events.append(asdict(event))

    def on_update(self, update):
        self.updates.append(
            (
                update.assignment.logical_name,
                update.staleness,
                update.metadata.copy(),
                update.returned_model["w"].copy(),
            )
        )


class _GateReplay(ReplayScheduler):
    def __init__(self, gate, early_id):
        super().__init__([1, 0, 3, 2])
        self.gate = gate
        self.early_id = early_id
        self.completions = []

    def on_result(self, call):
        self.completions.append(call.assignment.assignment_id)
        super().on_result(call)
        if call.assignment.assignment_id == self.early_id:
            self.gate.set()


@pytest.mark.parametrize("early_id", [0, 1])
def test_replay_fixes_acceptance_and_bases_when_physical_completion_order_changes(early_id):
    gate = threading.Event()
    scheduler = _GateReplay(gate, early_id)
    clients = [_Client(release=gate if index != early_id else None) for index in range(2)]
    application = _TraceApplication()
    output = _execute(application, clients, scheduler=scheduler, num_updates=4)
    accepted = [event for event in application.events if event["phase"] == "accept"]
    assert scheduler.completions[0] == early_id
    assert [event["assignment_id"] for event in accepted] == [1, 0, 3, 2]
    assert [event["base_version"] for event in accepted] == [0, 0, 2, 1]
    assert [update[1] for update in application.updates] == [0, 1, 0, 2]
    assert output["w"].item() == 4
    # Trace entries are JSON-ready and do not retain model/snapshot handles.
    json.dumps(application.events)


class _Seeds:
    def __init__(self, offset=100):
        self.offset = offset

    def __call__(self, call):
        seed = call.assignment.metadata["train_seed"] + self.offset
        return {"train_seeds": {call.client_name: seed}}


class _SeedClient:
    def __init__(self):
        self.seen = []

    @collab.publish
    def train(self, assignment_ids, model_version, global_model, logical_assignments, train_seeds):
        site = collab.site_name
        identity = assignment_ids[site]
        seed = train_seeds[site]
        self.seen.append((identity, logical_assignments[site], seed))
        return (
            {"w": global_model["w"] + seed},
            {},
            {
                "assignment_id": identity,
                "model_version": model_version,
                "logical_name": logical_assignments[site],
                "train_seed": seed,
            },
        )


@pytest.mark.parametrize("workers", [1, 2])
def test_logical_replay_is_independent_of_worker_count_and_transmits_assignment_seeds(workers):
    order = [2, 0, 4, 1, 5, 3, 8, 6, 10, 7, 11, 9]
    logical = [f"logical-{index % 2}" for index in range(12)]
    scheduler = ReplayScheduler(order, logical, [{"train_seed": index} for index in range(12)])
    application = _TraceApplication()
    clients = [_SeedClient() for _ in range(workers)]
    output = _execute(
        application,
        clients,
        scheduler=scheduler,
        train_kwargs=_Seeds(),
        num_active_jobs=3,
        num_updates=6,
        buffer_size=2,
    )
    accepted = [event for event in application.events if event["phase"] == "accept"]
    assert [event["assignment_id"] for event in accepted] == order
    assert [event["base_version"] for event in accepted] == [0, 0, 1, 0, 1, 0, 3, 2, 4, 2, 4, 3]
    assert sorted(item for client in clients for item in client.seen) == [
        (index, logical[index], index + 100) for index in range(12)
    ]
    assert [update[0] for update in application.updates] == [logical[index] for index in order]
    assert [update[2]["train_seed"] for update in application.updates] == [index + 100 for index in order]
    np.testing.assert_array_equal(output["w"], [633])


class _DiskSnapshots:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.refs = {}
        self.loads = 0
        self.closed = False

    def retain(self, version, model, adapter):
        root = Path(self.root_dir)
        root.mkdir(exist_ok=True)
        path = str(root / f"{version}.npz")
        if path not in self.refs:
            np.savez(path, **model)
            self.refs[path] = 0
        self.refs[path] += 1
        return path

    def load(self, handle):
        self.loads += 1
        with np.load(handle) as arrays:
            return dict(arrays)

    def release(self, handle):
        self.refs[handle] -= 1
        if not self.refs[handle]:
            Path(handle).unlink()
            del self.refs[handle]

    def close(self):
        for path in self.refs:
            Path(path).unlink()
        self.refs.clear()
        self.closed = True


class _DiskResults:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.paths = {}
        self.closed = False
        self.taken = []

    def put(self, assignment_id, result):
        root = Path(self.root_dir)
        root.mkdir(exist_ok=True)
        path = root / f"{assignment_id}.pkl"
        # Only this test's locally produced NumPy payloads are read back.
        path.write_bytes(pickle.dumps(result))
        self.paths[assignment_id] = path

    def take(self, assignment_id):
        path = self.paths.pop(assignment_id)
        value = pickle.loads(path.read_bytes())
        path.unlink()
        self.taken.append(assignment_id)
        return value

    def release(self, result):
        pass

    def close(self):
        for path in self.paths.values():
            path.unlink()
        self.paths.clear()
        self.closed = True


def test_offload_ports_retain_exact_bases_and_materialize_in_replay_order(tmp_path):
    snapshots = _DiskSnapshots(str(tmp_path / "bases"))
    results = _DiskResults(str(tmp_path / "results"))
    scheduler = ReplayScheduler([2, 0, 1, 3], ["a", "b", "a", "c"])
    output = _execute(
        _TraceApplication(),
        [_Client()],
        num_updates=4,
        num_active_jobs=3,
        scheduler=scheduler,
        snapshot_store=snapshots,
        result_store=results,
    )
    assert output["w"].item() == 4
    assert results.taken == [2, 0, 1, 3]
    assert snapshots.loads >= 8
    assert snapshots.closed and results.closed
    assert not list(tmp_path.rglob("*.npz")) and not list(tmp_path.rglob("*.pkl"))


class _HorizonReplay(ReplayScheduler):
    """Minimal example of a scheduler-owned simulated clock and stop horizon."""

    def __init__(self):
        super().__init__([0, 1, 2, 3], ["a", "b", "c", "d"])
        self.time = 0.0
        self.stopped = False

    def on_accept(self, assignment):
        super().on_accept(assignment)
        self.time += 1.0
        self.stopped = self.time >= 2.0

    @property
    def finished(self):
        return self.stopped


def test_scheduler_horizon_stops_before_version_budget_and_discards_partial_buffer(tmp_path):
    scheduler = _HorizonReplay()
    application = _TraceApplication()
    snapshots = _DiskSnapshots(str(tmp_path / "bases"))
    results = _DiskResults(str(tmp_path / "results"))
    output = _execute(
        application,
        [_Client()],
        num_updates=1,
        num_active_jobs=3,
        buffer_size=4,
        scheduler=scheduler,
        snapshot_store=snapshots,
        result_store=results,
    )
    assert scheduler.time == 2.0
    assert output["w"].item() == 0  # No partial publication at the time horizon.
    assert application.finished and application.closed
    assert [event["time"] for event in application.events if event["phase"] == "accept"] == [1.0, 2.0]
    assert snapshots.closed and results.closed and not snapshots.refs and not results.paths


@pytest.mark.parametrize("failure", [True, False])
def test_failure_or_bad_sequence_closes_all_extension_resources(tmp_path, failure):
    snapshots = _DiskSnapshots(str(tmp_path / "bases"))
    results = _DiskResults(str(tmp_path / "results"))
    application = _TraceApplication()
    # ID 1 cannot be reserved before acceptance of ID 0 with K=1.
    scheduler = ReplayScheduler([0, 1] if failure else [1, 0])
    with pytest.raises((ValueError, RuntimeError)):
        _execute(
            application, [_Client(failure=failure)], scheduler=scheduler, snapshot_store=snapshots, result_store=results
        )
    assert application.closed and not application.finished
    assert snapshots.closed and results.closed and not snapshots.refs and not results.paths


def test_engine_exposes_full_teacher_model_for_delta_payload_and_raw_metadata():
    seen = []
    engine = AsyncEngine(
        {"w": np.array([10], dtype=np.float32)},
        NumpyAsyncModelAdapter(),
        num_updates=1,
        num_active_jobs=1,
        update_transform=lambda update: seen.append(update) or update.delta,
    )
    job = engine.reserve(["logical-a"], [{"train_seed": 123}])[0]
    engine.accept(
        job,
        AsyncResult(
            job.assignment_id, 0, {"w": np.array([2], dtype=np.float32)}, is_delta=True, metadata={"proxy": [0.3, 0.7]}
        ),
    )
    assert seen[0].returned_model["w"].item() == 12
    assert seen[0].metadata == {"proxy": [0.3, 0.7]}
    assert seen[0].assignment.metadata == {"train_seed": 123}
    engine.close()


def test_recipe_export_reconstructs_extension_components(tmp_path, monkeypatch):
    module = _numpy_module(monkeypatch)
    recipe = AsyncRecipe(
        job_name="export_extensions",
        client=module.ArrayClient(),
        application=module.ArrayApplication(),
        model_adapter=NumpyAsyncModelAdapter(),
        num_updates=3,
        num_active_jobs=2,
        scheduler=ReplayScheduler([1, 0, 2], ["a", "b", "a"], [{"train_seed": 7}] * 3),
        train_kwargs=_Seeds(10),
        snapshot_store=MemorySnapshotStore(),
        result_store=MemoryResultStore(),
    )
    assert recipe.min_clients == 1
    recipe.export(str(tmp_path))
    config = json.loads(next(tmp_path.rglob("config_fed_server.json")).read_text())
    server = next(item for item in config["components"] if item["id"] == "_server")
    from nvflare.fuel.utils.component_builder import ComponentBuilder

    class Builder(ComponentBuilder):
        def get_module_scanner(self):
            return None

    restored = Builder().build_component(server)
    assert restored.scheduler.arrival_sequence == [1, 0, 2]
    assert restored.scheduler.logical_sequence == ["a", "b", "a"]
    assert restored.scheduler.assignment_metadata[0]["train_seed"] == 7
    assert restored.train_kwargs.offset == 10
    assert isinstance(restored.snapshot_store, MemorySnapshotStore)
    assert isinstance(restored.result_store, MemoryResultStore)


class _PhaseScheduler:
    """Exercise a logical training event that unlocks work during upload."""

    def __init__(self):
        self.time = 0.0
        self.phase = 0
        self.ready = set()
        self.available = []
        self.dispatched = []
        self.accepted = []

    def required_workers(self, num_active_jobs):
        return 1

    def initialize(self, engine, client_names):
        self.engine = engine
        self.available = list(client_names)

    def dispatch(self):
        from nvflare.collab.async_scheduling import ScheduledCall

        if not self.available:
            return []
        if not self.dispatched or (self.phase == 1 and len(self.dispatched) == 1):
            name = "first" if not self.dispatched else "second"
            assignment = self.engine.reserve([name])[0]
            self.dispatched.append((assignment.assignment_id, self.time))
            return [ScheduledCall(assignment, self.available.pop(0))]
        return []

    def on_result(self, call):
        self.ready.add(call.assignment.assignment_id)
        self.available.append(call.client_name)

    def next_result(self):
        if self.phase == 0 and 0 in self.ready:
            self.time = 1.0
            self.phase = 1  # Training completes; a client is eligible before upload.
            return None
        if self.phase == 1 and 1 in self.ready:
            self.time = 2.0
            return 1
        if self.phase == 2 and 0 in self.ready:
            self.time = 10.0
            return 0
        return None

    def on_accept(self, assignment):
        self.ready.remove(assignment.assignment_id)
        self.accepted.append(assignment.assignment_id)
        self.phase += 1

    @property
    def finished(self):
        return False

    def close(self):
        self.ready.clear()
        self.available.clear()


def test_phase_event_can_unlock_dispatch_while_all_earlier_uploads_are_held():
    scheduler = _PhaseScheduler()
    application = _TraceApplication()
    output = _execute(application, [_Client()], num_active_jobs=2, scheduler=scheduler)
    assert output["w"].item() == 2
    assert scheduler.dispatched == [(0, 0.0), (1, 1.0)]
    assert scheduler.accepted == [1, 0]
    accepted = [event for event in application.events if event["phase"] == "accept"]
    assert [event["time"] for event in accepted] == [2.0, 10.0]
    assert [event["base_version"] for event in accepted] == [0, 0]


class _TransformResources:
    def __init__(self, fail=False):
        self.fail = fail
        self.initialized = False
        self.closed = False

    def initialize(self):
        self.initialized = True
        if self.fail:
            raise RuntimeError("initialization failed")

    def __call__(self, update):
        assert self.initialized
        return update.delta

    def close(self):
        self.closed = True


@pytest.mark.parametrize("fail", [False, True])
def test_transform_resources_close_after_success_or_partial_initialization(fail):
    transform = _TransformResources(fail)
    application = _TraceApplication()
    if fail:
        with pytest.raises(RuntimeError, match="initialization failed"):
            _execute(application, [_Client()], update_transform=transform)
    else:
        _execute(application, [_Client()], update_transform=transform)
    assert transform.initialized and transform.closed and application.closed
