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

"""Exercise actual Collab groups and recipe export without downloading datasets."""

import importlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from examples.advanced.collab.numpy_async.job import NumpyModelOps
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.simulation_plan import SIMULATION_PLAN_PROP, SimulationPlanRuntime
from nvflare.collab import FedBuffRecipe, collab
from nvflare.collab.api.app import ClientApp, ServerApp
from nvflare.collab.api.constants import FL_CONTEXT_PROP
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.group_call_context import ResultWaiter
from nvflare.collab.api.proxy import Proxy
from nvflare.collab.fedbuff_recipe import FedBuffResult, FedBuffTrainingCoordinator, FedBuffWorkflow

from ._direct_dispatcher import _DirectDispatcher


class _ModelLoader:
    def load_model(self):
        return {"w": np.zeros(1, dtype=np.float32)}


class _Observer:
    def __init__(self, release=None):
        self.release = release
        self.versions = []
        self.closed = False
        self.finished = False

    def on_model(self, model, version):
        self.versions.append((version, model["w"].copy()))
        if version == 1 and self.release:
            self.release.set()

    def on_finish(self, model, version):
        self.finished = True

    def close(self):
        self.closed = True
        if self.release:
            self.release.set()


class _Client:
    def __init__(self, release=None, failure=False, malformed=False, increment=1):
        self.release = release
        self.failure = failure
        self.malformed = malformed
        self.increment = increment
        self.assignments = []
        self.seeds = []

    @collab.publish
    def train(self, assignment_ids, model_version, global_model, logical_assignments, train_seeds=None):
        if self.release:
            assert self.release.wait(timeout=2), "workflow waited for slow client before publishing"
        if self.failure:
            raise RuntimeError("training failed")
        name = collab.site_name
        self.assignments.append((assignment_ids[name], model_version))
        if train_seeds is not None:
            self.seeds.append((assignment_ids[name], train_seeds[name]))
        metadata = {
            "assignment_id": assignment_ids[name],
            "model_version": model_version,
            "logical_name": logical_assignments[name],
        }
        if self.malformed:
            metadata["assignment_id"] = -1
        return {"w": global_model["w"] + self.increment}, {"num_steps": 2}, metadata


class _ScaleUpdate:
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, update):
        return {name: value * self.scale for name, value in update.delta.items()}


def _execute(observer, clients, *, abort=None, runtime=None, **options):
    signal = abort or Signal()
    settings = dict(num_updates=2, concurrency=len(clients), call_timeout=0.5)
    settings.update(options)
    workflow = FedBuffWorkflow(_ModelLoader(), NumpyModelOps(), observer=observer, **settings)
    server = ServerApp(workflow)
    server._abort_signal = signal
    if runtime is not None:
        fl_ctx = FLContext()
        fl_ctx.set_prop(SIMULATION_PLAN_PROP, runtime)
        server.set_prop(FL_CONTEXT_PROP, fl_ctx)
    previous = get_call_context()
    try:
        with ThreadPoolExecutor(max_workers=len(clients)) as workers:
            proxies = []
            for index, client in enumerate(clients):
                name = f"site-{index + 1}"
                app = ClientApp(client)
                app.name = name
                app._abort_signal = signal
                app.initialize(app.new_context(name, name, set_call_ctx=False))
                dispatcher = _DirectDispatcher("client", app, client, signal, workers)
                proxies.append(Proxy(server, name, name, dispatcher, app.get_collab_interface()["client"]))
            server._client_proxies = proxies
            server.new_context("server", "server")
            return workflow.execute()
    finally:
        set_call_context(previous)


def test_real_nonblocking_groups_publish_while_slow_assignment_is_active():
    release = threading.Event()
    app = _Observer(release)
    fast, slow = _Client(), _Client(release)
    output = _execute(app, [fast, slow])
    assert [v for v, _ in app.versions] == [0, 1, 2]
    assert output["w"].item() == 2
    assert app.closed and app.finished
    assert len(fast.assignments) + len(slow.assignments) == 2


def test_callback_receipt_order_survives_delayed_per_call_completion(monkeypatch):
    delivered_a, release_a = threading.Event(), threading.Event()
    original = ResultWaiter.set_result

    def delay_completion(waiter, target_name, result):
        if target_name == "site-1":
            delivered_a.set()
            assert release_a.wait(timeout=2), "first reply was held behind per-call completion"
        return original(waiter, target_name, result)

    monkeypatch.setattr(ResultWaiter, "set_result", delay_completion)
    accepted = []

    def transform(update):
        accepted.append(update.assignment.client_name)
        return {name: value / (1 + update.staleness) for name, value in update.delta.items()}

    output = _execute(
        _Observer(release_a),
        [_Client(increment=2), _Client(release=delivered_a, increment=4)],
        update_transform=transform,
    )
    assert accepted == ["site-1", "site-2"]
    assert output["w"].item() == 4  # 2 + 4/2; reversing replies would produce 5.


def test_real_groups_sync_boundary():
    app = _Observer()
    clients = [_Client(), _Client()]
    output = _execute(app, clients, buffer_size=2, min_open_slots=2)
    assert output["w"].item() == 2
    assert all([version for _, version in client.assignments] == [0, 1] for client in clients)
    assert app.closed and app.finished


@pytest.mark.parametrize("physical_workers", [1, 2])
def test_planned_replay_binds_bases_before_physical_dispatch(tmp_path, physical_workers):
    plan = {
        "format_version": 1,
        "async_execution": {
            "logical_sequence": ["client-0", "client-1", "client-2", "client-3"],
            "arrival_sequence": [1, 0, 2, 3],
        },
    }
    runtime = SimulationPlanRuntime(plan, tmp_path / str(physical_workers))
    clients = [_Client() for _ in range(physical_workers)]
    output = _execute(
        _Observer(),
        clients,
        runtime=runtime,
        num_updates=4,
        concurrency=2,
        buffer_size=1,
    )
    runtime.assert_consumed()
    assignments = [json.loads(line) for line in (runtime.trace_dir / "assignments.jsonl").read_text().splitlines()]
    accepted = [json.loads(line) for line in (runtime.trace_dir / "accepted.jsonl").read_text().splitlines()]
    assert [item["base_version"] for item in assignments] == [0, 0, 1, 2]
    assert [item["assignment_id"] for item in accepted] == [1, 0, 2, 3]
    assert output["w"].item() == 4


def test_planned_replay_respects_explicit_physical_mapping(tmp_path):
    plan = {
        "format_version": 1,
        "async_execution": {
            "logical_sequence": ["a", "b"],
            "arrival_sequence": [1, 0],
            "physical_sites": ["site-2", "site-1"],
        },
    }
    runtime = SimulationPlanRuntime(plan, tmp_path)
    _execute(_Observer(), [_Client(), _Client()], runtime=runtime, num_updates=2, concurrency=2, buffer_size=1)
    dispatched = [json.loads(line) for line in (tmp_path / "dispatch.jsonl").read_text().splitlines()]
    assert dispatched == [{"assignment_id": 1, "site": "site-1"}, {"assignment_id": 0, "site": "site-2"}]


@pytest.mark.parametrize("physical_workers", [1, 2])
def test_timed_plan_uses_virtual_events_independent_of_worker_pool(tmp_path, physical_workers):
    profile = {
        "download": {"distribution": "constant", "value": 0},
        "train": {"distribution": "constant", "value": 0},
        "upload": {"distribution": "constant", "value": 1},
    }
    plan = {
        "format_version": 1,
        "async_execution": {
            "logical_sequence": ["a", "b", "c", "d"],
            "train_seeds": [11, 12, 13, 14],
            "timing": {
                "starts": [0, 0, 2.1, 3.2],
                "seed": 7,
                "profiles": {name: profile for name in ("a", "b", "c", "d")},
                "horizon": 10,
            },
        },
    }
    runtime = SimulationPlanRuntime(plan, tmp_path / str(physical_workers))
    clients = [_Client() for _ in range(physical_workers)]
    output = _execute(
        _Observer(),
        clients,
        runtime=runtime,
        num_updates=4,
        concurrency=2,
        buffer_size=1,
    )
    runtime.assert_consumed()
    assignments = [json.loads(line) for line in (runtime.trace_dir / "assignments.jsonl").read_text().splitlines()]
    accepted = [json.loads(line) for line in (runtime.trace_dir / "accepted.jsonl").read_text().splitlines()]
    assert [item["base_version"] for item in assignments] == [0, 0, 2, 3]
    assert [item["train_seed"] for item in assignments] == [11, 12, 13, 14]
    assert [item["assignment_id"] for item in accepted] == [0, 1, 2, 3]
    assert [item["virtual_time"] for item in accepted] == [1.0, 1.0, 3.1, 4.2]
    assert sorted(seed for client in clients for seed in client.seeds) == [(0, 11), (1, 12), (2, 13), (3, 14)]
    assert output["w"].item() == 4


def test_timed_plan_stops_acceptance_at_horizon_and_drains_started_results(tmp_path):
    profile = {phase: {"distribution": "constant", "value": 1} for phase in ("download", "train", "upload")}
    runtime = SimulationPlanRuntime(
        {
            "format_version": 1,
            "async_execution": {
                "logical_sequence": ["a", "b"],
                "timing": {"starts": [0, 0], "seed": 1, "profiles": {"a": profile, "b": profile}, "horizon": 0.5},
            },
        },
        tmp_path,
    )
    output = _execute(_Observer(), [_Client()], runtime=runtime, num_updates=2, concurrency=2, buffer_size=1)
    assert output["w"].item() == 0
    assert runtime.virtual_time == 0.5
    assert len(list((tmp_path / "results").glob("*.pkl"))) == 1
    assert not (tmp_path / "accepted.jsonl").exists()


def test_optional_transform_runs_through_collab_recipe_workflow():
    output = _execute(_Observer(), [_Client(), _Client()], update_transform=_ScaleUpdate())
    assert output["w"].item() == 1


@pytest.mark.parametrize("failure,malformed", [(True, False), (False, True)])
def test_group_errors_close_observer_and_never_publish(failure, malformed):
    app = _Observer()
    with pytest.raises((RuntimeError, ValueError)):
        _execute(app, [_Client(failure=failure, malformed=malformed)])
    assert app.closed and not app.finished
    assert len(app.versions) == 1


def test_abort_stops_admission_and_closes_observer():
    signal = Signal()
    signal.trigger(True)
    app = _Observer()
    client = _Client()
    with pytest.raises(RuntimeError, match="aborted"):
        _execute(app, [client], abort=signal)
    assert app.closed and client.assignments == []


def _numpy_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "advanced"))
    return importlib.import_module("collab.numpy_async.job")


def test_recipe_export_round_trip_loads_model_components(tmp_path, monkeypatch):
    module = _numpy_module(monkeypatch)
    recipe = module.make_recipe(num_clients=2, num_updates=3, buffer_size=2)
    recipe.server.update_transform = _ScaleUpdate(0.25)
    recipe.export(str(tmp_path))
    configs = list(tmp_path.rglob("config_fed_server.json"))
    assert len(configs) == 1
    config = json.loads(configs[0].read_text())
    server = next(item for item in config["components"] if item["id"] == "_server")
    assert server["path"] == "nvflare.collab.fedbuff_recipe.FedBuffWorkflow"
    assert server["args"]["model_loader"]["path"].endswith("ArrayModelLoader")
    assert server["args"]["observer"]["path"].endswith("ArrayObserver")
    assert server["args"]["num_updates"] == 3
    assert server["args"]["model_ops"]["path"].endswith("NumpyModelOps")
    from nvflare.fuel.utils.component_builder import ComponentBuilder

    class Builder(ComponentBuilder):
        def get_module_scanner(self):
            return None

    builder = Builder()
    restored = builder.build_component(server)
    np.testing.assert_array_equal(restored.model_loader.load_model()["weights"], np.zeros(3))
    assert restored.observer is not None
    assert restored.buffer_size == 2
    assert isinstance(restored.update_transform, _ScaleUpdate)
    assert restored.update_transform.scale == 0.25
    assert recipe.finalize() is recipe.finalize()


def test_async_imports_without_tensor_framework_or_synthesis():
    code = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'torchvision', 'kornia'):
            raise ImportError('optional dependency blocked')
sys.meta_path.insert(0, Block())
from nvflare.collab import FedBuffRecipe
from nvflare.collab.fedbuff_recipe import FedBuffResult, FedBuffTrainingCoordinator
from examples.advanced.collab.numpy_async.job import NumpyModelOps
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cifar_recipe_uses_library_scheduler_and_step_weights(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("tensorboard")
    folder = Path(__file__).resolve().parents[3] / "examples/advanced/collab/pt_async_cifar10"
    # Example scripts use directory-local imports. Keep test module caches isolated.
    names = ("job", "async_aggregator", "model", "trainer", "cifar10_data")
    saved = {name: sys.modules.pop(name, None) for name in names}
    monkeypatch.syspath_prepend(str(folder))
    try:
        job = importlib.import_module("job")
        args = job.define_parser().parse_args([])
        args.device = "cpu"
        args.data_root = "/tmp/cifar10"
        args.train_idx_root = "/tmp/prepared-cifar"
        recipe = job.make_recipe(args)
        assert isinstance(recipe, FedBuffRecipe)
        assert recipe.server.weight_metric == "num_steps"
        assert recipe.server.concurrency == 8
        assert recipe.server.buffer_size == 4
        assert recipe.server.min_open_slots == 1
        assert recipe.client._optimizer is None
        assert recipe.server.observer._eval_model is None
        recipe.finalize()
        recipe.export(str(tmp_path))
        assert list(tmp_path.rglob("async_aggregator.py"))
        assert list(tmp_path.rglob("trainer.py"))
        assert list(tmp_path.rglob("model.py"))
        server_config = json.loads(next(tmp_path.rglob("config_fed_server.json")).read_text())
        client_config = json.loads(next(tmp_path.rglob("config_fed_client.json")).read_text())
        server = next(item for item in server_config["components"] if item["id"] == "_server")
        client = next(item for item in client_config["components"] if item["id"] == "_client")
        assert server["args"]["observer"]["args"]["device"] == "cpu"
        assert client["args"]["device"] == "cpu"
    finally:
        for name, previous in saved.items():
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous


def test_cifar_model_ops_own_cpu_snapshots_and_preserve_nonfloating_buffers():
    torch = pytest.importorskip("torch")
    from examples.advanced.collab.pt_async_cifar10.model import PTModelOps

    initial = {"w": torch.tensor([10.0], requires_grad=True), "counter": torch.tensor(5)}
    coordinator = FedBuffTrainingCoordinator(initial, PTModelOps(), 1, 2, buffer_size=2, weight_metric="num_steps")
    initial["w"].detach().fill_(99)
    a, b = coordinator.dispatch(["a", "b"])
    first = {"w": torch.tensor([12.0]), "counter": torch.tensor(99)}
    second = {"w": torch.tensor([16.0]), "counter": torch.tensor(100)}
    assert not coordinator.accept(a, FedBuffResult(a.assignment_id, 0, first, {"num_steps": 1}))
    first["w"].fill_(999)
    assert coordinator.accept(b, FedBuffResult(b.assignment_id, 0, second, {"num_steps": 3}))
    assert coordinator.model["w"].item() == 15
    assert coordinator.model["counter"].item() == 5
    assert not coordinator.model["w"].requires_grad
    assert coordinator.model["w"].device.type == "cpu"


def test_cifar_training_keeps_optimizer_scheduler_and_num_steps(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    folder = Path(__file__).resolve().parents[3] / "examples/advanced/collab/pt_async_cifar10"
    saved = {name: sys.modules.pop(name, None) for name in ("model", "trainer")}
    monkeypatch.syspath_prepend(str(folder))
    previous = get_call_context()
    try:
        trainer_module = importlib.import_module("trainer")
        dataset = torch.utils.data.TensorDataset(torch.zeros(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))
        monkeypatch.setattr(trainer_module.datasets, "CIFAR10", lambda **kwargs: dataset)
        np.save(tmp_path / "site-1.npy", np.arange(4))
        trainer = trainer_module.Cifar10Trainer(
            data_root=str(tmp_path),
            train_idx_root=str(tmp_path),
            device="cpu",
            local_batch_size=2,
            aggregation_epochs=1,
            total_client_rounds=1,
        )
        app = ClientApp(trainer)
        app.name = "site-1"
        app._abort_signal = Signal()
        app.new_context("server", "site-1")
        trainer.initialize()
        optimizer = trainer._optimizer
        initial = trainer_module.get_model_params(trainer._local_model, target_device="cpu")
        first, metrics, metadata = trainer.train({"site-1": 0}, 0, initial, {"site-1": "site-1"})
        assert metrics["num_steps"] == 2
        assert metadata == {"assignment_id": 0, "model_version": 0, "logical_name": "site-1"}
        epoch = trainer._scheduler.last_epoch
        trainer.train({"site-1": 1}, 1, first, {"site-1": "site-1"})
        assert trainer._optimizer is optimizer and optimizer.state
        assert trainer._scheduler.last_epoch == epoch == trainer._scheduler.T_max
        assert all(not value.requires_grad and value.device.type == "cpu" for value in first.values())
    finally:
        set_call_context(previous)
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


def test_call_timeout_closes_observer_without_retrying():
    release = threading.Event()
    app = _Observer(release)
    with pytest.raises(RuntimeError, match="timed out"):
        _execute(app, [_Client(release)])
    assert app.closed and not app.finished
    assert len(app.versions) == 1
