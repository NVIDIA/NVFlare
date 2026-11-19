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

"""Focused research contracts for revival, phase eligibility and FedAvg bases."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def research(monkeypatch):
    folder = Path(__file__).resolve().parents[3] / "research" / "fedrevive"
    monkeypatch.syspath_prepend(str(folder))
    names = ("client", "data", "fedrevive", "job", "model", "schedule", "server", "storage", "update", "workflow")
    saved = {name: sys.modules.pop(name, None) for name in names}
    try:
        yield lambda name: importlib.import_module(name)
    finally:
        for name, previous in saved.items():
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous


def test_revival_uses_current_global_delta_and_refreshes_arriving_teacher(research, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from nvflare.collab.fedbuff_recipe import AcceptedUpdate, Assignment

    policy_module = research("update")
    policy = policy_module.FedReviveUpdate("unused")
    policy.run_dir = tmp_path
    policy.config = SimpleNamespace(freeze_versions=50)
    policy.teachers = research("fedrevive").TeacherBuffer(2)
    policy.estimator = SimpleNamespace(estimate=lambda name, model: torch.tensor([0.25, 0.75]))
    seen = []

    def revive(global_model, teachers, version):
        seen.append((global_model["w"].item(), [t.model["w"].item() for t in teachers], version))
        assert all(t.client_name == "client-0" for t in teachers)
        assert all(torch.equal(t.class_proxy, torch.tensor([0.25, 0.75])) for t in teachers)
        return {"w": torch.tensor([8.0])}, {}

    policy.reviver = SimpleNamespace(revive=revive)
    policy.teachers.add({"w": torch.tensor([1.0])}, torch.tensor([1.0, 0.0]), "client-0", 0)
    assignment = Assignment(7, "site-1", "site-1", 0, {"w": torch.tensor([0.0])})
    update = AcceptedUpdate(
        assignment,
        75,
        {"w": torch.tensor([20.0])},
        {"w": torch.tensor([2.0])},
        1.0,
        {"shard_name": "client-1"},
        {"w": torch.tensor([2.0])},
        {},
    )
    result = policy(update)
    assert seen == [(20.0, [2.0, 1.0], 75)]
    torch.testing.assert_close(result["w"], torch.tensor([5.0]))  # beta(75)=0.5
    assert update.current_model["w"].item() == 20 and update.delta["w"].item() == 2
    # An on-time upload still enters history but does not trigger revival.
    current = Assignment(8, "site-1", "client-0", 75, update.current_model)
    policy(AcceptedUpdate(current, 75, update.current_model, update.delta, 1.0, {}, update.returned_model, {}))
    assert len(seen) == 1 and len(policy.teachers) == 2


def test_phase_release_precedes_upload_and_horizon_rejects_due_result(research, tmp_path):
    schedule = research("schedule").PhaseSchedule(
        {"client-0": {"download_mean": 0.1, "train_mean": 1.0, "upload_mean": 0.2}},
        run_seed=10,
        max_time=0.1 + np.random.RandomState(10).exponential(1.0) + 0.1,
    )

    class Coordinator:
        run_dir = tmp_path
        active = {}
        finished = False
        available_slots = 1

        def reserve_for_phase(self, names, run_seed):
            assignment = SimpleNamespace(assignment_id=0, logical_name=names[0])
            self.active[0] = assignment
            return [assignment]

    schedule.initialize(Coordinator(), ["site-1"])
    calls = schedule.dispatch()
    assert len(calls) == 1 and schedule.logical == []
    assert schedule.next_result() is None  # download/train occur before physical completion
    assert schedule.logical == ["client-0"] and not schedule.stopped
    schedule.on_result("site-1", 0)
    assert schedule.next_result() is None and schedule.stopped  # upload exceeds horizon
    phases = (tmp_path / "phases.jsonl").read_text()
    assert all(f'"phase": "{name}"' in phases for name in ("download", "train", "upload"))
    schedule.close()


def test_fedavg_rejects_mixed_assignment_bases_within_a_publication(research):
    pytest.importorskip("torch")
    guard = research("update").FedAvgUpdate(2)
    delta = {"w": 1.0}

    def update(version):
        return SimpleNamespace(assignment=SimpleNamespace(base_version=version), delta=delta)

    assert guard(update(0)) is delta
    assert guard(update(0)) is delta
    guard(update(1))
    with pytest.raises(RuntimeError, match="different global versions"):
        guard(update(2))


def test_replay_plan_is_supplied_to_simenv_and_excluded_from_fedrevive_export(research, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("tensorboard")
    from nvflare.collab import FedBuffRecipe

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    job = research("job")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "async_execution": {
                    "logical_sequence": ["client-0", "client-1"],
                    "arrival_sequence": [1, 0],
                    "train_seeds": [101, 102],
                },
            }
        )
    )
    args = job.define_parser().parse_args(
        [
            "--arrival-mode",
            "replay",
            "--simulation-plan",
            str(plan_path),
            "--num-rounds",
            "2",
            "--concurrency",
            "2",
            "--num-clients",
            "1",
        ]
    )
    manifest = {"runtime_profiles": {"client-0": {}, "client-1": {}}}
    env = job.make_sim_env(args, manifest)
    assert env.num_clients == 1
    assert env.simulation_plan["async_execution"]["arrival_sequence"] == [1, 0]
    with pytest.raises(ValueError, match="unprepared logical clients.*client-1"):
        job.make_sim_env(args, {"runtime_profiles": {"client-0": {}}})

    recipe, _ = job.make_recipe(args)
    assert type(recipe) is FedBuffRecipe
    recipe.export(str(tmp_path / "job"))
    config = next((tmp_path / "job").rglob("config_fed_server.json")).read_text()
    workflow = next(item for item in json.loads(config)["components"] if item["id"] == "_server")
    assert workflow["path"] == "nvflare.collab.fedbuff_recipe.FedBuffWorkflow"
    assert workflow["args"]["num_rounds"] == 2
    assert str(plan_path) not in config and "simulation_plan" not in config


def test_replay_observer_records_horizon_when_no_model_was_published(research, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("tensorboard")
    from nvflare.apis.simulation_plan import SIMULATION_PLAN_PROP

    server = research("server")
    runtime = SimpleNamespace(virtual_time=5.0)
    monkeypatch.setattr(
        server,
        "collab",
        SimpleNamespace(fl_ctx=SimpleNamespace(get_prop={SIMULATION_PLAN_PROP: runtime}.get)),
    )
    observer = server.FedReviveObserver("unused", {"arrival_mode": "replay"})
    observer.run_dir = tmp_path
    observer.history = [{"model_version": 0, "simulated_time": None}]
    observer.simulated_time = None
    model = {"w": torch.tensor([1.0])}

    observer.on_finish(model, 0)

    result = json.loads((tmp_path / "results.json").read_text())
    assert result["model_version"] == 0 and result["simulated_time"] == 5.0
    assert result["history"] == [{"model_version": 0, "simulated_time": None, "final": True}]
    saved = torch.load(tmp_path / "final_model" / "global_model.pt", weights_only=True)
    torch.testing.assert_close(saved, model)


def test_simulator_preserves_workspace_across_arrival_modes(research, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("tensorboard")
    from nvflare.recipe import SimEnv
    from nvflare.recipe.sim_env import SIMULATOR_WORKSPACE_ROOT_ENV_VAR

    monkeypatch.setenv(SIMULATOR_WORKSPACE_ROOT_ENV_VAR, str(tmp_path))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    job = research("job")
    parser = job.define_parser()
    paper_args = parser.parse_args(["--workspace-root", str(tmp_path)])
    actual_args = parser.parse_args(["--arrival-mode", "actual", "--workspace-root", str(tmp_path)])
    paper = job.make_sim_env(paper_args, {})
    actual = job.make_sim_env(actual_args, {})
    fed_job = SimpleNamespace(name="fedrevive_fedrevive")
    workspace = tmp_path / fed_job.name
    trace = workspace / "phases.jsonl"
    deployments = []

    def deploy(env, submitted_job):
        assert env is paper, "a colliding launch reached the simulator"
        deployments.append(env)
        # Another launch must be rejected even before the simulator creates its output directory.
        assert not workspace.exists()
        with pytest.raises(FileExistsError, match="is reserved"):
            actual.deploy(submitted_job)
        workspace.mkdir()
        trace.write_text("paper run\n")
        return submitted_job.name

    monkeypatch.setattr(SimEnv, "deploy", deploy)
    assert paper.deploy(fed_job) == fed_job.name
    with pytest.raises(FileExistsError, match="already exists"):
        actual.deploy(fed_job)
    assert deployments == [paper]
    assert trace.read_text() == "paper run\n"
