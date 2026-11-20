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
from unittest.mock import Mock

import pytest


@pytest.fixture
def research(monkeypatch):
    folder = Path(__file__).resolve().parents[3] / "research" / "fedrevive"
    monkeypatch.syspath_prepend(str(folder))
    names = (
        "client",
        "data",
        "dfkd",
        "fedrevive",
        "job",
        "model",
        "schedule_standard",
        "schedule_shifted",
        "server",
        "update",
    )
    saved = {name: sys.modules.pop(name, None) for name in names}
    try:
        yield lambda name: importlib.import_module(name)
    finally:
        for name, previous in saved.items():
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous


@pytest.mark.parametrize(
    ("label", "eligible", "distinct"),
    [
        pytest.param(0, range(6), True, id="enough-images"),
        pytest.param(1, range(6, 8), False, id="scarce-class"),
        pytest.param(2, range(8), False, id="missing-class"),
    ],
)
def test_synthetic_pool_plans_class_aware_kd_batches(research, label, eligible, distinct):
    torch = pytest.importorskip("torch")
    pool = research("dfkd").SyntheticPool(max_images=8)
    images = torch.arange(8).reshape(8, 1)
    pool.add(images, torch.tensor([0, 0, 0, 0, 0, 0, 1, 1]))
    distribution = torch.eye(3)[label]

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(41)
        batches = pool.sample_batch_indices(distribution, batch_size=3, num_batches=2)
        torch.random.default_generator.manual_seed(41)
        assert batches == pool.sample_batch_indices(distribution, batch_size=3, num_batches=2)

    assert len(batches) == 2 and all(len(batch) == 3 for batch in batches)
    indices = [index for batch in batches for index in batch]
    assert set(indices).issubset(eligible)
    if distinct:
        assert len(set(indices)) == len(indices)
    for batch in batches:
        torch.testing.assert_close(pool.load_batch(batch), images[batch])


@pytest.mark.parametrize(
    ("version", "base_version", "student_delta", "processed_delta"),
    [(49, 0, None, 2.0), (75, 0, None, 1.0), (125, 50, 8.0, 5.0)],
    ids=("frozen", "synthesis-only", "distilled"),
)
def test_revival_uses_current_global_delta_and_refreshes_arriving_teacher(
    research, tmp_path, version, base_version, student_delta, processed_delta
):
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
        return (None if student_delta is None else {"w": torch.tensor([student_delta])}), {}

    policy.reviver = SimpleNamespace(revive=revive)
    policy.teachers.add({"w": torch.tensor([1.0])}, torch.tensor([1.0, 0.0]), "client-0", 0)
    assignment = Assignment(7, "site-1", "site-1", base_version, {"w": torch.tensor([0.0])})
    update = AcceptedUpdate(
        assignment,
        version,
        {"w": torch.tensor([20.0])},
        {"w": torch.tensor([2.0])},
        1.0,
        {"shard_name": "client-1"},
        {"w": torch.tensor([2.0])},
        {},
    )
    result = policy(update)
    assert seen == ([(20.0, [2.0, 1.0], version)] if version >= 50 else [])
    torch.testing.assert_close(result["w"], torch.tensor([processed_delta]))  # beta(75)=0.5 when eligible
    assert update.current_model["w"].item() == 20 and update.delta["w"].item() == 2
    # An on-time upload still enters history but does not trigger revival.
    current = Assignment(8, "site-1", "client-0", version, update.current_model)
    result = policy(
        AcceptedUpdate(current, version, update.current_model, update.delta, 1.0, {}, update.returned_model, {})
    )
    torch.testing.assert_close(result["w"], update.delta["w"])
    assert len(seen) == int(version >= 50) and len(policy.teachers) == 2


@pytest.mark.parametrize("module_name", ["schedule_standard", "schedule_shifted"])
def test_paper_plan_preserves_logical_release_ties_and_horizon(research, module_name):
    profiles = {
        f"client-{i}": {"download_mean": 0, "train_mean": 0, "upload_mean": delay, "upload_half_width": 0}
        for i, delay in enumerate((5, 1))
    }
    plan = research(module_name).make_paper_plan(
        profiles, run_seed=10, max_time=2, num_rounds=3, concurrency=2, min_open_slots=1, buffer_size=1
    )["async_execution"]
    assert plan["logical_sequence"] == ["client-0", "client-1", "client-0"]
    assert plan["base_versions"] == [0, 0, 1]
    assert plan["train_seeds"] == [10_000_001, 10_000_002, 10_000_003]
    events = plan["timing"]["events"]
    uploads = [event for event in events if event["phase"] == "upload"]
    assert uploads == [
        {"assignment_id": 1, "phase": "upload", "virtual_time": 1},
        {"assignment_id": 0, "phase": "upload", "virtual_time": 5},
    ]
    refill = {"assignment_id": 2, "phase": "start", "virtual_time": 1}
    assert events.index(uploads[0]) < events.index(refill) < events.index(uploads[1])


@pytest.mark.parametrize("module_name", ["schedule_standard", "schedule_shifted"])
def test_custom_paper_delays_use_independent_assignment_and_phase_rngs(research, module_name):
    planner = research(module_name)
    profiles = {"client-0": {"download_mean": 0.1, "train_mean": 1, "upload_mean": 0.2}}
    options = dict(run_seed=10, max_time=200, num_rounds=2, concurrency=1, min_open_slots=1, buffer_size=1)
    lognormal = planner.lognormal_training_sampler(0.4)

    def draw_extra_download_samples(assignment_id, logical_name, phase, profile, rng):
        if phase == "download":
            rng.random(20)
        return lognormal(assignment_id, logical_name, phase, profile, rng)

    def events(sampler):
        plan = planner.make_paper_plan(profiles, **options, delay_sampler=sampler)
        return plan["async_execution"]["timing"]["events"]

    custom_events = events(lognormal)
    assert custom_events == events(draw_extra_download_samples)
    assert custom_events != events(None)
    with pytest.raises(
        ValueError, match=r"download delay for assignment 0 \(client-0\) must be finite and nonnegative"
    ):
        events(lambda *args: float("nan"))


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


def test_evaluation_preserves_training_rng(research, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("tensorboard")
    from nvflare.apis.simulation_plan import ASYNC_EXECUTION, SIMULATION_PLAN_PROP

    server = research("server")
    dataset = torch.utils.data.TensorDataset(torch.zeros(2, 2), torch.tensor([0, 1]))
    monkeypatch.setattr(server.datasets, "CIFAR10", lambda **kwargs: dataset)
    monkeypatch.setattr(server, "create_model", lambda: torch.nn.Linear(2, 2))
    monkeypatch.setattr(server, "SummaryWriter", Mock)
    monkeypatch.setattr(
        server,
        "collab",
        SimpleNamespace(
            workspace=SimpleNamespace(get_run_dir=lambda _: str(tmp_path)),
            fl_ctx=SimpleNamespace(
                get_job_id=lambda: "job",
                get_prop={SIMULATION_PLAN_PROP: SimpleNamespace(plan={ASYNC_EXECUTION: {"timing": {}}})}.get,
            ),
        ),
    )
    observer = server.FedReviveObserver("unused", {"arrival_mode": "simulated"}, device="cpu", eval_interval=1)
    observer.initialize()
    assert observer.simulated_time == 0.0

    with torch.random.fork_rng(devices=[]):
        model = server.create_model().state_dict()
        training_rng = torch.random.get_rng_state()
        for version in (0, 1):
            observer.on_model(model, version)
            assert torch.equal(torch.random.get_rng_state(), training_rng)
    assert [record["model_version"] for record in observer.history] == [0, 1]


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
    profile = {"download_mean": 1, "train_mean": 1, "upload_mean": 1}
    paper_args.max_model_versions = 1
    paper = job.make_sim_env(paper_args, {"runtime_profiles": {"client-0": profile}})
    assert paper.simulation_plan["async_execution"]["timing"]["events"][0]["phase"] == "start"
    actual = job.make_sim_env(actual_args, {})
    fed_job = SimpleNamespace(name="fedrevive_fedrevive")
    workspace = tmp_path / fed_job.name
    reservation = workspace.with_name(f".{workspace.name}.running")
    trace = workspace / "phases.jsonl"
    deployments = []

    def deploy(env, submitted_job):
        assert env is paper, "a colliding launch reached the simulator"
        deployments.append(env)
        # Another launch must be rejected even before the simulator creates its output directory.
        assert not workspace.exists()
        with pytest.raises(FileExistsError, match="is reserved") as error:
            actual.deploy(submitted_job)
        assert str(reservation) in str(error.value)
        workspace.mkdir()
        trace.write_text("paper run\n")
        return submitted_job.name

    monkeypatch.setattr(SimEnv, "deploy", deploy)
    assert paper.deploy(fed_job) == fed_job.name
    with pytest.raises(FileExistsError, match="already exists"):
        actual.deploy(fed_job)
    assert deployments == [paper]
    assert trace.read_text() == "paper run\n"
