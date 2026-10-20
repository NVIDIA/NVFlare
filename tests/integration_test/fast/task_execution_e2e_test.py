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


import importlib
import json
import os
import socket
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
import yaml

from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity
from nvflare.recipe import SimEnv
from tests.hello_pt_test_utils import REPO_ROOT, load_hello_pt_module

ADVANCED_DIR = REPO_ROOT / "examples" / "advanced" / "hello-pt-environments"


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "environment,lifetime",
    [
        ("sim", "job"),
        ("sim", "task"),
        pytest.param(
            "poc",
            "job",
            marks=pytest.mark.xfail(
                condition=sys.platform == "darwin" and sys.version_info[:2] == (3, 14),
                strict=True,
                reason="Published baseline job-lifetime PocEnv exits RC 101 during shutdown on macOS/Python 3.14",
            ),
        ),
        ("poc", "task"),
    ],
)
def test_native_hello_pt_training_and_final_evaluation(tmp_path, monkeypatch, lifetime, environment):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.setenv("FL_LOG_LEVEL", "progress")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.chdir(ADVANCED_DIR)
    poc_module = importlib.import_module("nvflare.recipe.poc_env")
    monkeypatch.setattr(poc_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as example:
        args = example.parse_args(
            ["--env", environment, "--execution_lifetime", lifetime, "--num_rounds", "2", "--torch_threads", "1"]
        )
        recipe = example.create_recipe(args)
        env = (
            SimEnv(num_clients=2, workspace_root=str(tmp_path / "simulation"))
            if environment == "sim"
            else example.create_environment(args)
        )
        if environment == "poc":
            project = yaml.safe_load((REPO_ROOT / "tests/integration_test/data/projects/client_api.yml").read_text())
            # Reserve both ports together so xdist workers cannot reuse 8002/8003.
            with ExitStack() as stack:
                sockets = [stack.enter_context(socket.socket()) for _ in range(2)]
                for sock in sockets:
                    sock.bind(("127.0.0.1", 0))
                server = next(p for p in project["participants"] if p["type"] == "server")
                server["fed_learn_port"], server["admin_port"] = [s.getsockname()[1] for s in sockets]
            next(p for p in project["participants"] if p["type"] == "admin")["name"] = env.username
            project_path = tmp_path / "project.yml"
            project_path.write_text(yaml.safe_dump(project))
            env.project_conf_path = str(project_path)
        try:
            run = recipe.execute(env)
            result = run.get_result(timeout=240, clean_up=False)
            assert result is not None
            assert run.get_status() in example.SUCCESS_STATUSES
        finally:
            if environment == "poc":
                env.stop(clean_up=False)
    root = Path(result)
    model_files = list(root.rglob("FL_global_model.pt"))
    assert len(model_files) == 1
    assert isinstance(torch.load(model_files[0], map_location="cpu", weights_only=True)["model"], dict)
    results = json.loads(next(root.rglob("cross_val_results.json")).read_text())
    assert set(results) == {"site-1", "site-2"}
    assert all(30 <= site["SRV_FL_global_model.pt"]["accuracy"] <= 100 for site in results.values())
    if lifetime == "task":
        # POC's downloaded result contains the server workspace. Inspect the
        # retained, test-owned client workspaces for worker completion evidence.
        completion_root = root if environment == "sim" else tmp_path
        completions = [json.loads(path.read_text()) for path in completion_root.rglob("completion.json")]
        # Each site trains twice and evaluates each saved server model separately.
        expected = 4 + sum(len(site) for site in results.values())
        assert len(completions) == expected
        assert {item["identity"]["task_name"] for item in completions} == {"train", "validate"}
        assert len({item["worker_pid"] for item in completions}) == expected


@pytest.mark.timeout(300)
def test_task_pytorch_exchange_streams_large_tensors(tmp_path, monkeypatch):
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.client.config import ExchangeFormat

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    script = tmp_path / "tensor_client.py"
    script.write_text(
        """import torch
import nvflare.client as flare
from nvflare.client import FLModel

torch.set_num_threads(1)
flare.init()
while flare.is_running():
    model = flare.receive()
    # Exceeds TensorDecomposer's 2 MiB threshold: CJ must handle TENSOR_DOWNLOAD,
    # rather than relying on inline tensor reconstruction to register the codec.
    assert all(isinstance(t, torch.Tensor) for t in model.params.values())
    assert any(t.numel() * t.element_size() > 2 * 1024 * 1024 for t in model.params.values())
    assert all(torch.equal(t, torch.full_like(t, model.current_round)) for t in model.params.values())
    flare.send(FLModel(params={k: t + 1 for k, t in model.params.items()},
                       metrics={"accuracy": 1.0}, meta={"NUM_STEPS_CURRENT_ROUND": 1}))
"""
    )
    initial_ckpt = tmp_path / "initial.pt"
    torch.save({"model": {"weight": torch.zeros(1024, 1024)}}, initial_ckpt)
    recipe = FedAvgRecipe(
        name="task_pytorch_wire",
        model={"path": "torch.nn.Linear", "args": {"in_features": 1024, "out_features": 1024, "bias": False}},
        initial_ckpt=str(initial_ckpt),
        min_clients=2,
        num_rounds=2,
        train_script=str(script),
        execution_lifetime="task",
        server_expected_format=ExchangeFormat.PYTORCH,
    )
    env = SimEnv(num_clients=2, num_threads=2, workspace_root=str(tmp_path / "simulation"))
    run = recipe.execute(env)
    result = run.get_result(timeout=240, clean_up=False)
    assert result is not None and not env.last_run_failed
    root = Path(result)
    saved = torch.load(next(root.rglob("FL_global_model.pt")), map_location="cpu", weights_only=True)
    assert torch.equal(saved["model"]["weight"], torch.full((1024, 1024), 2.0))
    completions = [json.loads(path.read_text()) for path in root.rglob("completion.json")]
    assert len(completions) == len({item["worker_pid"] for item in completions}) == 4


@pytest.mark.timeout(900)
@pytest.mark.parametrize("exchange_format,transfer_type", [("numpy", "FULL"), ("pytorch", "FULL"), ("pytorch", "DIFF")])
def test_seeded_task_and_job_training_match_every_round(tmp_path, monkeypatch, exchange_format, transfer_type):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.chdir(ADVANCED_DIR)
    # Record real aggregated weights through the normal persistor contract.
    # This helper is exported with the test job, never installed in the runtime.
    recorder = tmp_path / "parity_recorder.py"
    recorder.write_text(
        """from pathlib import Path
import torch
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.client import FLModel
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

class OrderedFedAvg(ModelAggregator):
    # Three float32 contributions need a fixed summation order for exact parity.
    # Keep the existing weighted aggregation math, buffering only in this test.
    def __init__(self):
        super().__init__()
        self.reset_stats()

    def reset_stats(self):
        self.models = []

    def accept_model(self, model):
        self.models.append(model)

    def aggregate_model(self):
        params, metrics = WeightedAggregationHelper(), WeightedAggregationHelper()
        for model in sorted(self.models, key=lambda m: m.meta["client_name"]):
            weight = model.meta["NUM_STEPS_CURRENT_ROUND"]
            params.add(model.params, weight=weight, contributor_name=model.meta["client_name"],
                       contribution_round=model.current_round)
            metrics.add(model.metrics, weight=weight, contributor_name=model.meta["client_name"],
                        contribution_round=model.current_round)
        first = self.models[0]
        return FLModel(params=params.get_result(), params_type=first.params_type,
                       metrics=metrics.get_result(), current_round=first.current_round)

class RoundRecorder(PTFileModelPersistor):
    def __init__(self, model, output_dir, allow_numpy_conversion):
        super().__init__(model=model, allow_numpy_conversion=allow_numpy_conversion)
        self.output_dir = output_dir

    def save_model(self, ml, fl_ctx):
        super().save_model(ml, fl_ctx)
        folder = Path(self.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        torch.save({k: torch.as_tensor(v).clone() for k, v in ml["weights"].items()},
                   folder / f"round_{current_round}.pt")
"""
    )
    # Observe the real entry points at their API boundary. The training and
    # checkpoint code is copied unchanged into both jobs.
    client_recorders = {}
    for lifetime in ("job", "task"):
        path = tmp_path / f"capture_{lifetime}_client.py"
        path.write_text(
            """from pathlib import Path
import torch
import TRAINER_MODULE as trainer
import nvflare.client as flare

original_receive = flare.receive
original_send = flare.send
current_round = None

def snapshot(model, phase):
    folder = Path("parity")
    folder.mkdir(exist_ok=True)
    torch.save({"params": {k: torch.as_tensor(v).clone() for k, v in model.params.items()},
                "metrics": model.metrics,
                "steps": model.meta.get("NUM_STEPS_CURRENT_ROUND")},
               folder / f"{current_round}_{phase}.pt")

def receive(*args, **kwargs):
    global current_round
    model = original_receive(*args, **kwargs)
    if flare.is_train():
        current_round = model.current_round
        snapshot(model, "input")
    return model

def send(model, *args, **kwargs):
    if flare.is_train():
        snapshot(model, "output")
    return original_send(model, *args, **kwargs)

if __name__ == "__main__":
    flare.receive = receive
    flare.send = send
    trainer.main()
""".replace(
                "TRAINER_MODULE", "client"
            )
        )
        client_recorders[lifetime] = path
    monkeypatch.syspath_prepend(str(tmp_path))
    helper = importlib.import_module("parity_recorder")
    results = {}
    try:
        for lifetime in ("job", "task"):
            with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as example:
                args = example.parse_args(
                    [
                        "--execution_lifetime",
                        lifetime,
                        "--num_rounds",
                        "3",
                        "--n_clients",
                        "3",
                        "--seed",
                        "731",
                        "--torch_threads",
                        "1",
                    ]
                )
                factory = example.FedAvgRecipe

                def create_recorded_recipe(**kwargs):
                    kwargs["train_script"] = str(client_recorders[lifetime])
                    return factory(
                        **kwargs,
                        server_expected_format=exchange_format,
                        params_transfer_type=transfer_type,
                        aggregator=helper.OrderedFedAvg(),
                        model_persistor=helper.RoundRecorder(
                            example.create_model(seed=731),
                            str(tmp_path / lifetime / "rounds"),
                            allow_numpy_conversion=exchange_format != "pytorch",
                        ),
                    )

                monkeypatch.setattr(
                    example,
                    "FedAvgRecipe",
                    create_recorded_recipe,
                )
                recipe = example.create_recipe(args)
                for name in ("client.py", "model.py", "prepare_data.py"):
                    recipe.add_client_file(str(example.HELLO_PT_DIR / name))
                env = SimEnv(num_clients=3, num_threads=3, workspace_root=str(tmp_path / lifetime / "simulation"))
                run = recipe.execute(env)
                result = run.get_result(timeout=240, clean_up=False)
                assert result is not None and not env.last_run_failed
                results[lifetime] = Path(result)
    finally:
        sys.modules.pop("parity_recorder", None)
    for site in ("site-1", "site-2", "site-3"):
        for current_round in range(3):
            for phase in ("input", "output"):
                snapshots = []
                for mode in ("job", "task"):
                    paths = [p for p in results[mode].rglob(f"{current_round}_{phase}.pt") if site in p.parts]
                    assert len(paths) == 1, (mode, site, current_round, phase)
                    snapshots.append(torch.load(paths[0], weights_only=True))
                assert snapshots[0]["metrics"] == snapshots[1]["metrics"], (site, current_round, phase)
                assert snapshots[0]["steps"] == snapshots[1]["steps"], (site, current_round, phase)
                assert snapshots[0]["params"].keys() == snapshots[1]["params"].keys()
                for name in snapshots[0]["params"]:
                    assert torch.equal(snapshots[0]["params"][name], snapshots[1]["params"][name]), (
                        site,
                        current_round,
                        phase,
                        name,
                    )
    for current_round in range(3):
        states = [
            torch.load(tmp_path / mode / "rounds" / f"round_{current_round}.pt", weights_only=True)
            for mode in ("job", "task")
        ]
        assert states[0].keys() == states[1].keys()
        for name in states[0]:
            assert torch.equal(states[0][name], states[1][name]), (current_round, name)
    evaluations = [
        json.loads(next(results[mode].rglob("cross_val_results.json")).read_text()) for mode in ("job", "task")
    ]
    assert evaluations[0] == evaluations[1]
    completions = [json.loads(p.read_text()) for p in results["task"].rglob("completion.json")]
    training = [c for c in completions if c["identity"]["task_name"] == "train"]
    assert len(training) == len({c["worker_pid"] for c in training}) == 9
    checkpoints = list(results["task"].rglob("*.checkpoint"))
    assert len(checkpoints) == 3
    states = []
    for path in checkpoints:
        site = next(site for site in ("site-1", "site-2", "site-3") if site in path.parts)
        inspection = FileTaskArtifactStore(str(tmp_path / "inspect-checkpoint" / site))
        identity = TaskAttemptIdentity("simulate_job", site, "inspect", "train", "inspect")
        inspection.create_attempt(identity)
        inspection.prepare_checkpoint(identity, str(path))
        states.append(torch.load(inspection.checkpoint_paths(identity)["input"], weights_only=True))
    assert all("checkpoint" in c["diagnostics"] for c in training)
    assert all("checkpoint" not in c["diagnostics"] for c in completions if c not in training)
    assert {state["site_name"] for state in states} == {"site-1", "site-2", "site-3"}
    assert all(state["round"] == 2 for state in states)
    assert all(state["optimizer"]["state"] for state in states)
    assert all("momentum_buffer" in item for state in states for item in state["optimizer"]["state"].values())
