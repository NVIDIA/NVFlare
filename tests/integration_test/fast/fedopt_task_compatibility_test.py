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
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity
from nvflare.recipe import SimEnv, add_final_global_evaluation
from tests.hello_pt_test_utils import HELLO_PT_DIR, REPO_ROOT, load_hello_pt_module

FIXTURE_DIR = REPO_ROOT / "tests/integration_test/data/fedopt_task"


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _client_file(root, site, name):
    matches = [path for path in root.rglob(name) if site in path.parts and path.parent.name == "fedopt_evidence"]
    assert len(matches) == 1, (root, site, name, matches)
    return torch.load(matches[0], weights_only=True)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("exchange_format", [ExchangeFormat.NUMPY, ExchangeFormat.PYTORCH])
def test_fedopt_job_task_model_and_optimizer_continuity(tmp_path, monkeypatch, exchange_format):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.syspath_prepend(str(FIXTURE_DIR))
    observer = importlib.import_module("fedopt_observer")
    results = {}
    initial = None
    try:
        for lifetime in ("job", "task"):
            with load_hello_pt_module("job.py") as model_module:
                model = model_module.create_model(seed=731)
                initial = observer.snapshot(model.state_dict())
                recipe = FedOptRecipe(
                    name="fedopt_compatibility",
                    model=model,
                    min_clients=2,
                    num_rounds=3,
                    train_script=str(FIXTURE_DIR / "fedopt_observer.py"),
                    train_args="--dataset synthetic --seed 731 --torch_threads 1 --train_size 32 --test_size 16",
                    server_expected_format=exchange_format,
                    optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 0.5, "momentum": 0.6}},
                    lr_scheduler_args={
                        "path": "torch.optim.lr_scheduler.StepLR",
                        "args": {"step_size": 1, "gamma": 0.8},
                    },
                    device="cpu",
                    execution_lifetime=lifetime,
                )
                recipe._job.to_server(observer.FedOptObserver(str(tmp_path / lifetime / "rounds")))
                for name in ("client.py", "model.py", "prepare_data.py"):
                    recipe.add_client_file(str(HELLO_PT_DIR / name))
                add_final_global_evaluation(recipe)
                env = SimEnv(num_clients=2, num_threads=2, workspace_root=str(tmp_path / lifetime / "simulation"))
                run = recipe.execute(env)
                result = run.get_result(timeout=240, clean_up=False)
                assert result is not None and not env.last_run_failed
                results[lifetime] = Path(result)
                deployed_clients = list(results[lifetime].rglob("client.py"))
                assert len(deployed_clients) >= 2
                assert all(path.read_bytes() == (HELLO_PT_DIR / "client.py").read_bytes() for path in deployed_clients)
    finally:
        sys.modules.pop("fedopt_observer", None)

    server_pids = {mode: set() for mode in results}
    # Independently reconstruct the server SGD recurrence from the actual DIFFs.
    expected = initial
    momentum = {name: torch.zeros_like(value) for name, value in initial.items()}
    for current_round in range(3):
        states = [
            torch.load(tmp_path / mode / "rounds" / f"round_{current_round}.pt", weights_only=True) for mode in results
        ]
        for mode, state in zip(results, states):
            server_pids[mode].add(state.pop("pid"))
        _equal(*states)
        state = states[0]
        assert state["scheduler"]["last_epoch"] == current_round + 1
        assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.5 * 0.8 ** (current_round + 1))
        for index, name in enumerate(expected):
            momentum[name] = 0.6 * momentum[name] - state["diff"][name]
            expected[name] = expected[name] - (0.5 * 0.8**current_round) * momentum[name]
            torch.testing.assert_close(state["model"][name], expected[name], rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(state["optimizer"]["state"][index]["momentum_buffer"], momentum[name])
        for site in ("site-1", "site-2"):
            inputs = [_client_file(root, site, f"{current_round}_input.pt") for root in results.values()]
            _equal(*inputs)
            outputs = [_client_file(root, site, f"{current_round}_output.pt") for root in results.values()]
            for output in outputs:
                output.pop("pid")
            _equal(*outputs)
            assert outputs[0]["steps"] > 0
    assert all(len(pids) == 1 for pids in server_pids.values())

    for site in ("site-1", "site-2"):
        for mode, root in results.items():
            pids = {_client_file(root, site, f"{r}_output.pt")["pid"] for r in range(3)}
            assert len(pids) == (1 if mode == "job" else 3)
        checkpoints = [path for path in results["task"].rglob("*.checkpoint") if site in path.parts]
        assert len(checkpoints) == 1
        store = FileTaskArtifactStore(str(tmp_path / "inspect" / site))
        identity = TaskAttemptIdentity("simulate_job", site, "inspect", "train", "inspect")
        store.create_attempt(identity)
        store.prepare_checkpoint(identity, str(checkpoints[0]))
        saved = torch.load(store.checkpoint_paths(identity)["input"], weights_only=True)
        last = _client_file(results["job"], site, "2_output.pt")
        assert saved["round"] == 2 and saved["site_name"] == site
        for key in ("optimizer", "train_rng", "torch_rng"):
            _equal(saved[key], last[key])

    models = [
        torch.load(next(root.rglob("FL_global_model.pt")), weights_only=True)["model"] for root in results.values()
    ]
    _equal(*models)
    evaluations = [json.loads(next(root.rglob("cross_val_results.json")).read_text()) for root in results.values()]
    assert evaluations[0] == evaluations[1]
    assert set(evaluations[0]) == {"site-1", "site-2"}
    completions = [json.loads(path.read_text()) for path in results["task"].rglob("completion.json")]
    training = [item for item in completions if item["identity"]["task_name"] == "train"]
    assert len(training) == 6
    assert (
        len(completions)
        == len({item["worker_pid"] for item in completions})
        == 6 + sum(len(s) for s in evaluations[0].values())
    )
    assert {item["identity"]["task_name"] for item in completions} == {"train", "validate"}
    assert all("checkpoint" in item["diagnostics"] for item in training)
