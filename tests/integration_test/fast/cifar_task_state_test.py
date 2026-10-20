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
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nvflare.apis.dxo import DataKind
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe
from nvflare.app_opt.pt.recipes.fedprox import FedProxRecipe
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity
from nvflare.recipe import SimEnv

REPO_ROOT = Path(__file__).resolve().parents[3]
CIFAR_ROOT = REPO_ROOT / "examples/advanced/cifar10/pt"
SOURCE_DIR = CIFAR_ROOT / "src"
FIXTURE_DIR = REPO_ROOT / "tests/integration_test/data/cifar_task"


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _read_site(root, site, filename):
    files = [p for p in root.rglob(filename) if site in p.parts and p.parent.name == "cifar_state_evidence"]
    assert len(files) == 1, (site, filename, files)
    return torch.load(files[0], weights_only=True)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("algorithm", ["fedavg", "fedopt", "fedprox"])
@pytest.mark.parametrize(
    "num_workers", [pytest.param(0, id="no_data_workers"), pytest.param(2, id="persistent_workers")]
)
def test_shipped_cifar_clients_restore_native_training_state(tmp_path, monkeypatch, algorithm, num_workers):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    for directory in (SOURCE_DIR, FIXTURE_DIR, tmp_path):
        monkeypatch.syspath_prepend(str(directory))
    fixture = importlib.import_module("cifar_task_fixture")
    source = CIFAR_ROOT / "cifar10-sim" / f"cifar10_{algorithm}" / "client.py"
    client_copy = tmp_path / "cifar_client.py"
    shutil.copy2(source, client_copy)
    results = {}
    try:
        for lifetime in ("job", "task"):
            options = dict(
                name=f"cifar_{algorithm}",
                model=fixture.TinyCifarModel(),
                min_clients=2,
                num_rounds=3,
                train_script=str(FIXTURE_DIR / "cifar_task_fixture.py"),
                train_args=f"--num_workers {num_workers}",
            )
            if algorithm == "fedopt":
                recipe = FedOptRecipe(
                    **options,
                    device="cpu",
                    optimizer_args={"path": "torch.optim.SGD", "args": {"lr": 0.5, "momentum": 0.6}},
                )
                locator_id = "locator"
            else:
                recipe_class = FedAvgRecipe if algorithm == "fedavg" else FedProxRecipe
                recipe = recipe_class(**options, aggregator_data_kind=DataKind.WEIGHT_DIFF)
                locator_id = recipe._job.comp_ids["locator_id"]
            recipe.add_client_config({"execution_lifetime": lifetime})
            for path in (
                client_copy,
                SOURCE_DIR / "model.py",
                SOURCE_DIR / "train_utils.py",
                SOURCE_DIR / "task_checkpoint.py",
            ):
                recipe.add_client_file(str(path))
            recipe._job.add_file_to_clients(str(SOURCE_DIR / "data"), dest_dir="data")
            # Use the existing Job API in this independent slice. The separate
            # FedOpt recipe metadata patch enables the convenience helper.
            recipe._job.to_server(CrossSiteModelEval(model_locator_id=locator_id, submit_model_task_name=""))
            env = SimEnv(num_clients=2, num_threads=2, workspace_root=str(tmp_path / lifetime))
            run = recipe.execute(env)
            result = run.get_result(timeout=240, clean_up=False)
            assert result is not None and not env.last_run_failed
            results[lifetime] = Path(result)
            copies = list(results[lifetime].rglob("cifar_client.py"))
            assert len(copies) >= 2
            assert all(path.read_bytes() == source.read_bytes() for path in copies)
    finally:
        sys.modules.pop("cifar_task_fixture", None)

    for site in ("site-1", "site-2"):
        for current_round in range(3):
            for root in results.values():
                state = _read_site(root, site, f"{current_round}_state.pt")
                assert state["optimizer"]["state"]
                assert state["scheduler"]["last_epoch"] == 2 * (current_round + 1)
                output = _read_site(root, site, f"{current_round}_output.pt")
                assert output["steps"] == 8
            if current_round:
                # Check exactly what the fresh task process restores before
                # training. Ordinary PyTorch data-worker state is not included.
                previous = _read_site(results["task"], site, f"{current_round - 1}_state.pt")
                resumed = _read_site(results["task"], site, f"{current_round}_resume.pt")
                _equal(previous, resumed)
            if num_workers == 0:
                for suffix in ("input", "output", "state"):
                    states = [_read_site(root, site, f"{current_round}_{suffix}.pt") for root in results.values()]
                    _equal(*states)
        final = _read_site(results["task"], site, "2_state.pt")
        assert final["optimizer"]["param_groups"][0]["lr"] < 0.02
        checkpoints = [p for p in results["task"].rglob("*.checkpoint") if site in p.parts]
        assert len(checkpoints) == 1
        store = FileTaskArtifactStore(str(tmp_path / "inspect" / site))
        identity = TaskAttemptIdentity("simulate_job", site, "inspect", "train", "inspect")
        store.create_attempt(identity)
        store.prepare_checkpoint(identity, str(checkpoints[0]))
        saved = torch.load(store.checkpoint_paths(identity)["input"], weights_only=True)
        assert saved["round"] == 2 and saved["site_name"] == site
        for key in ("optimizer", "scheduler", "torch_rng", "python_rng", "numpy_rng"):
            _equal(saved[key], final[key])
    models = [
        torch.load(next(root.rglob("FL_global_model.pt")), weights_only=True)["model"] for root in results.values()
    ]
    evaluations = [json.loads(next(root.rglob("cross_val_results.json")).read_text()) for root in results.values()]
    if num_workers == 0:
        _equal(*models)
        assert evaluations[0] == evaluations[1]
    # With persistent workers, recreating the native DataLoader changes its
    # worker RNG streams and seed draws. These cases check checkpoint restore
    # and task execution, without claiming job/task numerical parity.
    completions = [json.loads(p.read_text()) for p in results["task"].rglob("completion.json")]
    assert (
        len(completions)
        == len({c["worker_pid"] for c in completions})
        == 6 + sum(len(v) for v in evaluations[0].values())
    )
    assert {c["identity"]["task_name"] for c in completions} == {"train", "validate"}
    assert sum("checkpoint" in c["diagnostics"] for c in completions) == 6
