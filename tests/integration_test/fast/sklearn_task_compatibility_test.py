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

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe
from nvflare.app_opt.sklearn.recipes.kmeans import KMeansFedAvgRecipe
from nvflare.app_opt.sklearn.recipes.svm import SVMFedAvgRecipe
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity
from nvflare.recipe import SimEnv, set_per_site_config

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/integration_test/data/sklearn_task"


def _equal(left, right, tolerance=0):
    if isinstance(left, np.ndarray):
        if tolerance and np.issubdtype(left.dtype, np.floating):
            np.testing.assert_allclose(left, right, rtol=tolerance, atol=tolerance)
        else:
            np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key], tolerance)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b, tolerance)
    elif tolerance and isinstance(left, (float, np.floating)):
        assert left == pytest.approx(right, rel=tolerance, abs=tolerance)
    else:
        assert left == right


def _read_site(root, site, name):
    paths = [p for p in root.rglob(name) if site in p.parts and p.parent.name == "sklearn_evidence"]
    assert len(paths) == 1, (site, name, paths)
    return joblib.load(paths[0])


def _ordered_support_records(params):
    # Support records are a multiset: preserve every row/label, compare independent
    # of client-result arrival order. Raw wire records remain in the run artifacts.
    if "support_x" not in params:
        return params
    records = np.column_stack((params["support_y"], params["support_x"]))
    order = np.lexsort(records.T[::-1])
    return {**params, "support_x": params["support_x"][order], "support_y": params["support_y"][order]}


@pytest.mark.timeout(600)
@pytest.mark.parametrize("algorithm", ["linear", "kmeans", "svm", "svm-linear"])
def test_shipped_sklearn_clients_continue_after_round_zero(tmp_path, monkeypatch, algorithm):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(ROOT), os.environ.get("PYTHONPATH", ""))))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.syspath_prepend(str(FIXTURE))
    observer = importlib.import_module("sklearn_observer")
    labels = np.tile([0, 1], 40)
    features = np.random.default_rng(731).normal(size=(80, 3)) + 3 * labels[:, None]
    data = tmp_path / "samples.csv"
    pd.DataFrame(np.column_stack((labels, features))).to_csv(data, header=False, index=False)
    source = ROOT / "examples/advanced" / f"sklearn-{algorithm.split('-')[0]}" / "client.py"
    script = tmp_path / "shipped_client.py"
    shutil.copy2(source, script)
    results = {}
    is_svm = algorithm.startswith("svm")
    num_rounds = 2 if is_svm else 3
    try:
        for lifetime in ("job", "task"):
            options = dict(
                name=f"sklearn_{algorithm}",
                min_clients=2,
                train_script=str(FIXTURE / "sklearn_observer.py"),
                execution_lifetime=lifetime,
            )
            if algorithm == "kmeans":
                recipe = KMeansFedAvgRecipe(**options, num_rounds=num_rounds, n_clusters=2)
            elif is_svm:
                recipe = SVMFedAvgRecipe(**options, kernel="linear" if algorithm == "svm-linear" else "rbf")
            else:
                recipe = SklearnFedAvgRecipe(
                    **options,
                    num_rounds=num_rounds,
                    model_params={
                        "n_classes": 2,
                        "learning_rate": "constant",
                        "eta0": 0.01,
                        "loss": "log_loss",
                        "penalty": "l2",
                        "fit_intercept": 1,
                    },
                    key_metric="AUC",
                )
            set_per_site_config(
                recipe,
                {
                    f"site-{site + 1}": {
                        "train_args": f"--data_path {data} --train_start {30 * site} "
                        f"--train_end {30 * (site + 1)} --valid_start 60 --valid_end 80"
                    }
                    for site in range(2)
                },
            )
            recipe.add_client_file(str(script))
            recipe._job.to_server(observer.SeededServerRun())
            env = SimEnv(clients=["site-1", "site-2"], num_threads=2, workspace_root=str(tmp_path / lifetime))
            run = recipe.execute(env)
            result = run.get_result(timeout=240, clean_up=False)
            assert result is not None and not env.last_run_failed
            results[lifetime] = Path(result)
            deployed = list(results[lifetime].rglob("shipped_client.py"))
            assert len(deployed) >= 2 and all(p.read_bytes() == source.read_bytes() for p in deployed)
    finally:
        sys.modules.pop("sklearn_observer", None)

    # K-means reductions can differ at floating-point roundoff precision.
    tolerance = 1e-12 if algorithm == "kmeans" else 0
    for site in ("site-1", "site-2"):
        pids = {}
        for lifetime, root in results.items():
            pids[lifetime] = {_read_site(root, site, f"{r}_output.joblib")["pid"] for r in range(num_rounds)}
        assert len(pids["job"]) == 1 and len(pids["task"]) == num_rounds
        for current_round in range(num_rounds):
            for phase in ("input", "output"):
                values = [_read_site(root, site, f"{current_round}_{phase}.joblib") for root in results.values()]
                for value in values:
                    value.pop("pid", None)
                    if is_svm:
                        value["params"] = _ordered_support_records(value["params"])
                    # Participant weights are recorded in arrival order, not name order.
                    info = value.get("meta", {}).get("metrics_aggregation_info")
                    if info is not None and "site_weights" in info:
                        info["site_weights"].sort(key=lambda item: item["name"])
                _equal(*values, tolerance=tolerance)
            if is_svm:
                output = _read_site(results["task"], site, f"{current_round}_output.joblib")
                assert set(output["params"]) == {"support_x", "support_y"}
                assert set(np.unique(output["params"]["support_y"])) == {0, 1}
                if current_round == 1:
                    assert output["metrics"]["AUC"] >= 0.9
            if algorithm == "linear":
                _equal(*[_read_site(root, site, f"{current_round}_classifier.joblib") for root in results.values()])
        checkpoints = [p for p in results["task"].rglob("*.checkpoint") if site in p.parts]
        if algorithm == "linear":
            assert len(checkpoints) == 1
            store = FileTaskArtifactStore(str(tmp_path / "inspect" / site))
            identity = TaskAttemptIdentity("simulate_job", site, "inspect", "train", "inspect")
            store.create_attempt(identity)
            store.prepare_checkpoint(identity, str(checkpoints[0]))
            saved = joblib.load(store.checkpoint_paths(identity)["input"])
            assert saved["round"] == 2 and saved["site_name"] == site
            _equal(observer.classifier_state(saved["model"]), _read_site(results["job"], site, "2_classifier.joblib"))
        else:
            assert not checkpoints
    models = [joblib.load(next(root.rglob("model_param.joblib"))) for root in results.values()]
    if is_svm:
        models = [_ordered_support_records(model) for model in models]
    _equal(*models, tolerance=tolerance)
    completions = [json.loads(p.read_text()) for p in results["task"].rglob("completion.json")]
    assert len(completions) == len({c["worker_pid"] for c in completions}) == 2 * num_rounds
    assert sum("checkpoint" in c["diagnostics"] for c in completions) == (
        2 * num_rounds if algorithm == "linear" else 0
    )
