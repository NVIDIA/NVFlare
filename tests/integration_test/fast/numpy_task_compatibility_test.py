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

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.recipe import SimEnv

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/integration_test/data/numpy_task/numpy_observer.py"


def _site_record(root, site, name):
    paths = [p for p in root.rglob(name) if site in p.parts and p.parent.name == "numpy_evidence"]
    assert len(paths) == 1, (site, name, paths)
    with np.load(paths[0]) as data:
        return {key: data[key].copy() for key in data.files}


@pytest.mark.timeout(600)
@pytest.mark.parametrize("update_type", ["full", "diff"])
def test_shipped_numpy_fedavg_matches_job_and_task(tmp_path, monkeypatch, update_type):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    source = ROOT / "examples/hello-world/hello-numpy/client.py"
    script = tmp_path / "shipped_client.py"
    shutil.copy2(source, script)
    observer = tmp_path / "numpy_observer.py"
    shutil.copy2(FIXTURE, observer)
    initial = np.arange(6, dtype=np.float32).reshape(2, 3)
    results = {}
    for lifetime in ("job", "task"):
        recipe = NumpyFedAvgRecipe(
            name=f"numpy_{update_type}",
            model=initial.tolist(),
            min_clients=2,
            num_rounds=3,
            train_script=str(observer),
            train_args=["--update_type", update_type],
            execution_lifetime=lifetime,
            key_metric="weight_mean",
        )
        recipe.add_client_file(str(script))
        env = SimEnv(num_clients=2, num_threads=2, workspace_root=str(tmp_path / lifetime))
        run = recipe.execute(env)
        result = run.get_result(timeout=240, clean_up=False)
        assert result is not None and not env.last_run_failed
        results[lifetime] = Path(result)
        deployed = list(results[lifetime].rglob("shipped_client.py"))
        assert len(deployed) >= 2 and all(p.read_bytes() == source.read_bytes() for p in deployed)
        model_files = list(results[lifetime].rglob("server.npy"))
        assert len(model_files) == 1
        np.testing.assert_array_equal(np.load(model_files[0]), initial + 3)

    for site in ("site-1", "site-2"):
        pids = {"job": set(), "task": set()}
        for current_round in range(3):
            for phase in ("input", "output"):
                records = [_site_record(root, site, f"{current_round}_{phase}.npz") for root in results.values()]
                if phase == "output":
                    for lifetime, record in zip(results, records):
                        pids[lifetime].add(int(record.pop("pid")))
                assert records[0].keys() == records[1].keys()
                for key in records[0]:
                    np.testing.assert_array_equal(records[0][key], records[1][key])
                expected = initial + current_round
                if phase == "output":
                    expected = np.ones_like(initial) if update_type == "diff" else expected + 1
                    assert float(records[0]["weight_mean"]) == float(np.mean(initial + current_round + 1))
                np.testing.assert_array_equal(records[0][NPConstants.NUMPY_KEY], expected)
        assert len(pids["job"]) == 1 and len(pids["task"]) == 3

    completions = [json.loads(p.read_text()) for p in results["task"].rglob("completion.json")]
    assert len(completions) == len({c["worker_pid"] for c in completions}) == 6
    assert not list(results["task"].rglob("*.checkpoint"))
