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
import torch

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
        args = example.parse_args(["--env", environment, "--execution_lifetime", lifetime, "--num_rounds", "2"])
        recipe = example.create_recipe(args)
        env = (
            SimEnv(num_clients=2, workspace_root=str(tmp_path / "simulation"))
            if environment == "sim"
            else example.create_environment(args)
        )
        run = recipe.execute(env)
        result = run.get_result(timeout=240, clean_up=False)
        assert result is not None
        assert run.get_status() in example.SUCCESS_STATUSES
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
