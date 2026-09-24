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
from pathlib import Path

import pytest

from nvflare.recipe import SimEnv
from tests.hello_pt_test_utils import REPO_ROOT, load_hello_pt_module

ADVANCED_DIR = REPO_ROOT / "examples" / "advanced" / "hello-pt-environments"
POC_MONITOR_TIMEOUT = 15 * 60


def _read_final_accuracies(result_path):
    result_files = list(Path(result_path).rglob("cross_val_results.json"))
    assert len(result_files) == 1
    with result_files[0].open() as result_file:
        final_results = json.load(result_file)
    return {
        site_name: site_results["SRV_FL_global_model.pt"]["accuracy"]
        for site_name, site_results in final_results.items()
    }


@pytest.mark.timeout(20 * 60)
def test_seeded_hello_pt_results_match_between_simulation_and_poc(tmp_path, monkeypatch):
    poc_env_module = importlib.import_module("nvflare.recipe.poc_env")
    poc_workspace = tmp_path / "poc-workspace"
    poc_workspace.mkdir()
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(poc_workspace))
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.chdir(ADVANCED_DIR)

    existing_pythonpath = os.environ.get("PYTHONPATH")
    source_pythonpath = str(REPO_ROOT)
    if existing_pythonpath:
        source_pythonpath = os.pathsep.join((source_pythonpath, existing_pythonpath))
    monkeypatch.setenv("PYTHONPATH", source_pythonpath)

    with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as job_module:
        simulation_recipe = job_module.create_recipe(job_module.parse_args([]))
        simulation_run = simulation_recipe.execute(SimEnv(num_clients=2, workspace_root=str(tmp_path / "simulation")))
        simulation_result = simulation_run.get_result()
        poc_args = job_module.parse_args(["--env", "poc"])
        poc_env = job_module.create_environment(poc_args)
        poc_run = job_module.create_recipe(poc_args).execute(poc_env)
        poc_result = poc_run.get_result(timeout=POC_MONITOR_TIMEOUT, clean_up=False)

    assert poc_result is not None
    assert poc_run.get_status() in job_module.SUCCESS_STATUSES
    assert _read_final_accuracies(poc_result) == _read_final_accuracies(simulation_result)
