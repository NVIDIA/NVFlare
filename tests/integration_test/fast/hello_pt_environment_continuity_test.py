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
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from nvflare.recipe import SimEnv

REPO_ROOT = Path(__file__).resolve().parents[3]
ADVANCED_DIR = REPO_ROOT / "examples" / "advanced" / "hello-pt-environments"
HELLO_PT_DIR = REPO_ROOT / "examples" / "hello-world" / "hello-pt"


@contextmanager
def _load_job_module():
    module_path = ADVANCED_DIR / "job.py"
    spec = importlib.util.spec_from_file_location("hello_pt_environment_continuity_job", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    original_sys_path = list(sys.path)
    original_modules = {name: sys.modules.pop(name, None) for name in ("model", "prepare_data")}
    sys.path.insert(0, str(ADVANCED_DIR))
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_sys_path
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _read_final_accuracies(result_path):
    result_files = list(Path(result_path).rglob("cross_val_results.json"))
    assert len(result_files) == 1
    with result_files[0].open() as result_file:
        final_results = json.load(result_file)
    return {
        site_name: site_results["SRV_FL_global_model.pt"]["accuracy"]
        for site_name, site_results in final_results.items()
    }


@pytest.mark.timeout(300)
def test_hello_pt_reuses_the_application_in_poc(tmp_path, monkeypatch, capsys):
    poc_env_module = importlib.import_module("nvflare.recipe.poc_env")
    poc_workspace = tmp_path / "poc-workspace"
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(poc_workspace))
    monkeypatch.chdir(ADVANCED_DIR)

    existing_pythonpath = os.environ.get("PYTHONPATH")
    source_pythonpath = str(REPO_ROOT)
    if existing_pythonpath:
        source_pythonpath = os.pathsep.join((source_pythonpath, existing_pythonpath))
    monkeypatch.setenv("PYTHONPATH", source_pythonpath)

    with _load_job_module() as job_module:
        simulation_recipe = job_module.create_recipe(job_module.parse_args([]))
        simulation_run = simulation_recipe.execute(SimEnv(num_clients=2, workspace_root=str(tmp_path / "simulation")))
        simulation_result = simulation_run.get_result()
        poc_result = job_module.main(["--env", "poc"])

    result_path = Path(poc_result).resolve()
    assert result_path.is_dir()
    assert list(result_path.rglob("FL_global_model.pt"))
    # Environment continuity means the fixed POC run reproduces the same
    # site-1/site-2 final accuracies as the deterministic simulation.
    assert _read_final_accuracies(poc_result) == _read_final_accuracies(simulation_result)
    assert poc_workspace.is_dir()
    output = capsys.readouterr().out
    assert "Job Status is: FINISHED:COMPLETED" in output
    assert f"Result can be found in: {poc_result}" in output
