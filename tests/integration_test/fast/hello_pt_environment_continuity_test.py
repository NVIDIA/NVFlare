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
    poc_workspace.mkdir()
    retained_result = poc_workspace / "retained-result"
    retained_result.write_text("previous CLI result")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(poc_workspace))
    # Keep admin transfers inside this test's workspace instead of linking them
    # to a developer's NVFLARE_HOME/examples directory.
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
        poc_result = job_module.main(["--env", "poc"])

    result_path = Path(poc_result).resolve()
    assert result_path.is_dir()
    assert list(result_path.rglob("FL_global_model.pt"))
    # Environment continuity means the fixed POC run reproduces the same
    # site-1/site-2 final accuracies as the deterministic simulation.
    assert _read_final_accuracies(poc_result) == _read_final_accuracies(simulation_result)
    recipe_workspaces = list(tmp_path.glob("poc-workspace.recipe-*"))
    assert len(recipe_workspaces) == 1
    assert result_path.is_relative_to(recipe_workspaces[0].resolve())
    assert not poc_env_module.PocEnv._running_services(
        *poc_env_module.setup_service_config(str(recipe_workspaces[0])), str(recipe_workspaces[0])
    )
    assert list(poc_workspace.iterdir()) == [retained_result]
    assert retained_result.read_text() == "previous CLI result"
    output = capsys.readouterr().out
    assert "Job Status is: FINISHED:COMPLETED" in output
    assert f"Result can be found in: {poc_result}" in output


@pytest.mark.timeout(180)
def test_failed_poc_job_retains_download_and_client_error_logs(tmp_path, monkeypatch, capsys):
    poc_env_module = importlib.import_module("nvflare.recipe.poc_env")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.chdir(ADVANCED_DIR)

    with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as job_module:
        # Invalid client batch size intentionally fails during actual client
        # execution, after provisioning and submission, without a data download.
        with pytest.raises(RuntimeError, match="unsuccessful status: FINISHED"):
            job_module.main(["--env", "poc", "--num_rounds", "1", "--batch_size", "0"])

    output = capsys.readouterr().err
    result_line = next(line for line in output.splitlines() if line.startswith("Result can be found in:"))
    result_path = Path(result_line.split(":", 1)[1].strip())
    assert result_path.is_dir()
    workspaces = list(tmp_path.glob("poc.recipe-*"))
    assert len(workspaces) == 1
    workspace = workspaces[0]
    logs = list(workspace.rglob("poc_console.log"))
    assert logs
    assert any("batch_size should be a positive integer" in log.read_text() for log in logs)
    assert str(workspace) in output
    assert all(str(log) in output for log in logs)
    assert not poc_env_module.PocEnv._running_services(
        *poc_env_module.setup_service_config(str(workspace)), str(workspace)
    )
