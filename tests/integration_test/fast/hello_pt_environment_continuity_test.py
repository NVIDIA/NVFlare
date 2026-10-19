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

from tests.hello_pt_test_utils import REPO_ROOT, assert_hello_pt_failure_summary, load_hello_pt_module

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
def test_hello_pt_application_runs_in_isolated_poc(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FL_LOG_LEVEL", "progress")
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
        poc_result = job_module.main(["--env", "poc"])

    result_path = Path(poc_result).resolve()
    assert result_path.is_dir()
    assert list(result_path.rglob("FL_global_model.pt"))
    final_accuracies = _read_final_accuracies(poc_result)
    assert set(final_accuracies) == {"site-1", "site-2"}
    assert min(final_accuracies.values()) >= 60.0
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
    # Live progress replays a bounded log tail. A delayed monitor need not see
    # every training phase, or poll while the job is still RUNNING.
    server_log = "\n".join(
        json.loads(line).get("message", "")
        for line in (result_path / "workspace" / "log.json").read_text().splitlines()
    )
    assert "ROUND 3 / 3" in server_log
    assert "✓ Aggregated 2 client updates" in server_log
    assert "Evaluating saved models on 2 clients" in server_log
    metrics = json.loads((result_path / "workspace" / "metrics" / "metrics_summary.json").read_text())
    assert metrics["final_round"] == 2
    assert "accuracy_after_local_training" in {metric["name"] for metric in metrics["final_aggregated_metrics"]}
    assert "SRV_FL_global_model.pt" in output
    assert "Job status: FINISHED:COMPLETED" in output
    assert "Job Meta:" not in output
    assert f"Results   {poc_result}" in output
    assert "Model evaluation · accuracy" in output


@pytest.mark.timeout(180)
@pytest.mark.parametrize("missing_cifar", [False, True])
def test_failed_poc_job_retains_download_and_client_error_logs(tmp_path, monkeypatch, capsys, missing_cifar):
    monkeypatch.setenv("FL_LOG_LEVEL", "concise")
    poc_env_module = importlib.import_module("nvflare.recipe.poc_env")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.chdir(ADVANCED_DIR)

    with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as job_module:
        # Both failures must occur on the clients after deployment and leave
        # the diagnostic logs available, without downloading data.
        if missing_cifar:
            client_args = ["--dataset", "cifar10", "--data_root", str(tmp_path / "missing")]
            expected_error = "python prepare_data.py --data_root"
        else:
            client_args = ["--batch_size", "0"]
            expected_error = "batch_size should be a positive integer"
        with pytest.raises(RuntimeError, match="unsuccessful status: FINISHED"):
            job_module.main(["--env", "poc", "--num_rounds", "1", *client_args])

    captured = capsys.readouterr()
    summary = captured.out.split("RUN SUMMARY", 1)[1]
    assert "NVIDIA FLARE · hello-pt" in summary
    assert "POC · 2 clients" in summary
    # Either client can fail first and abort its peer before the peer records
    # the same exception. Unit tests exercise these same assertions with both
    # orderings and with identical errors grouped together.
    assert_hello_pt_failure_summary(
        summary, expected_error=expected_error, expected_file="prepare_data.py:" if missing_cifar else "client.py:"
    )
    output = captured.err
    result_line = next(line for line in output.splitlines() if line.startswith("Result can be found in:"))
    result_path = Path(result_line.split(":", 1)[1].strip())
    assert result_path.is_dir()
    client_logs = {path.parent.name: path for path in result_path.glob("failure-logs-*/*/error_log.txt")}
    assert client_logs
    assert set(client_logs) <= {"site-1", "site-2"}
    assert any(expected_error in path.read_text() for path in client_logs.values())
    workspaces = list(tmp_path.glob("poc.recipe-*"))
    assert len(workspaces) == 1
    workspace = workspaces[0]
    # Verify every stored error stream was downloaded, without requiring a
    # cancelled peer to have emitted an error of its own.
    stored_logs = {
        path.name.removeprefix("ERRORLOG_"): path
        for path in (workspace / "jobs-storage" / result_path.name).glob("ERRORLOG_*")
        if path.stat().st_size > 0
    }
    assert set(client_logs) == set(stored_logs)
    assert all(path.read_bytes() == stored_logs[site].read_bytes() for site, path in client_logs.items())
    logs = list(workspace.rglob("poc_console.log"))
    assert logs
    assert any(expected_error in log.read_text() for log in logs)
    assert str(workspace) in output
    assert all(str(log) in output for log in logs)
    assert not poc_env_module.PocEnv._running_services(
        *poc_env_module.setup_service_config(str(workspace)), str(workspace)
    )
