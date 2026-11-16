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

from tests.hello_pt_test_utils import (
    HELLO_PT_DIR,
    REPO_ROOT,
    assert_hello_pt_failure_summary,
    load_hello_pt_module,
    run_hello_pt_export,
)

ADVANCED_DIR = REPO_ROOT / "examples" / "advanced" / "hello-pt-environments"
pytestmark = pytest.mark.usefixtures("single_threaded_torch")


@pytest.mark.parametrize("environment", ["sim", "poc"])
def test_task_cli_export_remains_available_for_native_sites(tmp_path, environment):
    run_hello_pt_export(
        [
            sys.executable,
            "job.py",
            "--env",
            environment,
            "--execution_lifetime",
            "task",
            "--export",
            "--export-dir",
            str(tmp_path),
        ],
        cwd=ADVANCED_DIR,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    config = json.loads((tmp_path / "hello-pt" / "app" / "config" / "config_fed_client.json").read_text())
    assert config["execution_lifetime"] == "task"
    assert config["format_version"] == 3


@pytest.mark.parametrize("resume_flag", ["--resume_training", "--no-resume_training"])
def test_beginner_task_cli_export_preserves_checkpoint_selection(tmp_path, resume_flag):
    run_hello_pt_export(
        [
            sys.executable,
            "job.py",
            "--execution_lifetime",
            "task",
            resume_flag,
            "--export",
            "--export-dir",
            str(tmp_path),
        ],
        cwd=HELLO_PT_DIR,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    config = json.loads((tmp_path / "hello-pt" / "app" / "config" / "config_fed_client.json").read_text())
    assert config["execution_lifetime"] == "task"
    assert config["executors"][0]["executor"]["args"]["task_script_args"] == [
        "--dataset",
        "synthetic",
        resume_flag,
    ]


@pytest.fixture
def poc_env_module(monkeypatch):
    module = importlib.import_module("nvflare.recipe.poc_env")
    get_job_result = module.PocEnv.get_job_result

    def bounded_job_result(env, job_id, timeout=0.0):
        # The CLI waits indefinitely by default. Let its real monitor time out
        # and close normally before pytest's outer guard interrupts cleanup.
        return get_job_result(env, job_id, timeout=timeout or 120)

    monkeypatch.setattr(module.PocEnv, "get_job_result", bounded_job_result)
    return module


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
def test_hello_pt_application_runs_in_isolated_poc(tmp_path, monkeypatch, capsys, poc_env_module):
    """Two clients aggregate and evaluate, retaining results in an isolated POC."""
    monkeypatch.setenv("FL_LOG_LEVEL", "progress")
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
        # Keep the real three-round learning/evaluation contract while bounding
        # each client's CPU use independently of the host's core count.
        poc_result = job_module.main(["--env", "poc", "--torch_threads", "1"])

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
def test_failed_poc_job_retains_download_and_client_error_logs(
    tmp_path, monkeypatch, capsys, missing_cifar, poc_env_module
):
    """A deployed client failure preserves diagnostics after the POC services stop."""
    monkeypatch.setenv("FL_LOG_LEVEL", "concise")
    monkeypatch.setattr(poc_env_module, "get_poc_workspace", lambda: str(tmp_path / "poc"))
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(REPO_ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.chdir(ADVANCED_DIR)

    with load_hello_pt_module("job.py", example_dir=ADVANCED_DIR) as job_module:
        # Either injected failure occurs in the client after deployment, without
        # downloading data. One client exercises diagnostic retention and cleanup;
        # peer cancellation and error grouping have deterministic unit coverage.
        if missing_cifar:
            client_args = ["--dataset", "cifar10", "--data_root", str(tmp_path / "missing")]
            expected_error = "python prepare_data.py --data_root"
        else:
            client_args = ["--batch_size", "0"]
            expected_error = "batch_size should be a positive integer"
        with pytest.raises(RuntimeError, match="unsuccessful status: FINISHED"):
            job_module.main(["--env", "poc", "--n_clients", "1", "--num_rounds", "1", *client_args])

    captured = capsys.readouterr()
    summary = captured.out.split("RUN SUMMARY", 1)[1]
    assert "NVIDIA FLARE · hello-pt" in summary
    assert "POC · 1 client" in summary
    assert_hello_pt_failure_summary(
        summary,
        expected_error=expected_error,
        expected_file="prepare_data.py:" if missing_cifar else "client.py:",
        expected_site="site-1",
    )
    output = captured.err
    result_line = next(line for line in output.splitlines() if line.startswith("Result can be found in:"))
    result_path = Path(result_line.split(":", 1)[1].strip())
    assert result_path.is_dir()
    assert (result_path / "workspace" / "log.txt").is_file()
    client_logs = {path.parent.name: path for path in result_path.glob("failure-logs-*/*/error_log.txt")}
    assert set(client_logs) == {"site-1"}
    assert expected_error in client_logs["site-1"].read_text()
    workspaces = list(tmp_path.glob("poc.recipe-*"))
    assert len(workspaces) == 1
    workspace = workspaces[0]
    # Verify the stored client error stream was downloaded intact.
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
