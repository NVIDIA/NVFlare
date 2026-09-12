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
import re
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.timeout(180)
@pytest.mark.parametrize("log_mode", [None, "concise", "msg_only"])
def test_numpy_example_gets_shared_reporting_without_example_changes(tmp_path, log_mode):
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.pop("NVFLARE_HOME", None)
    if log_mode is None:
        env.pop("FL_LOG_LEVEL", None)
    else:
        # The shared Recipe option must override an inherited setting, even
        # though this older example still declares its own --log_config option.
        env["FL_LOG_LEVEL"] = "verbose"
    env["PYTHONPATH"] = os.pathsep.join((str(repo_root), env.get("PYTHONPATH", "")))
    env["NVFLARE_SIMULATOR_WORKSPACE_ROOT"] = str(tmp_path)
    command = [sys.executable, "job.py", "--num_rounds", "1"]
    if log_mode is not None:
        command.extend(["--log_config", log_mode])
    completed = subprocess.run(
        command,
        cwd=repo_root / "examples" / "hello-world" / "hello-numpy",
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    final_summary = output.split("RUN SUMMARY", 1)[1]
    assert "NVIDIA FLARE · hello-numpy" in final_summary
    assert "Simulation · 2 clients" in final_summary
    assert "NVIDIA FLARE · hello-numpy" in output
    if log_mode in (None, "concise"):
        assert output.count("ROUND 1 / 1") == 1
        assert "✓ Aggregated 2 client updates" in output
        for site in ("site-1", "site-2"):
            assert re.search(rf"{site}\s+6(?:\s|$)", output)
        assert re.search(r"Aggregated\s+6(?:\s|$)", output)
        assert "Received weights:" not in output
        assert "END_RUN received" not in output
        server_log = tmp_path / "hello-numpy" / "server" / "log_fl.txt"
        assert "ROUND 1 / 1" in server_log.read_text()
        assert re.search(r"Aggregated\s+6(?:\s|$)", server_log.read_text())
        # The same actual client messages still reach diagnostic files.
        client_log = tmp_path / "hello-numpy" / "site-1" / "log.txt"
        assert "Received weights:" in client_log.read_text()
        assert "evaluation metrics:" in client_log.read_text()
        assert "END_RUN received" in client_log.read_text()
    else:
        assert "Round 0 started." in output
        for site in ("site-1", "site-2"):
            assert f"Client {site}, current_round=0" in output
            assert f"Client {site} evaluation metrics:" in output
            assert f"Client {site} finished training for round 0" in output
        assert "Aggregated 2/2 results" in output
    assert "✓ Completed" in output
    assert f"Results   {tmp_path / 'hello-numpy'}" in output
    assert "server/simulate_job/metrics/" in output
    summary_path = tmp_path / "hello-numpy" / "server" / "simulate_job" / "metrics" / "metrics_summary.json"
    summary = json.loads(summary_path.read_text())
    assert {m["name"] for m in summary["final_aggregated_metrics"]} == {"weight_mean"}
    assert "RUN SUMMARY" in output
    assert "Training · aggregated client metrics" in output
    assert re.search(r"1\s+6(?:\s|$)", output)


@pytest.mark.timeout(180)
def test_plain_fedjob_export_gets_progress_without_configured_writer(tmp_path):
    from nvflare.app_common.np.np_model_persistor import NPModelPersistor
    from nvflare.app_common.workflows.fedavg import FedAvg
    from nvflare.job_config.api import FedJob
    from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

    repo_root = Path(__file__).resolve().parents[3]
    job = FedJob(name="plain-numpy", min_clients=2)
    job.to_server(NPModelPersistor(), id="persistor")
    job.to_server(FedAvg(num_clients=2, num_rounds=2))
    job.to_clients(
        ScriptRunner(
            script=str(repo_root / "examples/hello-world/hello-numpy/client.py"), framework=FrameworkType.NUMPY
        )
    )
    job.export_job(str(tmp_path / "export"))
    exported = tmp_path / "export" / "plain-numpy"
    server_config = next(exported.rglob("config_fed_server.json"))
    assert "MetricsArtifactWriter" not in server_config.read_text()

    # Load the ordinary exported JSON through the CLI, without Recipe setup.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(repo_root), os.environ.get("PYTHONPATH", "")))}
    env.pop("NVFLARE_HOME", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nvflare.private.fed.app.simulator.simulator",
            str(exported),
            "-w",
            str(tmp_path / "workspace"),
            "-n",
            "2",
            "-t",
            "2",
            "-l",
            "concise",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )
    output = completed.stdout + completed.stderr
    (tmp_path / "console.txt").write_text(output)
    assert completed.returncode == 0, output
    for round_number in (1, 2):
        assert output.count(f"ROUND {round_number} / 2") == 1, output
    assert output.count("✓ Aggregated 2 client updates") == 2, output
    assert "Received weights:" not in output
    server_log = (tmp_path / "workspace/server/log_fl.txt").read_text()
    assert "ROUND 2 / 2" in server_log
    summary = next((tmp_path / "workspace/server").rglob("metrics_summary.json"))
    assert json.loads(summary.read_text())["final_aggregated_metrics"]
