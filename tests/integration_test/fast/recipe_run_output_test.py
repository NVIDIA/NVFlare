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
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.timeout(180)
def test_numpy_example_gets_shared_reporting_without_example_changes(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env.pop("NVFLARE_HOME", None)
    env["PYTHONPATH"] = os.pathsep.join((str(repo_root), env.get("PYTHONPATH", "")))
    env["NVFLARE_SIMULATOR_WORKSPACE_ROOT"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "job.py", "--num_rounds", "1"],
        cwd=repo_root / "examples" / "hello-world" / "hello-numpy",
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Executing job 'hello-numpy' with SimEnv" in output
    assert "Training round 1/1 started" in output
    for site in ("site-1", "site-2"):
        assert f"Client {site}, current_round=0" in output
        assert f"Client {site} evaluation metrics:" in output
        assert f"Client {site} finished training for round 0" in output
    assert "Aggregated 2/2 results" in output
    assert "Job hello-numpy status: FINISHED:COMPLETED" in output
    assert f"Result workspace: {tmp_path / 'hello-numpy'}" in output
    assert "Aggregated metrics summary:" in output
    assert "Round metrics:" in output
    assert "error_log.txt" in output
    summary_path = tmp_path / "hello-numpy" / "server" / "simulate_job" / "metrics" / "metrics_summary.json"
    summary = json.loads(summary_path.read_text())
    assert {m["name"] for m in summary["final_aggregated_metrics"]} == {"weight_mean"}
    assert str(summary_path) in output
