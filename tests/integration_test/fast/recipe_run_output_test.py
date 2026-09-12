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
        env["FL_LOG_LEVEL"] = log_mode
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
    assert "NVIDIA FLARE · hello-numpy" in output
    if log_mode in (None, "concise"):
        assert "ROUND 1 / 1" in output
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
