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

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARD = REPO_ROOT / "research" / "auto-fl-research" / "scripts" / "campaign_guard.py"
HEADER = "commit\tscore\truntime_seconds\tbudget\tstatus\ttarget\tdescription\tartifacts\n"


def _write_results(path, rows):
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")


def _run_guard(tmp_path, *args, env=None):
    state_path = tmp_path / "campaign_state.json"
    command = [
        sys.executable,
        str(GUARD),
        str(tmp_path / "results.tsv"),
        "--state",
        str(state_path),
        "--format",
        "json",
        *args,
    ]
    process_env = None if env is None else {**os.environ, **env}
    process = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=True, env=process_env)
    payload = json.loads(process.stdout)
    assert json.loads(state_path.read_text(encoding="utf-8")) == payload
    return payload


def test_campaign_guard_continues_uncapped_after_verified_default(tmp_path):
    _write_results(
        tmp_path / "results.tsv",
        [
            "abc\t0.84\t10\t--name baseline\tkeep\tjob.py\tbaseline\t/tmp/baseline",
            "def\t0.87\t20\t--name encoded\tkeep\tjob.py\tencoded defaults verified\t/tmp/encoded",
        ],
    )

    payload = _run_guard(tmp_path)

    assert payload["decision"] == "continue"
    assert payload["next_action"] == "launch_next_candidate_batch"
    assert payload["final_response_allowed"] is False
    assert "Do not produce a final answer" in payload["agent_instruction"]


def test_campaign_guard_requires_finalizing_pending_candidates(tmp_path):
    _write_results(
        tmp_path / "results.tsv",
        [
            "abc\t0.84\t10\t--name baseline\tkeep\tjob.py\tbaseline\t/tmp/baseline",
            "def\t0.86\t20\t--name candidate\tcandidate\tjob.py\tcandidate row\t/tmp/candidate",
        ],
    )

    payload = _run_guard(tmp_path)

    assert payload["decision"] == "continue"
    assert payload["reason"] == "pending_candidates"
    assert payload["next_action"] == "finalize_pending_candidates"
    assert payload["final_response_allowed"] is False


def test_campaign_guard_allows_final_report_after_explicit_cap(tmp_path):
    _write_results(
        tmp_path / "results.tsv",
        [
            "abc\t0.84\t10\t--name baseline\tkeep\tjob.py\tbaseline\t/tmp/baseline",
            "def\t0.86\t20\t--name candidate\tdiscard\tjob.py\tcandidate row\t/tmp/candidate",
        ],
    )

    payload = _run_guard(tmp_path, "--max-candidates", "1")

    assert payload["decision"] == "stop"
    assert payload["reason"] == "candidate_cap_exhausted"
    assert payload["next_action"] == "final_report"
    assert payload["final_response_allowed"] is True


def test_campaign_guard_ignores_ambient_env_cap(tmp_path):
    _write_results(
        tmp_path / "results.tsv",
        [
            "abc\t0.84\t10\t--name baseline\tkeep\tjob.py\tbaseline\t/tmp/baseline",
            "def\t0.86\t20\t--name candidate\tdiscard\tjob.py\tcandidate row\t/tmp/candidate",
        ],
    )

    payload = _run_guard(tmp_path, env={"AUTOFL_MAX_CANDIDATES": "1"})

    assert payload["decision"] == "continue"
    assert payload["reason"] == "continue"
    assert payload["candidate_cap"] is None
    assert payload["candidate_cap_source"] == "uncapped"
    assert payload["final_response_allowed"] is False


def test_campaign_guard_reports_best_score_for_minimization(tmp_path):
    _write_results(
        tmp_path / "results.tsv",
        [
            "abc\t0.84\t10\t--name baseline\tkeep\tjob.py\tbaseline\t/tmp/baseline",
            "def\t0.42\t20\t--name candidate\tkeep\tjob.py\tcandidate row\t/tmp/candidate",
        ],
    )

    payload = _run_guard(tmp_path, "--mode", "min")

    assert payload["best_score"] == 0.42
