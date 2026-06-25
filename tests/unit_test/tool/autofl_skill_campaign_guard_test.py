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

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

RESULT_FIELDS = [
    "status",
    "name",
    "score",
    "runtime_seconds",
    "changed_files",
    "diff_summary",
    "run_command",
    "artifacts",
    "failure_reason",
]


def _load_guard():
    repo_root = Path(__file__).parents[3]
    guard_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "campaign_guard.py"
    spec = importlib.util.spec_from_file_location("nvflare_autofl_skill_campaign_guard_test", guard_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_results(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(status, name, score="", diff_summary="candidate", run_command="python job.py"):
    return {
        "status": status,
        "name": name,
        "score": score,
        "runtime_seconds": "10.0",
        "changed_files": "none",
        "diff_summary": diff_summary,
        "run_command": run_command,
        "artifacts": "/tmp/run",
        "failure_reason": "",
    }


def test_guard_continues_uncapped_before_plateau():
    guard = _load_guard()
    rows = [
        _row("baseline", "baseline", "0.85"),
        _row("discard", "batch_size_64", "0.851"),
        _row("discard", "lr_0p01", "0.850"),
    ]

    state = guard.guard_state_for_rows(rows, plateau_threshold=4)

    assert state["schema_version"] == "nvflare.autofl.campaign_state.v1"
    assert state["decision"] == "continue"
    assert state["next_action"] == "launch_next_candidate"
    assert state["final_response_allowed"] is False
    assert state["candidate_cap"] is None
    assert state["candidate_attempts"] == 2
    assert state["best_score"] == 0.851


def test_guard_routes_plateau_to_literature_without_finalizing():
    guard = _load_guard()
    rows = [_row("baseline", "baseline", "0.85")]
    rows.extend(_row("discard", f"candidate_{idx}", "0.840") for idx in range(4))

    state = guard.guard_state_for_rows(rows, plateau_threshold=4)

    assert state["decision"] == "continue"
    assert state["reason"] == "plateau_literature"
    assert state["next_action"] == "run_literature_loop"
    assert state["final_response_allowed"] is False
    assert state["best_score"] == 0.85
    assert "Do not produce a final answer" in state["agent_instruction"]


def test_guard_literature_event_resets_plateau_clock():
    guard = _load_guard()
    rows = [_row("baseline", "baseline", "0.85")]
    rows.extend(_row("discard", f"before_lit_{idx}", "0.840") for idx in range(4))
    rows.append(_row("literature", "literature_review", "", diff_summary="literature review"))
    rows.extend(_row("discard", f"after_lit_{idx}", "0.841") for idx in range(2))

    state = guard.guard_state_for_rows(rows, plateau_threshold=4)

    assert state["reason"] == "continue"
    assert state["next_action"] == "launch_next_candidate"
    assert state["plateau"]["last_literature_event_index"] == 5
    assert state["plateau"]["scored_since_reset"] == 2


def test_guard_counts_candidate_with_baseline_in_description():
    guard = _load_guard()
    rows = [
        _row("baseline", "baseline", "0.85"),
        _row("discard", "weighted_rerun", "0.8503", diff_summary="weighted baseline escalated rerun"),
    ]

    state = guard.guard_state_for_rows(rows, max_candidates=1)

    assert state["candidate_attempts"] == 1
    assert state["decision"] == "stop"
    assert state["reason"] == "candidate_cap_exhausted"


def test_guard_cli_writes_campaign_state_json(tmp_path):
    repo_root = Path(__file__).parents[3]
    guard_path = repo_root / "skills" / "nvflare-autofl" / "scripts" / "campaign_guard.py"
    results_path = tmp_path / "results.tsv"
    state_path = tmp_path / "state.json"
    _write_results(
        results_path,
        [
            _row("baseline", "baseline", "0.85"),
            _row("discard", "candidate_1", "0.840"),
            _row("discard", "candidate_2", "0.841"),
        ],
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(guard_path),
            str(results_path),
            "--state",
            str(state_path),
            "--plateau-threshold",
            "2",
            "--format",
            "json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert json.loads(state_path.read_text(encoding="utf-8")) == payload
    assert payload["next_action"] == "run_literature_loop"
