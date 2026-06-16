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
import sys
from pathlib import Path

CHECKS_PARENT = Path(__file__).resolve().parents[4] / "dev_tools" / "agent" / "skills"
sys.path.insert(0, str(CHECKS_PARENT))

from checks import milestone8_checkpoint  # noqa: E402


def test_milestone8_checkpoint_marks_missing_benchmark_evidence_incomplete():
    result = milestone8_checkpoint.run_milestone8_checkpoint(_repo_root())

    assert result["status"] == "incomplete"
    checks = {check["id"]: check for check in result["checks"]}
    assert checks["skill_admission_lints"]["status"] == "ok"
    assert checks["inspect_routing"]["status"] == "ok"
    assert checks["install_list"]["status"] == "ok"
    assert checks["packaging"]["status"] == "ok"
    assert checks["benchmark_evidence"]["status"] == "manual_required"
    assert checks["benchmark_evidence"]["data"]["required_pairs"] == [
        {"agent": "codex", "skill": "nvflare-convert-pytorch"},
        {"agent": "codex", "skill": "nvflare-convert-lightning"},
        {"agent": "claude", "skill": "nvflare-convert-pytorch"},
        {"agent": "claude", "skill": "nvflare-convert-lightning"},
    ]


def test_milestone8_checkpoint_accepts_complete_benchmark_evidence(tmp_path):
    evidence_path = tmp_path / "stage5-evidence.json"
    evidence_path.write_text(json.dumps(_complete_benchmark_evidence()), encoding="utf-8")

    result = milestone8_checkpoint.run_milestone8_checkpoint(_repo_root(), benchmark_evidence=evidence_path)

    assert result["status"] == "ok"
    assert result["passed"] is True
    checks = {check["id"]: check for check in result["checks"]}
    assert checks["benchmark_evidence"]["status"] == "ok"


def test_milestone8_checkpoint_reports_missing_benchmark_pairs():
    payload = _complete_benchmark_evidence()
    payload["runs"] = payload["runs"][:-1]

    findings = milestone8_checkpoint._validate_benchmark_evidence_payload(payload)

    assert "missing benchmark run for agent=claude, skill=nvflare-convert-lightning" in findings


def _complete_benchmark_evidence():
    runs = []
    for agent in ("codex", "claude"):
        for skill in ("nvflare-convert-pytorch", "nvflare-convert-lightning"):
            runs.append(
                {
                    "agent": agent,
                    "skill": skill,
                    "correctness": "recorded",
                    "runtime_seconds": 1,
                    "dependency_behavior": "recorded",
                    "generated_structure": "recorded",
                    "token_usage": {"total_tokens": 1},
                    "metric_evidence": "recorded",
                    "artifact_location": "tmp/stage5/example",
                }
            )
    return {"schema_version": "1", "runs": runs}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
