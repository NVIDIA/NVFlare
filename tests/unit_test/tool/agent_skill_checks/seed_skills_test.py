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

import sys
from pathlib import Path

CHECKS_PARENT = Path(__file__).resolve().parents[4] / "dev_tools" / "agent" / "skills"
sys.path.insert(0, str(CHECKS_PARENT))

from checks.lints import run_v1_lints  # noqa: E402


def test_seed_skills_pass_v1_admission_lints():
    repo_root = Path(__file__).resolve().parents[4]

    result = run_v1_lints(repo_root / "skills")

    assert result["status"] == "ok"
    assert result["summary"]["skill_count"] >= 2
    assert result["findings"] == []


def test_diagnose_job_catalog_pins_recovery_categories():
    repo_root = Path(__file__).resolve().parents[4]
    skill_root = repo_root / "skills" / "nvflare-diagnose-job"
    skill_text = skill_root.joinpath("SKILL.md").read_text(encoding="utf-8")
    catalog_text = skill_root.joinpath("references/failure-patterns.md").read_text(encoding="utf-8")
    normalized_catalog = " ".join(catalog_text.split())
    rows = _failure_pattern_rows(catalog_text)

    assert "copying the category from the matched" in skill_text
    assert "Do not infer or override the category" in skill_text
    assert "copy the `Recovery Category` value from that same row exactly" in normalized_catalog

    round_timeout = rows["ROUND_TIMEOUT"]
    assert round_timeout["Recovery Category"] == "`ENVIRONMENT_FAILURE`"
    assert "timeout configuration" not in round_timeout["Next Action"]
    assert "temporary mitigation, not the primary fix" in round_timeout["Next Action"]

    partial_logs = rows["PARTIAL_LOG_VISIBILITY"]
    assert partial_logs["Recovery Category"] == "`UNKNOWN`"
    assert "before assigning root cause" in partial_logs["Next Action"]
    assert "do not classify the log-access problem as the job failure cause" in partial_logs["Next Action"]


def _failure_pattern_rows(catalog_text):
    rows = {}
    headers = []
    for line in catalog_text.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == "Pattern":
            headers = cells
            continue
        if not headers or set(cells[0]) <= {"-", " "}:
            continue
        rows[cells[0].strip("`")] = dict(zip(headers, cells))
    return rows
