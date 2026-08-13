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
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

COMMAND_TIMEOUT = 120
REPO_ROOT = Path(__file__).resolve().parents[4]
TIER1_SELECTION_SCRIPT = REPO_ROOT / "ci" / "should_run_skill_tier1.sh"
SKILL_DIRS = tuple(sorted(path.parent for path in (REPO_ROOT / "skills").glob("*/SKILL.md")))
# Fail closed: pytest silently skips a test parametrized over an empty sequence and still
# exits 0, so a moved or renamed skills layout would turn this gate green without ever
# scanning a skill.
assert SKILL_DIRS, f"no bundled skills discovered under {REPO_ROOT / 'skills'}"

# The deterministic Tier 1 gate. ``security`` invokes SkillSpector, and
# ``code-integrity`` invokes Bandit, Semgrep, and Gitleaks. LLM-backed checks
# remain disabled so this suite never requires a provider key.
TIER1_VALIDATE_CHECKS = "schema,security,pii,license,code-integrity,unicode,quality,lint"

# Standalone Tier 1 commands that are deterministic without a provider key: pii-scan only
# consults an LLM under --llm-verify, and the other two expose no LLM flags at all.
TIER1_STANDALONE_COMMANDS = ("quality-check", "pii-scan", "lint-scripts")

BLOCKING_SEVERITIES = ("critical", "high")

# Quality findings are emitted as advisory MEDIUM/LOW and never reach a blocking severity,
# so `quality-check` exits 0 even for a skill riddled with them: a skill with 12 findings
# still passes. The graded score is the only signal that actually moves, so gate on it.
# The bundled skills currently score 90.2-98.2 (grade A) and a deliberately poor skill
# scores 72.8 (grade C); 85.0 leaves headroom below today's worst while still catching a
# real regression.
QUALITY_SCORE_FLOOR = 85.0
KEY_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_SECRET_ACCESS_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "SKILL_EVAL_EMBEDDING_API_KEY",
    "SKILL_EVAL_LLM_API_KEY",
    "SKILL_EVAL_LLM_PROVIDER",
)


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda path: path.name)
def test_skillevaluator_tier1_validate_of_bundled_skill(skill_dir, tmp_path):
    """Every bundled skill passes the keyless Tier 1 validation gate."""
    skillevaluator = _require_skillevaluator()

    reports_dir = tmp_path / "reports"
    command = [
        skillevaluator,
        "validate",
        str(skill_dir),
        "--checks",
        TIER1_VALIDATE_CHECKS,
        "--no-llm",
        "--no-dedup",
        "-r",
        "json",
        "-o",
        str(reports_dir),
    ]

    completed = _run(command)
    assert completed.returncode == 0, _failure_message(command, completed)

    report = _load_report(reports_dir, completed)
    severity_counts = report["severity_counts"]
    scanned = {entry["name"]: entry for entry in report["skills"]}

    assert report["overall_passed"] is True, _failure_message(command, completed)
    assert report["total_errors"] == 0, _failure_message(command, completed)
    assert not report.get("incomplete_scans"), (
        f"{skill_dir.name} Tier 1 scan was incomplete: {report['incomplete_scans']}\n"
        f"{_failure_message(command, completed)}"
    )
    assert skill_dir.name in scanned, f"{skill_dir.name} missing from the report: {sorted(scanned)}"
    assert scanned[skill_dir.name]["passed"] is True, _failure_message(command, completed)
    # Advisory MEDIUM/LOW quality findings are expected and do not gate the lane;
    # a CRITICAL or HIGH finding on a shipped skill does.
    for severity in BLOCKING_SEVERITIES:
        assert severity_counts[severity] == 0, (
            f"{skill_dir.name} has {severity_counts[severity]} {severity.upper()} Tier 1 finding(s)\n"
            f"{_failure_message(command, completed)}"
        )

    quality = _quality_entry(report, skill_dir.name)
    assert quality["overall_score"] >= QUALITY_SCORE_FLOOR, (
        f"{skill_dir.name} quality score {quality['overall_score']} (grade {quality['grade']}) "
        f"fell below the {QUALITY_SCORE_FLOOR} floor\n{_failure_message(command, completed)}"
    )


def _quality_entry(report, skill_name):
    summary = report.get("quality_summary") or []
    entries = [entry for entry in summary if entry.get("skill_name") == skill_name]
    assert entries, f"no quality_summary entry for {skill_name}; got {[e.get('skill_name') for e in summary]}"
    return entries[0]


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda path: path.name)
@pytest.mark.parametrize("command_name", TIER1_STANDALONE_COMMANDS)
def test_skillevaluator_tier1_standalone_command_of_bundled_skill(command_name, skill_dir, tmp_path):
    """The standalone keyless Tier 1 commands succeed against every bundled skill."""
    skillevaluator = _require_skillevaluator()

    reports_dir = tmp_path / "reports"
    command = [skillevaluator, command_name, str(skill_dir), "-r", "cli", "-o", str(reports_dir)]

    completed = _run(command)
    assert completed.returncode == 0, _failure_message(command, completed)


def _require_skillevaluator():
    # CI sets this from the changed-path selector. Developers can opt in on any
    # supported platform by setting NVFLARE_RUN_SKILL_TIER1=true.
    if os.environ.get("NVFLARE_RUN_SKILL_TIER1") != "true":
        pytest.skip("Tier 1 skill security scan was not selected for this change")

    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        pytest.skip("skillevaluator supports Python 3.12 and 3.13")

    # Not a pip-installed development dependency: it cannot be resolved alongside the
    # dev extras (click>=8.3.3 vs NVFlare's click==8.1.7 below 3.14), so CI installs it
    # into an isolated virtual environment and exposes only the CLI on PATH. Skipping
    # when absent keeps local runs working; CI gates because its install step is not
    # allowed to fail.
    skillevaluator = shutil.which("skillevaluator")
    if not skillevaluator:
        if os.environ.get("NVFLARE_SKILL_TIER1_REQUIRED") == "true":
            pytest.fail("skillevaluator CLI is required for the selected Tier 1 skill security scan")
        pytest.skip("skillevaluator CLI not on PATH; install .[skill_eval] in a separate environment to run this check")
    return skillevaluator


@pytest.mark.parametrize(
    ("changed_paths", "expected"),
    [
        (["skills/nvflare-fed-stats/SKILL.md"], "true"),
        (["tests/unit_test/tool/agent_skill_checks/skillevaluator_tier1_test.py"], "true"),
        (["dev_tools/agent/skills/checks/lints.py"], "true"),
        ([".github/workflows/premerge.yml"], "true"),
        (["ci/should_run_skill_tier1.sh"], "true"),
        (["setup.cfg"], "true"),
        (["nvflare/apis/fl_context.py", "docs/user_guide/index.rst"], "false"),
    ],
)
def test_tier1_selector(changed_paths, expected):
    completed = subprocess.run(
        ["bash", str(TIER1_SELECTION_SCRIPT)],
        cwd=REPO_ROOT,
        input="\n".join(changed_paths),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected


def _run(command):
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
        env=_keyless_env(),
        check=False,
    )


def _load_report(reports_dir, completed):
    # The JSON reporter names its output with a run timestamp, so glob rather than guess.
    reports = sorted(reports_dir.glob("*.json"))
    assert reports, f"skillevaluator did not create a JSON report in {reports_dir}\nstdout:\n{completed.stdout}"
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def _failure_message(command, completed):
    return (
        f"{' '.join(command)} failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _keyless_env():
    env = os.environ.copy()
    env["SKILLEVALUATOR_TELEMETRY_ENABLED"] = "false"
    for name in KEY_ENV_NAMES:
        env.pop(name, None)
    return env
