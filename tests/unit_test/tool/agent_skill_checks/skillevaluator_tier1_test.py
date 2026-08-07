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
SKILL_DIRS = tuple(sorted(path.parent for path in (REPO_ROOT / "skills").glob("*/SKILL.md")))

# The keyless deterministic check set. The full Tier 1 security scan additionally needs
# the `security` extra plus external Semgrep, SkillSpector, and Gitleaks binaries, and
# LLM-backed checks need a provider key, so neither belongs in the unit-test lane.
TIER1_VALIDATE_CHECKS = "schema,pii,license,unicode,quality,lint"

# Standalone Tier 1 commands that are deterministic without a provider key: pii-scan only
# consults an LLM under --llm-verify, and the other two expose no LLM flags at all.
TIER1_STANDALONE_COMMANDS = ("quality-check", "pii-scan", "lint-scripts")

BLOCKING_SEVERITIES = ("critical", "high")
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
    assert skill_dir.name in scanned, f"{skill_dir.name} missing from the report: {sorted(scanned)}"
    assert scanned[skill_dir.name]["passed"] is True, _failure_message(command, completed)
    # Advisory MEDIUM/LOW quality findings are expected and do not gate the lane;
    # a CRITICAL or HIGH finding on a shipped skill does.
    for severity in BLOCKING_SEVERITIES:
        assert severity_counts[severity] == 0, (
            f"{skill_dir.name} has {severity_counts[severity]} {severity.upper()} Tier 1 finding(s)\n"
            f"{_failure_message(command, completed)}"
        )


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
    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        pytest.skip("skillevaluator is included in NVFlare development dependencies on Python 3.12 and 3.13")

    skillevaluator = shutil.which("skillevaluator")
    assert skillevaluator, "skillevaluator must be installed by NVFlare's Python 3.12/3.13 development dependencies"
    return skillevaluator


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
