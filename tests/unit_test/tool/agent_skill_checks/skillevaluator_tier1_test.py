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

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Full scanner-backed checks need a separate CI path with scanner dependencies and tools such as gitleaks.
TIER1_VALIDATE_CHECKS = "schema,pii,license,unicode,quality,lint"
COMMAND_TIMEOUT = 30
KEY_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_SECRET_ACCESS_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "SKILL_EVAL_EMBEDDING_API_KEY",
    "SKILL_EVAL_LLM_API_KEY",
    "SKILL_EVAL_LLM_PROVIDER",
)


@pytest.mark.parametrize("command_name", ["validate", "quality-check", "pii-scan", "lint-scripts"])
def test_skillevaluator_tier1_keyless_commands(command_name, tmp_path):
    if sys.version_info < (3, 12) or sys.version_info >= (3, 14):
        pytest.skip("skillevaluator supports Python 3.12 and 3.13")

    skillevaluator = shutil.which("skillevaluator")
    if not skillevaluator:
        pytest.skip("skillevaluator is optional; install .[skill_eval] or put the CLI on PATH to run this check")

    skill_dir = _write_sample_skill(tmp_path)
    reports_dir = tmp_path / "reports"
    env = _keyless_env()
    command = _tier1_command(command_name, skill_dir, reports_dir)

    completed = subprocess.run(
        [skillevaluator, *command],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, (
        f"skillevaluator {' '.join(command)} failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _keyless_env():
    env = os.environ.copy()
    env["SKILLEVALUATOR_TELEMETRY_ENABLED"] = "false"
    for name in KEY_ENV_NAMES:
        env.pop(name, None)
    return env


def _tier1_command(command_name, skill_dir, reports_dir):
    common_args = [str(skill_dir), "-r", "cli", "-o", str(reports_dir)]
    if command_name == "validate":
        return [
            "validate",
            str(skill_dir),
            "--checks",
            TIER1_VALIDATE_CHECKS,
            "--no-llm",
            "--no-dedup",
            "-r",
            "cli",
            "-o",
            str(reports_dir),
        ]

    # Standalone Tier 1 commands are deterministic: pii-scan only uses LLMs with --llm-verify, and the others
    # do not expose LLM flags.
    return [command_name, *common_args]


def _write_sample_skill(tmp_path):
    skill_dir = tmp_path / "sample-tier-one"
    skill_dir.mkdir()
    skill_dir.joinpath("SKILL.md").write_text(
        """---
name: sample-tier-one
description: "Use when validating a harmless sample skill for deterministic SkillEvaluator Tier 1 command coverage."
license: Apache-2.0
metadata:
  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"
  tags:
    - validation
---

# Sample Tier One

## Purpose

Use this sample only for deterministic SkillEvaluator command coverage in NVFLARE tests.

## Use When

Use when a test needs a small skill directory with no external side effects.

## Do Not Use When

Do not use this sample as runtime guidance for an agent.

## Instructions

1. Read the request.
2. Report that this is a fixture.
3. Stop without running tools.

## Examples

User asks for fixture validation; answer that the fixture is valid.

## Limitations

This fixture does not perform real NVFLARE work.

## Troubleshooting

If validation fails, inspect the SkillEvaluator report for the failing check.
""",
        encoding="utf-8",
    )
    return skill_dir
