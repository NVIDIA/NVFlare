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

COMMAND_TIMEOUT = 30
REPO_ROOT = Path(__file__).resolve().parents[4]
SKILL_DIRS = tuple(sorted(path.parent for path in (REPO_ROOT / "skills").glob("*/SKILL.md")))
# Fail closed: pytest silently skips a test parametrized over an empty sequence and still
# exits 0, so a moved or renamed skills layout would turn this gate green without ever
# scanning a skill.
assert SKILL_DIRS, f"no bundled skills discovered under {REPO_ROOT / 'skills'}"
BLOCKING_SEVERITIES = {"HIGH", "CRITICAL"}
KEY_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_PROXY_API_KEY",
    "ANTHROPIC_PROXY_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_SECRET_ACCESS_KEY",
    "GOOGLE_API_KEY",
    "LANGSMITH_API_KEY",
    "NVIDIA_INFERENCE_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "SKILLSPECTOR_MODEL",
    "SKILLSPECTOR_PROVIDER",
)


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda path: path.name)
def test_skillspector_keyless_static_scan_of_bundled_skill(skill_dir, tmp_path):
    if sys.version_info[:2] != (3, 14):
        pytest.skip("skillspector is included in NVFlare development dependencies on Python 3.14")

    skillspector = shutil.which("skillspector")
    assert skillspector, "skillspector must be installed by NVFlare's Python 3.14 development dependencies"

    report_path = tmp_path / f"{skill_dir.name}-skillspector-report.json"
    command = [
        skillspector,
        "scan",
        str(skill_dir),
        "--format",
        "json",
        "--no-llm",
        "--output",
        str(report_path),
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
        env=_keyless_env(),
        check=False,
    )

    assert report_path.is_file(), f"skillspector did not create the JSON report\nstdout:\n{completed.stdout}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    risk_assessment = report["risk_assessment"]
    risk_score = risk_assessment["score"]
    issues = report["issues"]

    assert report["skill"]["name"] == skill_dir.name
    assert isinstance(risk_score, (int, float)) and not isinstance(risk_score, bool)
    assert 0 <= risk_score <= 100
    assert isinstance(issues, list)

    # Agent Skills evaluation suites are self-contained under evals/, including
    # adversarial fixtures used to verify prompt-injection resistance. Scan the
    # complete installed skill tree, but distinguish those declared fixtures from
    # runtime guidance. A HIGH/CRITICAL issue anywhere other than evals/files/
    # blocks the skill. Fixture findings remain visible in the JSON report and
    # may make SkillSpector return 1 because its aggregate risk score is high.
    blocking_runtime_issues = [
        issue for issue in issues if issue["severity"] in BLOCKING_SEVERITIES and not _is_eval_fixture(issue)
    ]
    assert not blocking_runtime_issues
    assert completed.returncode in {0, 1}, (
        f"{' '.join(command)} failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    if completed.returncode == 1:
        assert any(issue["severity"] in BLOCKING_SEVERITIES for issue in issues), (
            "SkillSpector returned a risk failure without a blocking finding\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _is_eval_fixture(issue):
    location = issue.get("location")
    if not isinstance(location, dict):
        return False
    file_path = location.get("file")
    return isinstance(file_path, str) and file_path.replace("\\", "/").startswith("evals/files/")


def _keyless_env():
    env = os.environ.copy()
    env["LANGCHAIN_TRACING_V2"] = "false"
    env["LANGSMITH_TRACING"] = "false"
    for name in KEY_ENV_NAMES:
        env.pop(name, None)
    return env
