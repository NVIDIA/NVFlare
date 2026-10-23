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
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / "ci" / "should_run_full_ci.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "premerge.yml"


@pytest.mark.parametrize(
    ("changed_paths", "expected"),
    [
        (
            [
                "README.md",
                "docs/quickstart.rst",
                "docs/resources/federated_learning_overview.png",
                "examples/tutorials/README.md",
            ],
            "false",
        ),
        (["docs/llms.txt.in", "docs/_static/custom.css"], "false"),
        (["CONTRIBUTING.md", "CITATION.cff", "LICENSE"], "false"),
        (["docs/resources/step_ca_admin.tpl"], "true"),
        (["docs/conf.py"], "true"),
        (["nvflare/apis/fl_context.py"], "true"),
        (["pyproject.toml"], "true"),
        ([".github/workflows/premerge.yml"], "true"),
        (["ci/should_run_full_ci.sh"], "true"),
        (["skills/nvflare-fed-stats/SKILL.md"], "true"),
        (["tests/unit_test/tool/agent_skill_checks/fixtures/valid/nvflare-example-skill/SKILL.md"], "true"),
        (["tests/fixtures/input.rst"], "true"),
        (["docs/quickstart.rst", "tests/unit_test/apis/fl_context_test.py"], "true"),
        ([], "true"),
    ],
)
def test_full_ci_selector(changed_paths, expected):
    completed = subprocess.run(
        ["bash", str(SELECTOR)],
        cwd=REPO_ROOT,
        input="\n".join(changed_paths),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected


@pytest.fixture
def selection_repository(tmp_path):
    """Run the workflow shell against real revisions in an isolated repository."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=CI Selector Test",
                "-c",
                "user.email=ci-selector@example.com",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=repo,
            text=True,
        ).strip()

    git("init", "-q")
    (repo / "ci").mkdir()
    (repo / "ci" / SELECTOR.name).write_text(SELECTOR.read_text())
    git("add", ".")
    git("commit", "-qm", "Base")
    base_sha = git("rev-parse", "HEAD")

    def run_selection(changed_path="README.md", selector=None, missing_selector=False, **event):
        changed_file = repo / changed_path
        changed_file.parent.mkdir(parents=True, exist_ok=True)
        changed_file.write_text("changed\n")
        git("add", ".")
        git("commit", "-qm", "Change")
        if missing_selector:
            (repo / "ci" / SELECTOR.name).unlink()
        elif selector is not None:
            # Drain the input so invalid-output cases cannot pass accidentally
            # through the pipeline-error fallback after git receives SIGPIPE.
            (repo / "ci" / SELECTOR.name).write_text("cat >/dev/null\n" + selector)

        output = tmp_path / "github-output"
        env = {
            **os.environ,
            "EVENT_NAME": "pull_request",
            "BASE_SHA": base_sha,
            "HEAD_SHA": git("rev-parse", "HEAD"),
            "GITHUB_OUTPUT": str(output),
            **event,
        }
        steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["full-ci-selection"]["steps"]
        step = next(step for step in steps if step.get("id") == "select")
        # Match GitHub's explicit bash shell, including pipeline failure handling.
        assert step.get("shell") == "bash"
        completed = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        return output.read_text(), completed.stdout

    return run_selection


@pytest.mark.parametrize(
    ("changed_path", "expected"),
    [
        ("README.md", "false"),
        ("nvflare/example.py", "true"),
        ("docs/resources/step_ca_admin.tpl", "true"),
        ("tests/unit_test/tool/agent_skill_checks/fixtures/valid/nvflare-example-skill/SKILL.md", "true"),
    ],
)
@pytest.mark.parametrize("event_name", ["pull_request", "merge_group"])
def test_workflow_selects_full_ci(selection_repository, changed_path, expected, event_name):
    output, _ = selection_repository(changed_path=changed_path, EVENT_NAME=event_name)
    assert output == f"run-full-ci={expected}\n"


@pytest.mark.parametrize(
    ("selector", "warning"),
    [
        (None, "CI selection failed"),
        ("exit 1\n", "CI selection failed"),
        ("echo false; exit 1\n", "CI selection failed"),
        ("", "invalid result"),
        ("echo invalid\n", "invalid result"),
        ("printf 'false\\ntrue\\n'\n", "invalid result"),
    ],
)
def test_workflow_runs_full_ci_when_selector_fails(selection_repository, selector, warning):
    output, log = selection_repository(selector=selector, missing_selector=selector is None)
    assert output == "run-full-ci=true\n"
    assert "::warning::" in log and warning in log


def test_workflow_runs_full_ci_when_diff_fails(selection_repository):
    # A successful selector must not hide a failed upstream git command.
    output, log = selection_repository(BASE_SHA="missing-revision", selector="echo false\n")
    assert output == "run-full-ci=true\n"
    assert "::warning::" in log


@pytest.mark.parametrize(
    "event",
    [
        {"EVENT_NAME": "push"},
        {"EVENT_NAME": "workflow_dispatch"},
        {"BASE_SHA": ""},
        {"BASE_SHA": "0" * 40},
        {"HEAD_SHA": ""},
    ],
)
def test_workflow_requires_full_ci_without_a_supported_change_set(selection_repository, event):
    output, _ = selection_repository(selector="echo false\n", **event)
    assert output == "run-full-ci=true\n"


@pytest.mark.parametrize("job_name", ["style-check", "unit-tests", "wheel-build"])
@pytest.mark.parametrize(
    ("selection_result", "run_full_ci", "skill_result", "succeeds"),
    [
        ("success", "true", "success", True),
        ("success", "false", "success", True),
        ("failure", "false", "success", False),
        ("skipped", "", "success", False),
        ("success", "", "success", False),
        ("success", "invalid", "success", False),
        ("success", "true", "failure", False),
    ],
)
def test_required_jobs_do_not_pass_failed_selection(job_name, selection_result, run_full_ci, skill_result, succeeds):
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"][job_name]
    # A status function is required to start jobs after a failed dependency.
    assert job["if"] == "${{ !cancelled() }}"
    guard = job["steps"][0]
    assert guard["name"] == "Require successful CI selection"
    assert "if" not in guard
    assert not guard.get("continue-on-error", False)
    assert guard["env"]["SELECTION_RESULT"] == "${{ needs.full-ci-selection.result }}"
    assert guard["env"]["RUN_FULL_CI"] == "${{ needs.full-ci-selection.outputs.run-full-ci }}"
    if job_name == "unit-tests":
        assert guard["env"]["SKILL_SELECTION_RESULT"] == "${{ needs.skill-validation-selection.result }}"
    elif skill_result != "success":
        succeeds = True  # These jobs do not depend on skill selection.

    completed = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", guard["run"]],
        env={
            **os.environ,
            "SELECTION_RESULT": selection_result,
            "RUN_FULL_CI": run_full_ci,
            "SKILL_SELECTION_RESULT": skill_result,
        },
        capture_output=True,
        text=True,
    )
    assert (completed.returncode == 0) == succeeds
    if not succeeds:
        assert "::error::" in completed.stdout
