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

import pytest

CHECKS_PARENT = Path(__file__).resolve().parents[4] / "dev_tools" / "agent" / "skills"
sys.path.insert(0, str(CHECKS_PARENT))

from checks import lints as lints_module  # noqa: E402
from checks.lints import (  # noqa: E402
    MAX_SKILL_TEXT_FILE_BYTES,
    _run_v1_lints_with_records,
    run_v1_lints,
    validate_skills,
)

LINT_SKILL_FRONTMATTER = "skill-frontmatter-lint"
LINT_SKILL_MD_SIZE = "skill-md-size-lint"
LINT_SKILL_TRIGGER = "skill-trigger-lint"
LINT_SKILL_TRIGGER_OVERLAP = "skill-trigger-overlap-lint"
LINT_SKILL_GLOBAL_NEGATIVE = "skill-global-negative-lint"
LINT_SKILL_POLICY_COVERAGE = "skill-policy-coverage-lint"
LINT_SKILL_PROCESS_METRIC = "skill-process-metric-lint"
LINT_SKILL_COMMAND_DRIFT = "skill-command-drift-lint"
LINT_SKILL_HELPER_SCRIPT = "skill-helper-script-lint"
LINT_SKILL_FIXTURE = "skill-fixture-lint"
LINT_SKILL_RUNTIME_BOUNDARY = "skill-runtime-boundary-lint"
REQUIRED_FINDING_FIELDS = {"id", "severity", "file", "message", "hint"}


def test_run_v1_lints_passes_complete_skill(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")

    result = run_v1_lints(tmp_path / "skills")

    assert result["status"] == "ok"
    assert result["findings"] == []
    assert result["summary"]["error_count"] == 0
    assert {"error", "warning", "info"}.isdisjoint(result["summary"])
    assert set(result["checks"]) == {
        LINT_SKILL_FRONTMATTER,
        LINT_SKILL_MD_SIZE,
        LINT_SKILL_TRIGGER,
        LINT_SKILL_TRIGGER_OVERLAP,
        LINT_SKILL_GLOBAL_NEGATIVE,
        LINT_SKILL_POLICY_COVERAGE,
        LINT_SKILL_PROCESS_METRIC,
        LINT_SKILL_COMMAND_DRIFT,
        LINT_SKILL_HELPER_SCRIPT,
        LINT_SKILL_FIXTURE,
        LINT_SKILL_RUNTIME_BOUNDARY,
    }


def test_run_v1_lints_reports_frontmatter_prefix(tmp_path):
    _write_skill(tmp_path / "skills", "example-skill")

    result = run_v1_lints(tmp_path / "skills")

    assert _has_finding(result, LINT_SKILL_FRONTMATTER, "skill-name-prefix-required")
    _assert_structured_findings(result)


def test_run_v1_lints_allows_internal_skill_without_nvflare_prefix(tmp_path):
    _write_skill(tmp_path / "skills", "example-skill", status="internal")

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_FRONTMATTER])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_run_v1_lints_accepts_public_category_frontmatter(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-category-skill", category="diagnosis")

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_FRONTMATTER])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_run_v1_lints_reports_skill_md_size(tmp_path):
    body = "\n".join(f"line {i}" for i in range(205))
    _write_skill(tmp_path / "skills", "nvflare-large-skill", body=body)

    result = run_v1_lints(tmp_path / "skills")

    assert _has_finding(result, LINT_SKILL_MD_SIZE, "skill-md-too-large")
    _assert_structured_findings(result)


def test_run_v1_lints_does_not_parse_oversized_skill_md(monkeypatch, tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-large-skill")
    with skill_dir.joinpath("SKILL.md").open("ab") as stream:
        stream.truncate(MAX_SKILL_TEXT_FILE_BYTES + 1)
    monkeypatch.setattr(
        lints_module,
        "_try_parse_frontmatter",
        lambda _path: (_ for _ in ()).throw(AssertionError("oversized SKILL.md should not be parsed")),
    )

    result, records = _run_v1_lints_with_records(tmp_path / "skills", checks=[LINT_SKILL_MD_SIZE])

    assert records[0].metadata == {}
    assert records[0].text == ""
    assert records[0].body == ""
    assert _has_finding(result, LINT_SKILL_MD_SIZE, "skill-md-too-large")


def test_load_evals_rejects_oversized_evals_json(tmp_path):
    evals_path = tmp_path / "evals.json"
    with evals_path.open("wb") as stream:
        stream.truncate(MAX_SKILL_TEXT_FILE_BYTES + 1)

    evals, error = lints_module._load_evals(evals_path)

    assert evals == []
    assert error == f"evals/evals.json exceeds size limit ({MAX_SKILL_TEXT_FILE_BYTES} bytes)"


def test_line_for_field_does_not_read_oversized_skill_md(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    with skill_file.open("wb") as stream:
        stream.truncate(MAX_SKILL_TEXT_FILE_BYTES + 1)

    assert lints_module._line_for_field(skill_file, "name") == 1


def test_run_v1_lints_reports_missing_trigger_evals(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-trigger-skill", evals={"evals": []})

    result = run_v1_lints(tmp_path / "skills")

    assert _has_finding(result, LINT_SKILL_TRIGGER, "skill-positive-trigger-eval-missing")
    assert _has_finding(result, LINT_SKILL_TRIGGER, "skill-adjacent-negative-eval-missing")
    assert _has_finding(result, LINT_SKILL_GLOBAL_NEGATIVE, "skill-global-negative-eval-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_reports_trigger_overlap_without_negative_boundary(tmp_path):
    evals_one = _default_evals("nvflare-convert-one", adjacent_negative=False)
    evals_two = _default_evals("nvflare-convert-two", adjacent_negative=False)
    _write_skill(
        tmp_path / "skills",
        "nvflare-convert-one",
        description="Convert PyTorch training code to FLARE.",
        body="Use when converting PyTorch training code.\n",
        evals=evals_one,
    )
    _write_skill(
        tmp_path / "skills",
        "nvflare-convert-two",
        description="Convert PyTorch training code to FLARE.",
        body="Use when converting PyTorch training code.\n",
        evals=evals_two,
    )

    result = run_v1_lints(tmp_path / "skills")

    assert _has_finding(result, LINT_SKILL_TRIGGER_OVERLAP, "skill-trigger-overlap")
    _assert_structured_findings(result)


def test_run_v1_lints_reports_non_convert_trigger_overlap_from_name_family(tmp_path):
    evals_one = _default_evals("nvflare-route-one", adjacent_negative=False)
    evals_two = _default_evals("nvflare-route-two", adjacent_negative=False)
    _write_skill(
        tmp_path / "skills",
        "nvflare-route-one",
        description="Route ambiguous FLARE project requests using inspect and readiness evidence.",
        body="Use when routing ambiguous FLARE project requests with inspect evidence.\n",
        evals=evals_one,
    )
    _write_skill(
        tmp_path / "skills",
        "nvflare-route-two",
        description="Route ambiguous FLARE project requests using inspect and readiness evidence.",
        body="Use when routing ambiguous FLARE project requests with inspect evidence.\n",
        evals=evals_two,
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_TRIGGER_OVERLAP])

    assert _has_finding(result, LINT_SKILL_TRIGGER_OVERLAP, "skill-trigger-overlap")
    _assert_structured_findings(result)


@pytest.mark.parametrize("lint_id", ["skill-catalog-category-lint", "agent-doc-crosslink-lint"])
def test_run_v1_lints_rejects_retired_design_doc_lints(tmp_path, lint_id):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")

    with pytest.raises(ValueError, match=lint_id):
        run_v1_lints(tmp_path / "skills", checks=[lint_id])


def test_run_v1_lints_reports_policy_without_behavior_ids(tmp_path):
    evals = _default_evals("nvflare-policy-skill", include_behavior_ids=False)
    _write_skill(
        tmp_path / "skills", "nvflare-policy-skill", body="The agent must validate before submit.\n", evals=evals
    )

    result = run_v1_lints(tmp_path / "skills")

    assert _has_finding(result, LINT_SKILL_POLICY_COVERAGE, "skill-policy-coverage-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_reports_unknown_nvflare_command(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-command-skill", body="Run `nvflare unknown --format json`.\n")

    result = run_v1_lints(tmp_path / "skills")

    assert _has_finding(result, LINT_SKILL_COMMAND_DRIFT, "skill-command-drift")
    finding = _finding(result, LINT_SKILL_COMMAND_DRIFT, "skill-command-drift")
    assert isinstance(finding["line"], int)
    _assert_structured_findings(result)


def test_run_v1_lints_reports_command_drift_before_unsafe_token(tmp_path):
    _write_skill(
        tmp_path / "skills",
        "nvflare-command-skill",
        body="Run `nvflare agent unknown $HOME/skills`.\n",
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_COMMAND_DRIFT])

    assert _has_finding(result, LINT_SKILL_COMMAND_DRIFT, "skill-command-drift")
    finding = _finding(result, LINT_SKILL_COMMAND_DRIFT, "skill-command-drift")
    assert "unknown nvflare agent command 'unknown'" in finding["message"]
    _assert_structured_findings(result)


def test_run_v1_lints_parses_quoted_nvflare_command_with_shlex(tmp_path):
    _write_skill(
        tmp_path / "skills",
        "nvflare-command-skill",
        body='Run `nvflare agent skills install --skill "nvflare-valid-skill" --target /tmp/skills`.\n',
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_COMMAND_DRIFT])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_run_v1_lints_skips_trigger_overlap_when_skill_count_exceeds_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFLARE_AGENT_MAX_TRIGGER_OVERLAP_SKILLS", "1")
    _write_skill(tmp_path / "skills", "nvflare-convert-one")
    _write_skill(tmp_path / "skills", "nvflare-convert-two")

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_TRIGGER_OVERLAP])

    assert result["status"] == "ok"
    assert result["findings"] == []
    assert result["skipped_checks"] == [
        {
            "id": LINT_SKILL_TRIGGER_OVERLAP,
            "reason": "group 'nvflare-convert' has 2 skills; limit is 1",
        }
    ]


def test_run_v1_lints_reports_helper_script_without_test(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-helper-skill")
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("helper.py").write_text("print('{}')\n", encoding="utf-8")

    result = run_v1_lints(tmp_path / "skills")

    assert _has_finding(result, LINT_SKILL_HELPER_SCRIPT, "skill-helper-tests-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_helper_script_ignores_symlink_loop(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-helper-skill")
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("helper.py").write_text("print('{}')\n", encoding="utf-8")
    _symlink_dir_or_skip(scripts_dir, scripts_dir / "loop")

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_HELPER_SCRIPT])

    assert _has_finding(result, LINT_SKILL_HELPER_SCRIPT, "skill-helper-tests-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_skips_oversized_helper_script_content_checks(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-helper-skill")
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    helper = scripts_dir / "helper.py"
    helper.write_text("promoted_to: nvflare agent helper\n", encoding="utf-8")
    with helper.open("ab") as stream:
        stream.truncate(MAX_SKILL_TEXT_FILE_BYTES + 1)
    tests_dir = skill_dir / "tests"
    tests_dir.mkdir()
    tests_dir.joinpath("helper_test.txt").write_text("helper test placeholder\n", encoding="utf-8")

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_HELPER_SCRIPT])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_run_v1_lints_reports_missing_fixture_file(tmp_path):
    evals = _default_evals("nvflare-fixture-skill")
    evals["evals"][0]["files"] = ["files/missing.py"]
    _write_skill(tmp_path / "skills", "nvflare-fixture-skill", evals=evals, write_fixture=False)

    result = run_v1_lints(tmp_path / "skills")

    assert _has_finding(result, LINT_SKILL_FIXTURE, "skill-fixture-file-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_fixture_file_check_ignores_symlink_loop(tmp_path):
    evals = _default_evals("nvflare-fixture-skill")
    evals["evals"][0]["files"] = ["files/input.py"]
    _write_skill(tmp_path / "skills", "nvflare-fixture-skill", evals=evals, write_fixture=False)
    files_dir = tmp_path / "dev_tools" / "agent" / "skill_evals" / "nvflare-fixture-skill" / "files"
    files_dir.mkdir()
    files_dir.joinpath("input.py").write_text("print('hello')\n", encoding="utf-8")
    _symlink_dir_or_skip(files_dir, files_dir / "loop")

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_FIXTURE])

    assert _has_finding(result, LINT_SKILL_FIXTURE, "skill-fixture-notes-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_does_not_require_fixtures_for_conceptual_file_mentions(tmp_path):
    evals = {
        "skill_name": "nvflare-fixture-skill",
        "evals": [
            {
                "id": "conceptual-file-guidance",
                "prompt": "Explain how to create a dataset file naming convention.",
                "expected_output": "A written explanation, not edited files.",
                "files": [],
                "assertions": ["Mentions file naming without creating artifacts."],
                "nvflare": {"expected_skill": "nvflare-fixture-skill"},
            }
        ],
        "nvflare": {"category": "conversion"},
    }
    _write_skill(tmp_path / "skills", "nvflare-fixture-skill", evals=evals, write_fixture=False)

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_FIXTURE])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_run_v1_lints_reference_text_scan_ignores_symlink_loop(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-command-skill")
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    references_dir.joinpath("guide.md").write_text("Run `nvflare unknown --format json`.\n", encoding="utf-8")
    _symlink_dir_or_skip(references_dir, references_dir / "loop")

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_COMMAND_DRIFT])

    assert _has_finding(result, LINT_SKILL_COMMAND_DRIFT, "skill-command-drift")
    _assert_structured_findings(result)


def test_run_v1_lints_rejects_fixture_paths_that_escape_skill_dir(tmp_path):
    evals = _default_evals("nvflare-fixture-skill")
    evals["evals"][0]["files"] = ["../outside.py"]
    _write_skill(tmp_path / "skills", "nvflare-fixture-skill", evals=evals)

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_FIXTURE])

    assert _has_finding(result, LINT_SKILL_FIXTURE, "skill-fixture-path-escape")
    _assert_structured_findings(result)


def test_run_v1_lints_supports_check_selection(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    _write_skill(tmp_path / "skills", "nvflare-other-skill", evals={"evals": []})

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_FRONTMATTER])

    assert result["status"] == "ok"
    assert result["checks"] == [LINT_SKILL_FRONTMATTER]
    assert result["summary"]["skill_count"] == 2


def test_run_v1_lints_skips_shared_reference_dirs(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    shared_dir = tmp_path / "skills" / "_shared"
    shared_dir.mkdir()
    shared_dir.joinpath("reference.md").write_text("shared guidance\n", encoding="utf-8")

    result = run_v1_lints(tmp_path / "skills")

    assert result["status"] == "ok"
    assert result["summary"]["skill_count"] == 1
    assert result["findings"] == []


def test_validate_skills_filters_summary_to_requested_skill(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    _write_skill(tmp_path / "skills", "nvflare-other-skill", evals={"evals": []})

    result = validate_skills(tmp_path / "skills", skill_name="nvflare-valid-skill")

    assert result["status"] == "ok"
    assert result["requested_skill"] == "nvflare-valid-skill"
    assert result["summary"]["skill_count"] == 1
    assert result["findings"] == []


def test_validate_skills_keeps_global_findings_for_requested_skill(tmp_path):
    result = validate_skills(tmp_path / "missing-skills", skill_name="nvflare-valid-skill")

    assert result["status"] == "failed"
    assert result["summary"]["error_count"] == 1
    finding = _finding(result, LINT_SKILL_FRONTMATTER, "skills-root-missing")
    assert finding["global"] is True
    assert "skill" not in finding


def test_validate_skills_uses_requested_size_limit_without_mutating_default(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")

    limited = validate_skills(
        tmp_path / "skills",
        skill_name="nvflare-valid-skill",
        max_skill_md_lines=2,
    )
    default = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_MD_SIZE])

    assert _has_finding(limited, LINT_SKILL_MD_SIZE, "skill-md-too-large")
    assert default["status"] == "ok"
    assert default["findings"] == []


def test_run_v1_lints_reports_design_doc_reference_in_runtime_content(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    references = skill_dir / "references"
    references.mkdir()
    references.joinpath("workflow.md").write_text(
        "Follow the operating model in docs/design/agent_skill_operating_model.md.\n",
        encoding="utf-8",
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    assert _has_finding(result, LINT_SKILL_RUNTIME_BOUNDARY, "skill-runtime-design-doc-ref")
    _assert_structured_findings(result)


def test_run_v1_lints_reports_evaluator_hook_in_skill_md(tmp_path):
    _write_skill(
        tmp_path / "skills",
        "nvflare-valid-skill",
        body=(
            "Use when converting PyTorch training code.\n"
            "Do not use for Kubernetes deployment.\n"
            "After a failure, add or update the eval case in evals/evals.json.\n"
        ),
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    assert _has_finding(result, LINT_SKILL_RUNTIME_BOUNDARY, "skill-runtime-evaluator-hook")


@pytest.mark.parametrize(
    "hook_line",
    [
        "Set NVFLARE_SKILL_EVAL=on before running.",
        "Enable with eval=on in the config.",
        "Run the conversion with the --eval flag.",
        "Export NVFLARE_EVAL_MODE=1 for the grader.",
        "Only relevant to the eval harness, not the runtime agent.",
    ],
)
def test_run_v1_lints_reports_evaluator_hook_spellings(tmp_path, hook_line):
    _write_skill(
        tmp_path / "skills",
        "nvflare-valid-skill",
        body=(
            "Use when converting PyTorch training code.\n" "Do not use for Kubernetes deployment.\n" f"{hook_line}\n"
        ),
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    assert _has_finding(result, LINT_SKILL_RUNTIME_BOUNDARY, "skill-runtime-evaluator-hook")


def test_run_v1_lints_reports_design_doc_reference_without_trailing_separator(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    references = skill_dir / "references"
    references.mkdir()
    references.joinpath("workflow.md").write_text(
        "See the docs/design directory for the operating-model policy.\n",
        encoding="utf-8",
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    assert _has_finding(result, LINT_SKILL_RUNTIME_BOUNDARY, "skill-runtime-design-doc-ref")


@pytest.mark.parametrize(
    "safe_line",
    [
        "Keep the source project's benchmark dataset loading.",
        "Report the DEFAULT_EVALUATION_METRIC from the source.",
        "Consult docs/designer notes if present.",
    ],
)
def test_run_v1_lints_does_not_flag_legitimate_runtime_words(tmp_path, safe_line):
    _write_skill(
        tmp_path / "skills",
        "nvflare-valid-skill",
        body=("Use when converting PyTorch training code.\nDo not use for Kubernetes deployment.\n" f"{safe_line}\n"),
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    assert result["findings"] == []


def test_run_v1_lints_scans_non_public_skill_runtime_content(tmp_path):
    _write_skill(
        tmp_path / "skills",
        "nvflare-draft-skill",
        status="draft",
        body=(
            "Use when converting PyTorch training code.\n"
            "Do not use for Kubernetes deployment.\n"
            "See docs/design/agent_skill_operating_model.md for the policy.\n"
        ),
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    assert _has_finding(result, LINT_SKILL_RUNTIME_BOUNDARY, "skill-runtime-design-doc-ref")


def test_run_v1_lints_scans_non_markdown_runtime_files(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    references = skill_dir / "references"
    references.mkdir()
    references.joinpath("helper.py").write_text(
        "# see docs/design/agent_skill_operating_model.md\nprint('ok')\n",
        encoding="utf-8",
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    assert _has_finding(result, LINT_SKILL_RUNTIME_BOUNDARY, "skill-runtime-design-doc-ref")


def test_run_v1_lints_reports_benchmark_instruction_in_scripts(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("helper.py").write_text(
        "# record results for the benchmark harness\nprint('ok')\n",
        encoding="utf-8",
    )
    skill_dir.joinpath("tests").mkdir()
    skill_dir.joinpath("tests", "helper_test.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    assert _has_finding(result, LINT_SKILL_RUNTIME_BOUNDARY, "skill-runtime-evaluator-hook")


def test_run_v1_lints_reports_design_doc_reference_in_shared_content(tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "nvflare-valid-skill")
    shared = root / "_shared"
    shared.mkdir()
    shared.joinpath("conversion-workflow.md").write_text(
        "See docs/design/agent_skill_operating_model.md for the policy.\n",
        encoding="utf-8",
    )

    result = run_v1_lints(root, checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    findings = [f for f in result["findings"] if f.get("code") == "skill-runtime-design-doc-ref"]
    assert findings and findings[0].get("global") is True


def test_run_v1_lints_allows_evaluation_language_in_runtime_content(tmp_path):
    _write_skill(
        tmp_path / "skills",
        "nvflare-valid-skill",
        body=(
            "Use when converting PyTorch training code.\n"
            "Do not use for Kubernetes deployment.\n"
            "Convert the evaluation loop and report metrics from trainer.validate().\n"
            "When the task is evaluate-only, select the FedEval recipe and evaluate the model.\n"
        ),
    )

    result = run_v1_lints(tmp_path / "skills", checks=[LINT_SKILL_RUNTIME_BOUNDARY])

    assert result["findings"] == []


def _write_skill(
    root,
    name,
    *,
    description="Convert PyTorch training code into a FLARE job.",
    body="Use when converting PyTorch training code.\nDo not use for Kubernetes deployment.\n",
    evals=None,
    category="conversion",
    write_fixture=True,
    status=None,
):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    status_line = f"status: {status}\n" if status else ""
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        'min_flare_version: "2.8.0"\n'
        "blast_radius: edits_files\n"
        f"category: {category}\n"
        f"{status_line}"
        "---\n"
        "\n"
        f"{body}",
        encoding="utf-8",
    )
    # Eval suites live outside the skill tree, one dir per skill name under the
    # default eval root beside the skills root (dev_tools/agent/skill_evals/).
    evals_dir = root.parent / "dev_tools" / "agent" / "skill_evals" / name
    evals_dir.mkdir(parents=True)
    if write_fixture:
        files_dir = evals_dir / "files"
        files_dir.mkdir()
        files_dir.joinpath("input.py").write_text("print('hello')\n", encoding="utf-8")
        files_dir.joinpath("README.md").write_text(
            "Source: synthetic fixture for deterministic agent skill lint tests.\n",
            encoding="utf-8",
        )
    evals_dir.joinpath("evals.json").write_text(
        json.dumps(evals if evals is not None else _default_evals(name, category=category), indent=2),
        encoding="utf-8",
    )
    return skill_dir


def _symlink_dir_or_skip(target, link):
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as e:
        pytest.skip(f"directory symlink is not available in this environment: {e}")


def _default_evals(name, *, category="conversion", adjacent_negative=True, include_behavior_ids=True):
    data = {
        "skill_name": name,
        "evals": [
            {
                "id": "positive",
                "prompt": "Convert PyTorch training code into a FLARE job.",
                "expected_output": "A validated FLARE job.",
                "files": ["files/input.py"],
                "assertions": ["Uses the expected skill."],
                "nvflare": {
                    "expected_skill": name,
                    "process_metrics": [
                        {
                            "id": "turns_to_acceptable",
                            "description": "number of turns before an acceptable result",
                        }
                    ],
                },
            },
            {
                "id": "global-negative",
                "prompt": "Deploy a React application.",
                "expected_output": "No FLARE skill should trigger.",
                "files": [],
                "assertions": ["No FLARE skill is selected."],
                "nvflare": {"expected_skill": "no_skill", "global_negative": True},
            },
        ],
    }
    if category is not None:
        data["nvflare"] = {"category": category}
    if adjacent_negative:
        data["evals"].append(
            {
                "id": "adjacent-negative",
                "prompt": "Deploy a FLARE startup kit to Kubernetes.",
                "expected_output": "A deployment skill should trigger.",
                "files": [],
                "assertions": ["Conversion skill is not selected."],
                "nvflare": {"expected_skill": "nvflare-deploy-k8s", "negative_for": name},
            }
        )
    if include_behavior_ids:
        data["evals"][0]["nvflare"].update(
            {
                "mandatory_behavior": [{"id": "inspect-first", "description": "runs inspect before editing"}],
                "prohibited_behavior": [{"id": "no-production-submit", "description": "does not submit"}],
                "optional_behavior": [{"id": "summarize", "description": "summarizes result"}],
            }
        )
    return data


def _has_finding(result, lint_id, code):
    return any(finding["id"] == lint_id and finding.get("code") == code for finding in result["findings"])


def _finding(result, lint_id, code):
    matches = [finding for finding in result["findings"] if finding["id"] == lint_id and finding.get("code") == code]
    assert matches, f"expected finding id={lint_id!r} code={code!r}; got {result['findings']!r}"
    return matches[0]


def _assert_structured_findings(result):
    assert result["findings"]
    for finding in result["findings"]:
        assert REQUIRED_FINDING_FIELDS.issubset(finding), finding
        assert finding["severity"] in {"error", "warning", "info"}
        assert finding["file"]
        assert finding["message"]
        assert finding["hint"]
        if "line" in finding:
            assert isinstance(finding["line"], int)
            assert finding["line"] > 0
    json.dumps(result["findings"])
