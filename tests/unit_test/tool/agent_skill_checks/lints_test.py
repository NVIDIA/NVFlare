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

import pytest

from nvflare.tool.agent_skill_checks import lints as lints_module
from nvflare.tool.agent_skill_checks.lints import (
    MAX_SKILL_TEXT_FILE_BYTES,
    V1_LINT_IDS,
    _parse_conversion_table,
    _parse_product_catalog,
    _run_v1_lints_with_records,
    run_v1_lints,
    validate_skills,
)

LINT_SKILL_FRONTMATTER = "skill-frontmatter-lint"
LINT_SKILL_MD_SIZE = "skill-md-size-lint"
LINT_SKILL_TRIGGER = "skill-trigger-lint"
LINT_SKILL_TRIGGER_OVERLAP = "skill-trigger-overlap-lint"
LINT_SKILL_CATALOG_CATEGORY = "skill-catalog-category-lint"
LINT_SKILL_GLOBAL_NEGATIVE = "skill-global-negative-lint"
LINT_SKILL_POLICY_COVERAGE = "skill-policy-coverage-lint"
LINT_SKILL_PROCESS_METRIC = "skill-process-metric-lint"
LINT_SKILL_COMMAND_DRIFT = "skill-command-drift-lint"
LINT_SKILL_HELPER_SCRIPT = "skill-helper-script-lint"
LINT_SKILL_FIXTURE = "skill-fixture-lint"
LINT_AGENT_DOC_CROSSLINK = "agent-doc-crosslink-lint"
REQUIRED_FINDING_FIELDS = {"id", "severity", "file", "message", "hint"}


def test_run_v1_lints_passes_complete_skill(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    docs_root = _write_design_docs(tmp_path, ["nvflare-valid-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert result["status"] == "ok"
    assert result["findings"] == []
    assert result["summary"]["error_count"] == 0
    assert {"error", "warning", "info"}.isdisjoint(result["summary"])
    assert set(result["checks"]) == {
        LINT_SKILL_FRONTMATTER,
        LINT_SKILL_MD_SIZE,
        LINT_SKILL_TRIGGER,
        LINT_SKILL_TRIGGER_OVERLAP,
        LINT_SKILL_CATALOG_CATEGORY,
        LINT_SKILL_GLOBAL_NEGATIVE,
        LINT_SKILL_POLICY_COVERAGE,
        LINT_SKILL_PROCESS_METRIC,
        LINT_SKILL_COMMAND_DRIFT,
        LINT_SKILL_HELPER_SCRIPT,
        LINT_SKILL_FIXTURE,
        LINT_AGENT_DOC_CROSSLINK,
    }


def test_run_v1_lints_reports_frontmatter_prefix(tmp_path):
    _write_skill(tmp_path / "skills", "example-skill")
    docs_root = _write_design_docs(tmp_path, ["example-skill"], category="Orient")

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert _has_finding(result, LINT_SKILL_FRONTMATTER, "skill-name-prefix-required")
    _assert_structured_findings(result)


def test_run_v1_lints_allows_internal_skill_without_nvflare_prefix(tmp_path):
    _write_skill(tmp_path / "skills", "example-skill", status="internal")
    docs_root = _write_design_docs(tmp_path, ["example-skill"], category="Orient")

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_FRONTMATTER])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_run_v1_lints_reports_skill_md_size(tmp_path):
    body = "\n".join(f"line {i}" for i in range(205))
    _write_skill(tmp_path / "skills", "nvflare-large-skill", body=body)
    docs_root = _write_design_docs(tmp_path, ["nvflare-large-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

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
    docs_root = _write_design_docs(tmp_path, ["nvflare-trigger-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert _has_finding(result, LINT_SKILL_TRIGGER, "skill-positive-trigger-eval-missing")
    assert _has_finding(result, LINT_SKILL_TRIGGER, "skill-adjacent-negative-eval-missing")
    assert _has_finding(result, LINT_SKILL_GLOBAL_NEGATIVE, "skill-global-negative-eval-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_reports_missing_catalog_entry(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-one-skill")
    _write_skill(tmp_path / "skills", "nvflare-two-skill")
    docs_root = _write_design_docs(tmp_path, ["nvflare-one-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert _has_finding(result, LINT_SKILL_CATALOG_CATEGORY, "skill-catalog-entry-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_reports_trigger_overlap_without_negative_boundary(tmp_path):
    evals_one = _default_evals("nvflare-one-skill", adjacent_negative=False)
    evals_two = _default_evals("nvflare-two-skill", adjacent_negative=False)
    _write_skill(
        tmp_path / "skills",
        "nvflare-one-skill",
        description="Convert PyTorch training code to FLARE.",
        body="Use when converting PyTorch training code.\n",
        evals=evals_one,
    )
    _write_skill(
        tmp_path / "skills",
        "nvflare-two-skill",
        description="Convert PyTorch training code to FLARE.",
        body="Use when converting PyTorch training code.\n",
        evals=evals_two,
    )
    docs_root = _write_design_docs(tmp_path, ["nvflare-one-skill", "nvflare-two-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert _has_finding(result, LINT_SKILL_TRIGGER_OVERLAP, "skill-trigger-overlap")
    _assert_structured_findings(result)


def test_run_v1_lints_reports_policy_without_behavior_ids(tmp_path):
    evals = _default_evals("nvflare-policy-skill", include_behavior_ids=False)
    _write_skill(
        tmp_path / "skills", "nvflare-policy-skill", body="The agent must validate before submit.\n", evals=evals
    )
    docs_root = _write_design_docs(tmp_path, ["nvflare-policy-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert _has_finding(result, LINT_SKILL_POLICY_COVERAGE, "skill-policy-coverage-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_reports_unknown_nvflare_command(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-command-skill", body="Run `nvflare unknown --format json`.\n")
    docs_root = _write_design_docs(tmp_path, ["nvflare-command-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

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
    docs_root = _write_design_docs(tmp_path, ["nvflare-command-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_COMMAND_DRIFT])

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
    docs_root = _write_design_docs(tmp_path, ["nvflare-command-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_COMMAND_DRIFT])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_run_v1_lints_skips_trigger_overlap_when_skill_count_exceeds_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("NVFLARE_AGENT_MAX_TRIGGER_OVERLAP_SKILLS", "1")
    _write_skill(tmp_path / "skills", "nvflare-one-skill")
    _write_skill(tmp_path / "skills", "nvflare-two-skill")
    docs_root = _write_design_docs(tmp_path, ["nvflare-one-skill", "nvflare-two-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_TRIGGER_OVERLAP])

    assert result["status"] == "ok"
    assert result["findings"] == []
    assert result["skipped_checks"] == [
        {
            "id": LINT_SKILL_TRIGGER_OVERLAP,
            "reason": "category 'Conversion' has 2 skills; limit is 1",
        }
    ]


def test_run_v1_lints_reports_helper_script_without_test(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-helper-skill")
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("helper.py").write_text("print('{}')\n", encoding="utf-8")
    docs_root = _write_design_docs(tmp_path, ["nvflare-helper-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert _has_finding(result, LINT_SKILL_HELPER_SCRIPT, "skill-helper-tests-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_helper_script_ignores_symlink_loop(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-helper-skill")
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("helper.py").write_text("print('{}')\n", encoding="utf-8")
    _symlink_dir_or_skip(scripts_dir, scripts_dir / "loop")
    docs_root = _write_design_docs(tmp_path, ["nvflare-helper-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_HELPER_SCRIPT])

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
    docs_root = _write_design_docs(tmp_path, ["nvflare-helper-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_HELPER_SCRIPT])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_run_v1_lints_reports_missing_fixture_file(tmp_path):
    evals = _default_evals("nvflare-fixture-skill")
    evals["evals"][0]["files"] = ["evals/files/missing.py"]
    _write_skill(tmp_path / "skills", "nvflare-fixture-skill", evals=evals, write_fixture=False)
    docs_root = _write_design_docs(tmp_path, ["nvflare-fixture-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert _has_finding(result, LINT_SKILL_FIXTURE, "skill-fixture-file-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_fixture_file_check_ignores_symlink_loop(tmp_path):
    evals = _default_evals("nvflare-fixture-skill")
    evals["evals"][0]["files"] = ["evals/files/input.py"]
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-fixture-skill", evals=evals, write_fixture=False)
    files_dir = skill_dir / "evals" / "files"
    files_dir.mkdir()
    files_dir.joinpath("input.py").write_text("print('hello')\n", encoding="utf-8")
    _symlink_dir_or_skip(files_dir, files_dir / "loop")
    docs_root = _write_design_docs(tmp_path, ["nvflare-fixture-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_FIXTURE])

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
    docs_root = _write_design_docs(tmp_path, ["nvflare-fixture-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_FIXTURE])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_run_v1_lints_reference_text_scan_ignores_symlink_loop(tmp_path):
    skill_dir = _write_skill(tmp_path / "skills", "nvflare-command-skill")
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    references_dir.joinpath("guide.md").write_text("Run `nvflare unknown --format json`.\n", encoding="utf-8")
    _symlink_dir_or_skip(references_dir, references_dir / "loop")
    docs_root = _write_design_docs(tmp_path, ["nvflare-command-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_COMMAND_DRIFT])

    assert _has_finding(result, LINT_SKILL_COMMAND_DRIFT, "skill-command-drift")
    _assert_structured_findings(result)


def test_run_v1_lints_rejects_fixture_paths_that_escape_skill_dir(tmp_path):
    evals = _default_evals("nvflare-fixture-skill")
    evals["evals"][0]["files"] = ["../outside.py"]
    _write_skill(tmp_path / "skills", "nvflare-fixture-skill", evals=evals)
    docs_root = _write_design_docs(tmp_path, ["nvflare-fixture-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_FIXTURE])

    assert _has_finding(result, LINT_SKILL_FIXTURE, "skill-fixture-path-escape")
    _assert_structured_findings(result)


def test_run_v1_lints_reports_broken_doc_crosslink(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-doc-skill")
    docs_root = _write_design_docs(tmp_path, ["nvflare-doc-skill"])
    docs_root.joinpath("agent_integration.md").write_text(
        docs_root.joinpath("agent_integration.md").read_text(encoding="utf-8")
        + "\n[missing](missing.md)\n[bad anchor](agent_integration.md#nope)\n",
        encoding="utf-8",
    )

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert _has_finding(result, LINT_AGENT_DOC_CROSSLINK, "agent-doc-link-missing")
    assert _has_finding(result, LINT_AGENT_DOC_CROSSLINK, "agent-doc-anchor-missing")
    _assert_structured_findings(result)


def test_run_v1_lints_checks_current_skills_architecture_doc(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-doc-skill")
    docs_root = _write_design_docs(tmp_path, ["nvflare-doc-skill"])
    docs_root.joinpath("skills_architecture.md").write_text(
        "# Skills Architecture\n\n[missing](missing.md)\n",
        encoding="utf-8",
    )

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_AGENT_DOC_CROSSLINK])

    assert _has_finding(result, LINT_AGENT_DOC_CROSSLINK, "agent-doc-link-missing")


def test_run_v1_lints_supports_check_selection(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    _write_skill(tmp_path / "skills", "nvflare-other-skill", evals={"evals": []})
    docs_root = _write_design_docs(tmp_path, ["nvflare-valid-skill", "nvflare-other-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_FRONTMATTER])

    assert result["status"] == "ok"
    assert result["checks"] == [LINT_SKILL_FRONTMATTER]
    assert result["summary"]["skill_count"] == 2


def test_run_v1_lints_skips_shared_reference_dirs(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    shared_dir = tmp_path / "skills" / "_shared"
    shared_dir.mkdir()
    shared_dir.joinpath("reference.md").write_text("shared guidance\n", encoding="utf-8")
    docs_root = _write_design_docs(tmp_path, ["nvflare-valid-skill"])

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)

    assert result["status"] == "ok"
    assert result["summary"]["skill_count"] == 1
    assert result["findings"] == []


def test_run_v1_lints_records_doc_dependent_overlap_skip(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")

    result = run_v1_lints(
        tmp_path / "skills",
        docs_root=tmp_path / "missing-docs",
        checks=[LINT_SKILL_TRIGGER_OVERLAP],
    )

    assert result["status"] == "ok"
    assert result["skipped_checks"] == [{"id": LINT_SKILL_TRIGGER_OVERLAP, "reason": "docs root is not available"}]


def test_run_v1_lints_doc_crosslinks_skip_oversized_doc(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    docs_root = _write_design_docs(tmp_path, ["nvflare-valid-skill"])
    doc_path = docs_root / "agent_implementation_plan.md"
    doc_path.write_text("[Broken](missing.md)\n", encoding="utf-8")
    with doc_path.open("ab") as stream:
        stream.truncate(MAX_SKILL_TEXT_FILE_BYTES + 1)

    result = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_AGENT_DOC_CROSSLINK])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_catalog_parsers_skip_oversized_docs(tmp_path):
    docs_root = _write_design_docs(tmp_path, ["nvflare-valid-skill"])
    product_doc = docs_root / "agent_integration.md"
    conversion_doc = docs_root / "agent_skill_authoring.md"
    for doc_path in (product_doc, conversion_doc):
        with doc_path.open("ab") as stream:
            stream.truncate(MAX_SKILL_TEXT_FILE_BYTES + 1)

    assert _parse_product_catalog(product_doc) == {}
    assert _parse_conversion_table(conversion_doc) == {}


def test_validate_skills_filters_summary_to_requested_skill(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    _write_skill(tmp_path / "skills", "nvflare-other-skill", evals={"evals": []})
    docs_root = _write_design_docs(tmp_path, ["nvflare-valid-skill", "nvflare-other-skill"])

    result = validate_skills(tmp_path / "skills", skill_name="nvflare-valid-skill", docs_root=docs_root)

    assert result["status"] == "ok"
    assert result["requested_skill"] == "nvflare-valid-skill"
    assert result["summary"]["skill_count"] == 1
    assert result["findings"] == []


def test_validate_skills_excludes_unattributed_doc_findings_for_requested_skill(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    docs_root = _write_design_docs(tmp_path, ["nvflare-valid-skill"])
    docs_root.joinpath("agent_implementation_plan.md").write_text(
        "# Agent Implementation Plan\n\n[Missing](missing.md)\n",
        encoding="utf-8",
    )

    all_result = run_v1_lints(tmp_path / "skills", docs_root=docs_root)
    requested_result = validate_skills(tmp_path / "skills", skill_name="nvflare-valid-skill", docs_root=docs_root)

    assert _has_finding(all_result, LINT_AGENT_DOC_CROSSLINK, "agent-doc-link-missing")
    assert requested_result["status"] == "ok"
    assert requested_result["findings"] == []


def test_validate_skills_keeps_global_findings_for_requested_skill(tmp_path):
    result = validate_skills(tmp_path / "missing-skills", skill_name="nvflare-valid-skill", docs_root=tmp_path / "docs")

    assert result["status"] == "failed"
    assert result["summary"]["error_count"] == 1
    finding = _finding(result, LINT_SKILL_FRONTMATTER, "skills-root-missing")
    assert finding["global"] is True
    assert "skill" not in finding


def test_validate_skills_uses_requested_size_limit_without_mutating_default(tmp_path):
    _write_skill(tmp_path / "skills", "nvflare-valid-skill")
    docs_root = _write_design_docs(tmp_path, ["nvflare-valid-skill"])

    limited = validate_skills(
        tmp_path / "skills",
        skill_name="nvflare-valid-skill",
        docs_root=docs_root,
        max_skill_md_lines=2,
    )
    default = run_v1_lints(tmp_path / "skills", docs_root=docs_root, checks=[LINT_SKILL_MD_SIZE])

    assert _has_finding(limited, LINT_SKILL_MD_SIZE, "skill-md-too-large")
    assert default["status"] == "ok"
    assert default["findings"] == []


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
        f"{status_line}"
        "---\n"
        "\n"
        f"{body}",
        encoding="utf-8",
    )
    evals_dir = skill_dir / "evals"
    evals_dir.mkdir()
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


def _write_design_docs(tmp_path, skills, *, category="Conversion", tier="bundle"):
    docs_root = tmp_path / "docs"
    docs_root.mkdir(exist_ok=True)
    catalog_rows = "\n".join(f"| {category} | `{skill}` | {tier} | Test skill. |" for skill in skills)
    conversion_rows = "\n".join(f"| PyTorch | `{skill}` | Test scope. | Test fixture. | {tier} |" for skill in skills)
    lint_rows = "\n".join(f"| `{lint_id}` | Test lint definition. |" for lint_id in V1_LINT_IDS)

    docs_root.joinpath("agent_integration.md").write_text(
        "# Agent Integration\n\n"
        "## Product Skill Catalog\n\n"
        "| Category | Skill | Tier | Purpose |\n"
        "| --- | --- | --- | --- |\n"
        f"{catalog_rows}\n",
        encoding="utf-8",
    )
    docs_root.joinpath("agent_skill_authoring.md").write_text(
        "# Agent Skill Authoring\n\n"
        "## Conversion Skill Families\n\n"
        "| Code Family | Skill | Scope | Current Repo Evidence | Tier |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{conversion_rows}\n",
        encoding="utf-8",
    )
    docs_root.joinpath("agent_skill_evaluation.md").write_text(
        "# Agent Skill Evaluation\n\n"
        "## V1 Engineering Lints\n\n"
        "| Check | Definition |\n"
        "| --- | --- |\n"
        f"{lint_rows}\n",
        encoding="utf-8",
    )
    docs_root.joinpath("agent_implementation_plan.md").write_text(
        "# Agent Implementation Plan\n\n" "[V1 Engineering Lints](agent_skill_evaluation.md#v1-engineering-lints)\n",
        encoding="utf-8",
    )
    docs_root.joinpath("agent_skills_deferred_roadmap.md").write_text(
        "# Agent Skills Deferred Roadmap\n",
        encoding="utf-8",
    )
    docs_root.joinpath("skills_architecture.md").write_text(
        "# Skills Architecture\n\n" "[V1 Engineering Lints](agent_skill_evaluation.md#v1-engineering-lints)\n",
        encoding="utf-8",
    )
    return docs_root


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
                "files": ["evals/files/input.py"],
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
