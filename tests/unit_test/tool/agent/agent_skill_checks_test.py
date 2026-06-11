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
from pathlib import Path

from nvflare.tool.agent_skill_checks.lints import run_v1_lints, validate_skills


def test_process_metric_lint_requires_process_metrics(tmp_path):
    _write_skill(
        tmp_path,
        "nvflare-test-skill",
        {
            "skill_name": "nvflare-test-skill",
            "evals": [
                {
                    "id": "test-positive",
                    "prompt": "Use this test skill.",
                    "expected_output": "The skill runs.",
                    "files": [],
                    "assertions": ["The skill runs."],
                    "nvflare": {
                        "expected_skill": "nvflare-test-skill",
                        "mandatory_behavior": [{"id": "run-test", "description": "runs the test workflow"}],
                    },
                }
            ],
        },
    )

    result = run_v1_lints(tmp_path, checks=["skill-process-metric-lint"])

    assert result["status"] == "failed"
    assert result["findings"][0]["code"] == "skill-process-metric-missing"


def test_process_metric_lint_accepts_process_metrics(tmp_path):
    _write_skill(
        tmp_path,
        "nvflare-test-skill",
        {
            "skill_name": "nvflare-test-skill",
            "evals": [
                {
                    "id": "test-positive",
                    "prompt": "Use this test skill.",
                    "expected_output": "The skill runs.",
                    "files": [],
                    "assertions": ["The skill runs."],
                    "nvflare": {
                        "expected_skill": "nvflare-test-skill",
                        "mandatory_behavior": [{"id": "run-test", "description": "runs the test workflow"}],
                        "process_metrics": [
                            {
                                "id": "turns_to_acceptable",
                                "description": "number of turns before the result is acceptable",
                            }
                        ],
                    },
                }
            ],
        },
    )

    result = run_v1_lints(tmp_path, checks=["skill-process-metric-lint"])

    assert result["status"] == "ok"
    assert result["findings"] == []


def test_helper_script_json_warning_applies_only_to_python_helpers(tmp_path):
    skill_dir = _write_skill(
        tmp_path,
        "nvflare-test-skill",
        {
            "skill_name": "nvflare-test-skill",
            "evals": [],
        },
    )
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("emit_json.sh").write_text("#!/bin/sh\njq . input.json\n", encoding="utf-8")
    tests_dir = skill_dir / "tests"
    tests_dir.mkdir()
    tests_dir.joinpath("emit_json_test.txt").write_text("shell helper test placeholder\n", encoding="utf-8")

    result = run_v1_lints(tmp_path, checks=["skill-helper-script-lint"])

    assert not any(finding.get("code") == "skill-helper-json-unclear" for finding in result["findings"])


def test_helper_script_json_warning_ignores_python_json_reader(tmp_path):
    skill_dir = _write_skill(
        tmp_path,
        "nvflare-test-skill",
        {
            "skill_name": "nvflare-test-skill",
            "evals": [],
        },
    )
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    scripts_dir.joinpath("read_json.py").write_text(
        "import json\n"
        "with open('input.json') as f:\n"
        "    data = json.load(f)\n"
        "print(data.get('name', 'missing'))\n",
        encoding="utf-8",
    )
    tests_dir = skill_dir / "tests"
    tests_dir.mkdir()
    tests_dir.joinpath("read_json_test.txt").write_text("python helper test placeholder\n", encoding="utf-8")

    result = run_v1_lints(tmp_path, checks=["skill-helper-script-lint"])

    assert not any(finding.get("code") == "skill-helper-json-unclear" for finding in result["findings"])


def test_trigger_overlap_limit_uses_current_environment(tmp_path, monkeypatch):
    _write_skill(tmp_path, "nvflare-left", {"skill_name": "nvflare-left", "evals": []})
    _write_skill(tmp_path, "nvflare-right", {"skill_name": "nvflare-right", "evals": []})
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    docs_root.joinpath("agent_integration.md").write_text(
        "| Category | Skill | Tier | Purpose |\n"
        "| --- | --- | --- | --- |\n"
        "| Conversion | `nvflare-left` | Seed | Test fixture. |\n"
        "| Conversion | `nvflare-right` | Seed | Test fixture. |\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NVFLARE_AGENT_MAX_TRIGGER_OVERLAP_SKILLS", "1")

    result = run_v1_lints(tmp_path, docs_root=docs_root, checks=["skill-trigger-overlap-lint"])

    assert result["skipped_checks"][0]["id"] == "skill-trigger-overlap-lint"
    assert "limit is 1" in result["skipped_checks"][0]["reason"]


def test_iter_skill_text_files_skips_oversized_references(tmp_path):
    from nvflare.tool.agent_skill_checks import lints

    skill_dir = _write_skill(tmp_path, "nvflare-test-skill", {"skill_name": "nvflare-test-skill", "evals": []})
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    references_dir.joinpath("large.md").write_text("x" * (lints.MAX_SKILL_TEXT_FILE_BYTES + 1), encoding="utf-8")
    references_dir.joinpath("small.md").write_text("small reference\n", encoding="utf-8")

    files = [path.name for path, _text in lints._iter_skill_text_files(skill_dir)]

    assert "large.md" not in files
    assert "small.md" in files


def test_doc_crosslink_lint_reads_each_doc_once(tmp_path, monkeypatch):
    from nvflare.tool.agent_skill_checks import lints

    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    docs_root.joinpath("agent_integration.md").write_text("# Integration\n", encoding="utf-8")
    docs_root.joinpath("agent_skill_evaluation.md").write_text(
        "# Evaluation\n" + "\n".join(f"`{lint_id}`" for lint_id in lints.V1_LINT_IDS),
        encoding="utf-8",
    )
    read_counts: dict[Path, int] = {}
    original_read_text = Path.read_text

    def counting_read_text(path, *args, **kwargs):
        if docs_root in path.parents:
            read_counts[path] = read_counts.get(path, 0) + 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    result = run_v1_lints(skills_root, docs_root=docs_root, checks=["agent-doc-crosslink-lint"])

    assert result["status"] == "ok"
    assert read_counts == {
        docs_root / "agent_integration.md": 1,
        docs_root / "agent_skill_evaluation.md": 1,
    }


def test_doc_crosslink_lint_accepts_anchor_in_linked_markdown_outside_canonical_docs(tmp_path):
    from nvflare.tool.agent_skill_checks import lints

    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    docs_root.joinpath("agent_integration.md").write_text("[README usage](README.md#usage)\n", encoding="utf-8")
    docs_root.joinpath("agent_skill_evaluation.md").write_text(
        "# Evaluation\n" + "\n".join(f"`{lint_id}`" for lint_id in lints.V1_LINT_IDS),
        encoding="utf-8",
    )
    docs_root.joinpath("README.md").write_text("# Usage\n", encoding="utf-8")

    result = run_v1_lints(skills_root, docs_root=docs_root, checks=["agent-doc-crosslink-lint"])

    assert result["status"] == "ok"
    assert not any(finding.get("code") == "agent-doc-anchor-missing" for finding in result["findings"])


def test_validate_skills_reuses_loaded_skill_records(tmp_path, monkeypatch):
    from nvflare.tool.agent_skill_checks import lints

    _write_skill(tmp_path, "nvflare-test-skill", {"skill_name": "nvflare-test-skill", "evals": []})
    load_count = 0
    original_load = lints._load_skill_records

    def counting_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(lints, "_load_skill_records", counting_load)

    result = validate_skills(tmp_path, skill_name="nvflare-test-skill")

    assert result["requested_skill"] == "nvflare-test-skill"
    assert load_count == 1


def _write_skill(root, name, evals):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Test skill fixture.\n"
        'min_flare_version: "2.8.0"\n'
        "blast_radius: read_only\n"
        "---\n"
        "\n"
        "# Test Skill\n"
        "\n"
        "## Use When\n"
        "\n"
        "Use when testing skill process metrics.\n",
        encoding="utf-8",
    )
    evals_dir = skill_dir / "evals"
    evals_dir.mkdir()
    evals_dir.joinpath("evals.json").write_text(json.dumps(evals), encoding="utf-8")
    return skill_dir
