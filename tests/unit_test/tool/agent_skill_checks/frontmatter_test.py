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

from pathlib import Path

import pytest

from nvflare.tool.agent_skill_checks.frontmatter import (
    SkillFrontmatterError,
    parse_skill_frontmatter,
    validate_skill_dir,
    validate_skills_root,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_skill_frontmatter_reads_required_fields():
    metadata = parse_skill_frontmatter(FIXTURES / "valid" / "nvflare-example-skill" / "SKILL.md")

    assert metadata["name"] == "nvflare-example-skill"
    assert metadata["description"] == "Example fixture skill used by frontmatter validator tests."
    assert metadata["min_flare_version"] == "2.8.0"
    assert metadata["blast_radius"] == "read_only"


def test_validate_skill_dir_accepts_fixture_skill():
    result = validate_skill_dir(FIXTURES / "valid" / "nvflare-example-skill")

    assert result.ok
    assert result.metadata["name"] == "nvflare-example-skill"
    assert result.issues == ()


def test_validate_skill_dir_requires_directory_name_to_match_frontmatter(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-other-name", name="nvflare-example-skill")

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert _issue_codes(result) == {"skill-name-directory-mismatch"}


def test_validate_skill_dir_reports_missing_required_fields(tmp_path):
    skill_dir = tmp_path / "nvflare-missing-fields"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-missing-fields\n"
        "description: Missing two required fields.\n"
        "---\n"
        "\n"
        "# Missing Fields\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert _issue_codes(result) == {"skill-frontmatter-field-required"}
    assert len(result.issues) == 2


def test_validate_skill_dir_rejects_invalid_blast_radius(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-invalid-radius", blast_radius="global_admin")

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert _issue_codes(result) == {"skill-blast-radius-invalid"}


def test_parse_skill_frontmatter_requires_delimiters(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("name: nvflare-no-frontmatter\n", encoding="utf-8")

    with pytest.raises(SkillFrontmatterError, match="must start"):
        parse_skill_frontmatter(skill_file)


def test_validate_skill_dir_reports_missing_skill_file(tmp_path):
    skill_dir = tmp_path / "nvflare-missing-skill-md"
    skill_dir.mkdir()

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert _issue_codes(result) == {"skill-md-missing"}


def test_validate_skills_root_skips_non_skill_files(tmp_path):
    _write_skill(tmp_path, "nvflare-valid-one")
    (tmp_path / "README.md").write_text("not a skill\n", encoding="utf-8")

    results = validate_skills_root(tmp_path)

    assert len(results) == 1
    assert results[0].ok
    assert results[0].metadata["name"] == "nvflare-valid-one"


def test_validate_skills_root_reports_missing_root(tmp_path):
    results = validate_skills_root(tmp_path / "missing")

    assert len(results) == 1
    assert not results[0].ok
    assert _issue_codes(results[0]) == {"skills-root-missing"}


def _write_skill(tmp_path, skill_name, *, name=None, blast_radius="read_only"):
    skill_dir = tmp_path / skill_name
    skill_dir.mkdir()
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        f"name: {name or skill_name}\n"
        "description: Test skill fixture.\n"
        'min_flare_version: "2.8.0"\n'
        f"blast_radius: {blast_radius}\n"
        "---\n"
        "\n"
        "# Test Skill\n",
        encoding="utf-8",
    )
    return skill_dir


def _issue_codes(result):
    return {issue.code for issue in result.issues}
