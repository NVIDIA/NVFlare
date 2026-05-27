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


def test_parse_skill_frontmatter_accepts_utf8_bom(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_bytes(
        b"\xef\xbb\xbf---\n"
        b"name: nvflare-bom-skill\n"
        b"description: Test skill fixture.\n"
        b'min_flare_version: "2.8.0"\n'
        b"blast_radius: read_only\n"
        b"---\n"
        b"\n"
        b"# Test Skill\n"
    )

    metadata = parse_skill_frontmatter(skill_file)

    assert metadata["name"] == "nvflare-bom-skill"


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


def test_validate_skill_dir_reports_wrong_type_fields(tmp_path):
    skill_dir = tmp_path / "nvflare-wrong-type"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-wrong-type\n"
        "description: Wrong type fixture.\n"
        "min_flare_version: 2.8\n"
        "blast_radius: read_only\n"
        "---\n"
        "\n"
        "# Wrong Type\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert _issue_codes(result) == {"skill-frontmatter-field-type"}
    assert "min_flare_version" in result.issues[0].message
    assert "float=2.8" in result.issues[0].message


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


def test_parse_skill_frontmatter_requires_closing_delimiter(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\n" "name: nvflare-no-closing-delimiter\n" "description: Missing closing delimiter.\n",
        encoding="utf-8",
    )

    with pytest.raises(SkillFrontmatterError, match="must end"):
        parse_skill_frontmatter(skill_file)


def test_parse_skill_frontmatter_rejects_malformed_yaml(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: [unclosed\n---\n", encoding="utf-8")

    with pytest.raises(SkillFrontmatterError, match="failed to parse YAML"):
        parse_skill_frontmatter(skill_file)


def test_parse_skill_frontmatter_rejects_non_mapping(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\n- name\n- description\n---\n", encoding="utf-8")

    with pytest.raises(SkillFrontmatterError, match="must be a mapping"):
        parse_skill_frontmatter(skill_file)


def test_validate_skill_dir_reports_missing_directory(tmp_path):
    result = validate_skill_dir(tmp_path / "nvflare-missing-directory")

    assert not result.ok
    assert _issue_codes(result) == {"skill-dir-missing"}


def test_validate_skill_dir_reports_missing_skill_file(tmp_path):
    skill_dir = tmp_path / "nvflare-missing-skill-md"
    skill_dir.mkdir()

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert _issue_codes(result) == {"skill-md-missing"}


def test_validate_skills_root_skips_non_skill_files(tmp_path):
    _write_skill(tmp_path, "nvflare-valid-one")
    (tmp_path / "README.md").write_text("not a skill\n", encoding="utf-8")
    (tmp_path / ".hidden-dir").mkdir()

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
