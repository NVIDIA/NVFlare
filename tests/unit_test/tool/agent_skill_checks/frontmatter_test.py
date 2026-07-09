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
import sys
from pathlib import Path

import pytest

CHECKS_PARENT = Path(__file__).resolve().parents[4] / "dev_tools" / "agent" / "skills"
sys.path.insert(0, str(CHECKS_PARENT))

from checks.frontmatter import (  # noqa: E402
    SPEC_TOP_LEVEL_FIELDS,
    SkillFrontmatterError,
    parse_skill_frontmatter,
    should_skip_skill_dir,
    validate_skill_dir,
    validate_skills_root,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILLS_ROOT = REPO_ROOT / "skills"


def test_shipped_skills_frontmatter_is_agentskills_spec_compliant():
    # Every shipped skill must keep only agentskills.io top-level keys; NVFLARE
    # custom fields (min_flare_version, blast_radius, category, ...) live under
    # `metadata:`. Locks in the spec alignment so a top-level custom field can't
    # silently regress.
    skill_dirs = [d for d in sorted(SKILLS_ROOT.iterdir()) if not should_skip_skill_dir(d)]
    assert skill_dirs, "no shipped skills found"
    for skill_dir in skill_dirs:
        metadata = parse_skill_frontmatter(skill_dir / "SKILL.md")
        extra = set(metadata) - SPEC_TOP_LEVEL_FIELDS
        assert not extra, f"{skill_dir.name}: non-spec top-level frontmatter keys {sorted(extra)}"
        assert isinstance(metadata.get("metadata"), dict)
        assert "min_flare_version" in metadata["metadata"]
        assert "blast_radius" in metadata["metadata"]
        assert validate_skill_dir(skill_dir).ok


def test_validate_skill_dir_rejects_top_level_custom_field(tmp_path):
    # A custom field left at the top level (not nested under metadata) is flagged.
    skill_dir = tmp_path / "nvflare-top-level-custom"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-top-level-custom\n"
        "description: Fixture.\n"
        'min_flare_version: "2.8.0"\n'
        "metadata:\n"
        "  blast_radius: read_only\n"
        "  category: Test\n"
        "---\n\n# Skill\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert "skill-frontmatter-field-unsupported" in {issue.code for issue in result.issues}


FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_skill_frontmatter_reads_required_fields():
    metadata = parse_skill_frontmatter(FIXTURES / "valid" / "nvflare-example-skill" / "SKILL.md")

    assert metadata["name"] == "nvflare-example-skill"
    assert metadata["description"] == "Example fixture skill used by frontmatter validator tests."
    assert metadata["metadata"]["min_flare_version"] == "2.8.0"
    assert metadata["metadata"]["blast_radius"] == "read_only"
    assert metadata["metadata"]["category"] == "Test"


def test_parse_skill_frontmatter_accepts_utf8_bom(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_bytes(
        b"\xef\xbb\xbf---\n"
        b"name: nvflare-bom-skill\n"
        b"description: Test skill fixture.\n"
        b'min_flare_version: "2.8.0"\n'
        b"blast_radius: read_only\n"
        b"category: Test\n"
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


def test_validate_skill_dir_allows_name_prefix_for_visibility_aware_lint(tmp_path):
    skill_dir = _write_skill(tmp_path, "example-skill")

    result = validate_skill_dir(skill_dir)

    assert result.ok
    assert result.metadata["name"] == "example-skill"


def test_validate_skill_dir_reports_missing_required_fields(tmp_path):
    skill_dir = tmp_path / "nvflare-missing-fields"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-missing-fields\n"
        "description: Missing required fields.\n"
        "metadata:\n"
        '  author: "Test Author <test-author@nvidia.com>"\n'
        "---\n"
        "\n"
        "# Missing Fields\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert _issue_codes(result) == {"skill-frontmatter-field-required"}
    assert len(result.issues) == 3


def test_validate_skill_dir_reports_wrong_type_fields(tmp_path):
    skill_dir = tmp_path / "nvflare-wrong-type"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-wrong-type\n"
        "description: Wrong type fixture.\n"
        "metadata:\n"
        '  author: "Test Author <test-author@nvidia.com>"\n'
        "  min_flare_version: 2.8\n"
        "  blast_radius: read_only\n"
        "  category: Test\n"
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


def test_validate_skill_dir_requires_category_for_public_skill(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-missing-category", category=None)

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert _issue_codes(result) == {"skill-frontmatter-field-required"}
    assert "category" in result.issues[0].message


@pytest.mark.parametrize("status", ["draft", "internal", "private"])
def test_validate_skill_dir_allows_missing_category_for_non_public_skill(tmp_path, status):
    skill_dir = _write_skill(tmp_path, f"nvflare-{status}-skill", category=None, status=status)

    result = validate_skill_dir(skill_dir)

    assert result.ok
    assert result.issues == ()


def test_validate_skill_dir_accepts_category_frontmatter(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-category-skill", category="Test")

    result = validate_skill_dir(skill_dir)

    assert result.ok
    assert result.metadata["metadata"]["category"] == "Test"
    assert result.issues == ()


def test_validate_skill_dir_rejects_wrong_type_category(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-category-skill", category=123)

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert _issue_codes(result) == {"skill-frontmatter-field-type"}
    assert "category" in result.issues[0].message


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


def test_parse_skill_frontmatter_rejects_yaml_anchors_and_aliases(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: nvflare-anchor-skill\n"
        "description: &shared Test skill fixture.\n"
        'min_flare_version: "2.8.0"\n'
        "blast_radius: *shared\n"
        "---\n",
        encoding="utf-8",
    )

    with pytest.raises(SkillFrontmatterError, match="anchors or aliases"):
        parse_skill_frontmatter(skill_file)


def test_parse_skill_frontmatter_rejects_symlink(tmp_path):
    target = tmp_path / "target.md"
    target.write_text("---\nname: nvflare-link\ndescription: Link.\n---\n", encoding="utf-8")
    skill_file = tmp_path / "SKILL.md"
    try:
        skill_file.symlink_to(target)
    except (NotImplementedError, OSError) as e:
        pytest.skip(f"file symlink is not available in this environment: {e}")

    with pytest.raises(SkillFrontmatterError, match="must not be a symlink"):
        parse_skill_frontmatter(skill_file)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is not available on this platform")
def test_validate_skill_dir_rejects_special_file_without_opening_it(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-special-file")
    os.mkfifo(skill_dir / "references.fifo")

    result = validate_skill_dir(skill_dir)

    assert "skill-special-file-not-allowed" in _issue_codes(result)


@pytest.mark.skipif(not Path("/dev/zero").exists(), reason="/dev/zero is not available on this platform")
def test_parse_skill_frontmatter_rejects_character_device_without_reading_it():
    with pytest.raises(SkillFrontmatterError, match="must be a regular file"):
        parse_skill_frontmatter(Path("/dev/zero"))


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


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="unreadable-file test requires a non-root POSIX user (root bypasses file permissions)",
)
def test_validate_skill_dir_reports_unreadable_skill_file(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-unreadable-skill")
    skill_file = skill_dir / "SKILL.md"
    skill_file.chmod(0o000)
    try:
        result = validate_skill_dir(skill_dir)
    finally:
        skill_file.chmod(0o644)

    assert not result.ok
    assert _issue_codes(result) == {"skill-md-unreadable"}


def test_validate_skills_root_skips_non_skill_files(tmp_path):
    _write_skill(tmp_path, "nvflare-valid-one")
    (tmp_path / "README.md").write_text("not a skill\n", encoding="utf-8")
    (tmp_path / ".hidden-dir").mkdir()
    (tmp_path / "_shared").mkdir()

    results = validate_skills_root(tmp_path)

    assert len(results) == 1
    assert results[0].ok
    assert results[0].metadata["name"] == "nvflare-valid-one"


def test_validate_skills_root_reports_missing_root(tmp_path):
    results = validate_skills_root(tmp_path / "missing")

    assert len(results) == 1
    assert not results[0].ok
    assert _issue_codes(results[0]) == {"skills-root-missing"}


def _write_skill(tmp_path, skill_name, *, name=None, blast_radius="read_only", category="Test", status=None):
    skill_dir = tmp_path / skill_name
    skill_dir.mkdir()
    category_line = f"  category: {category}\n" if category is not None else ""
    status_line = f"  status: {status}\n" if status is not None else ""
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        f"name: {name or skill_name}\n"
        "description: Test skill fixture.\n"
        "metadata:\n"
        '  author: "Test Author <test-author@nvidia.com>"\n'
        '  min_flare_version: "2.8.0"\n'
        f"  blast_radius: {blast_radius}\n"
        f"{category_line}"
        f"{status_line}"
        "---\n"
        "\n"
        "# Test Skill\n",
        encoding="utf-8",
    )
    return skill_dir


def _issue_codes(result):
    return {issue.code for issue in result.issues}


def test_validate_skill_dir_rejects_invalid_name_charset(tmp_path):
    # Company (NVCARPS) name constraints: lowercase alphanumeric + hyphens, no
    # leading/trailing/consecutive hyphens, at most 64 chars.
    for bad_name in ("nvflare--double", "nvflare-Upper", "nvflare-trailing-", "-nvflare-leading"):
        skill_dir = _write_skill(tmp_path, bad_name)
        result = validate_skill_dir(skill_dir)
        assert "skill-frontmatter-name-invalid" in _issue_codes(result), bad_name


def test_validate_skill_dir_rejects_name_over_64_chars(tmp_path):
    long_name = "nvflare-" + "a" * 64
    skill_dir = _write_skill(tmp_path, long_name)

    result = validate_skill_dir(skill_dir)

    assert "skill-frontmatter-name-invalid" in _issue_codes(result)


def test_validate_skill_dir_rejects_description_over_1024_chars(tmp_path):
    skill_dir = tmp_path / "nvflare-long-description"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-long-description\n"
        f"description: {'x' * 1025}\n"
        "metadata:\n"
        '  author: "Test Author <test-author@nvidia.com>"\n'
        '  min_flare_version: "2.8.0"\n'
        "  blast_radius: read_only\n"
        "  category: Test\n"
        "---\n\n# Skill\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert "skill-frontmatter-description-too-long" in _issue_codes(result)


def test_validate_skill_dir_rejects_compatibility_over_500_chars(tmp_path):
    skill_dir = tmp_path / "nvflare-long-compat"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-long-compat\n"
        "description: Test skill fixture.\n"
        f"compatibility: {'y' * 501}\n"
        "metadata:\n"
        '  author: "Test Author <test-author@nvidia.com>"\n'
        '  min_flare_version: "2.8.0"\n'
        "  blast_radius: read_only\n"
        "  category: Test\n"
        "---\n\n# Skill\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert "skill-frontmatter-compatibility-too-long" in _issue_codes(result)


def test_validate_skill_dir_rejects_skill_md_over_500_lines(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-long-body")
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        skill_file.read_text(encoding="utf-8") + ("filler line\n" * 500),
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert "skill-md-too-long" in _issue_codes(result)


def test_validate_skill_dir_requires_author_in_metadata(tmp_path):
    # Company (NVCARPS) requirement: metadata.author is mandatory.
    skill_dir = tmp_path / "nvflare-no-author"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-no-author\n"
        "description: Test skill fixture.\n"
        "metadata:\n"
        '  min_flare_version: "2.8.0"\n'
        "  blast_radius: read_only\n"
        "  category: Test\n"
        "---\n\n# Skill\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert not result.ok
    assert any(
        issue.code == "skill-frontmatter-field-required" and "'author'" in issue.message for issue in result.issues
    )


@pytest.mark.parametrize(
    "author",
    [
        "nvflare",  # bare project name, no contact
        "Jane Doe <jane@gmail.com>",  # non-NVIDIA email
        "federatedlearning@nvidia.com",  # email without a display name
    ],
)
def test_validate_skill_dir_rejects_author_without_team_identity_format(tmp_path, author):
    # The NVIDIA skills catalog convention: author is a team identity with an
    # NVIDIA email so the support contact survives outside the repo.
    skill_dir = tmp_path / "nvflare-bad-author"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-bad-author\n"
        "description: Test skill fixture.\n"
        "metadata:\n"
        f'  author: "{author}"\n'
        '  min_flare_version: "2.8.0"\n'
        "  blast_radius: read_only\n"
        "  category: Test\n"
        "---\n\n# Skill\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert "skill-author-format-invalid" in _issue_codes(result)


def test_validate_skill_dir_accepts_top_level_license_and_version(tmp_path):
    # The NVIDIA skills catalog declares `license` and `version` at the top
    # level; both must pass the top-level field check.
    skill_dir = tmp_path / "nvflare-licensed"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-licensed\n"
        "description: Test skill fixture.\n"
        "license: Apache-2.0\n"
        'version: "0.1.0"\n'
        "metadata:\n"
        '  author: "NVIDIA FLARE Team <federatedlearning@nvidia.com>"\n'
        '  min_flare_version: "2.8.0"\n'
        "  blast_radius: read_only\n"
        "  category: Test\n"
        "---\n\n# Skill\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert result.ok, result.issues


def test_validate_skill_dir_accepts_optional_title(tmp_path):
    # NVCARPS allows an optional top-level display title.
    skill_dir = tmp_path / "nvflare-with-title"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: nvflare-with-title\n"
        'title: "NVFLARE With Title"\n'
        "description: Test skill fixture.\n"
        "metadata:\n"
        '  author: "Test Author <test-author@nvidia.com>"\n'
        '  min_flare_version: "2.8.0"\n'
        "  blast_radius: read_only\n"
        "  category: Test\n"
        "---\n\n# Skill\n",
        encoding="utf-8",
    )

    result = validate_skill_dir(skill_dir)

    assert result.ok, result.issues
