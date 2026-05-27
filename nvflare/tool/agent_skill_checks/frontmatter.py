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

"""Minimal V1 validation for NVFLARE agent skill frontmatter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

SKILL_FILE_NAME = "SKILL.md"
REQUIRED_FRONTMATTER_FIELDS = ("name", "description", "min_flare_version", "blast_radius")
VALID_BLAST_RADIUS = frozenset(
    {
        "read_only",
        "edits_files",
        "runs_simulator",
        "submits_poc",
        "submits_production",
    }
)


@dataclass(frozen=True)
class SkillValidationIssue:
    code: str
    message: str
    path: str


@dataclass(frozen=True)
class SkillValidationResult:
    skill_dir: str
    metadata: Mapping[str, Any]
    issues: tuple[SkillValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


class SkillFrontmatterError(ValueError):
    """Raised when SKILL.md frontmatter cannot be parsed."""


def parse_skill_frontmatter(skill_file: Path | str) -> dict[str, Any]:
    """Parse YAML frontmatter from a SKILL.md file."""
    path = Path(skill_file)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillFrontmatterError("SKILL.md must start with YAML frontmatter delimiter '---'")

    close_index = _find_closing_delimiter(lines)
    if close_index is None:
        raise SkillFrontmatterError("SKILL.md frontmatter must end with delimiter '---'")

    raw_frontmatter = "\n".join(lines[1:close_index])
    try:
        metadata = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError as e:
        raise SkillFrontmatterError(f"failed to parse YAML frontmatter: {e}") from e

    if not isinstance(metadata, dict):
        raise SkillFrontmatterError("SKILL.md frontmatter must be a mapping")
    return metadata


def validate_skill_dir(skill_dir: Path | str) -> SkillValidationResult:
    """Validate one guide-compatible skill directory."""
    path = Path(skill_dir)
    issues: list[SkillValidationIssue] = []
    metadata: dict[str, Any] = {}

    skill_file = path / SKILL_FILE_NAME
    if not path.is_dir():
        issues.append(_issue("skill-dir-missing", "skill path must be a directory", path))
        return SkillValidationResult(str(path), metadata, tuple(issues))
    if not skill_file.is_file():
        issues.append(_issue("skill-md-missing", "skill directory must contain SKILL.md", skill_file))
        return SkillValidationResult(str(path), metadata, tuple(issues))

    try:
        metadata = parse_skill_frontmatter(skill_file)
    except SkillFrontmatterError as e:
        issues.append(_issue("skill-frontmatter-invalid", str(e), skill_file))
        return SkillValidationResult(str(path), metadata, tuple(issues))

    _validate_required_fields(metadata, skill_file, issues)
    _validate_name_matches_directory(metadata.get("name"), path, skill_file, issues)
    _validate_blast_radius(metadata.get("blast_radius"), skill_file, issues)

    return SkillValidationResult(str(path), metadata, tuple(issues))


def validate_skills_root(skills_root: Path | str) -> list[SkillValidationResult]:
    """Validate every skill directory under a skills root."""
    root = Path(skills_root)
    if not root.is_dir():
        return [SkillValidationResult(str(root), {}, (_issue("skills-root-missing", "skills root is missing", root),))]

    results = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.name.startswith(".") or not child.is_dir():
            continue
        results.append(validate_skill_dir(child))
    return results


def _find_closing_delimiter(lines: list[str]) -> Optional[int]:
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return i
    return None


def _validate_required_fields(
    metadata: Mapping[str, Any], skill_file: Path, issues: list[SkillValidationIssue]
) -> None:
    for field in REQUIRED_FRONTMATTER_FIELDS:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append(
                _issue("skill-frontmatter-field-required", f"frontmatter field '{field}' is required", skill_file)
            )


def _validate_name_matches_directory(
    name: Any, skill_dir: Path, skill_file: Path, issues: list[SkillValidationIssue]
) -> None:
    if isinstance(name, str) and name.strip() and name != skill_dir.name:
        issues.append(
            _issue(
                "skill-name-directory-mismatch",
                f"frontmatter name '{name}' must match directory name '{skill_dir.name}'",
                skill_file,
            )
        )


def _validate_blast_radius(radius: Any, skill_file: Path, issues: list[SkillValidationIssue]) -> None:
    if isinstance(radius, str) and radius.strip() and radius not in VALID_BLAST_RADIUS:
        issues.append(
            _issue(
                "skill-blast-radius-invalid",
                f"blast_radius must be one of: {', '.join(sorted(VALID_BLAST_RADIUS))}",
                skill_file,
            )
        )


def _issue(code: str, message: str, path: Path) -> SkillValidationIssue:
    return SkillValidationIssue(code=code, message=message, path=str(path))
