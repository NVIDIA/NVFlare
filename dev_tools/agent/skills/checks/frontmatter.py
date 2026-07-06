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

"""Validation for NVFLARE agent skill frontmatter."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

SKILL_FILE_NAME = "SKILL.md"
YAML_ANCHOR_OR_ALIAS_RE = re.compile(r"(^|[:\s\[{,])([&*])[A-Za-z0-9_-]+(?=\s|$|[,}\]])")
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
NVFLARE_METADATA_FIELDS = ("min_flare_version", "max_flare_version", "blast_radius", "skill_version", "status")


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
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillFrontmatterError("SKILL.md must start with YAML frontmatter delimiter '---'")

    close_index = _find_closing_delimiter(lines)
    if close_index is None:
        raise SkillFrontmatterError("SKILL.md frontmatter must end with delimiter '---'")

    raw_frontmatter = "\n".join(lines[1:close_index])
    if YAML_ANCHOR_OR_ALIAS_RE.search(raw_frontmatter):
        raise SkillFrontmatterError("SKILL.md frontmatter must not use YAML anchors or aliases")
    try:
        metadata = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError as e:
        raise SkillFrontmatterError(f"failed to parse YAML frontmatter: {e}") from e

    if not isinstance(metadata, dict):
        raise SkillFrontmatterError("SKILL.md frontmatter must be a mapping")
    return metadata


def normalize_skill_metadata(frontmatter: Mapping[str, Any]) -> dict[str, Any]:
    """Return NVFLARE metadata from spec-compliant or legacy frontmatter."""
    normalized = dict(frontmatter)
    extension = frontmatter.get("metadata")
    if extension is None:
        return normalized
    if not isinstance(extension, dict):
        raise SkillFrontmatterError("SKILL.md frontmatter field 'metadata' must be a mapping")
    for field in NVFLARE_METADATA_FIELDS:
        nested_value = extension.get(field)
        if nested_value is None:
            continue
        legacy_value = frontmatter.get(field)
        if legacy_value is not None and legacy_value != nested_value:
            raise SkillFrontmatterError(f"conflicting values for frontmatter field '{field}'")
        normalized[field] = nested_value
    return normalized


def validate_skill_dir(skill_dir: Path | str) -> SkillValidationResult:
    """Validate one guide-compatible skill directory."""
    path = Path(skill_dir)
    issues: list[SkillValidationIssue] = []
    metadata: dict[str, Any] = {}

    skill_file = path / SKILL_FILE_NAME
    if path.is_symlink():
        issues.append(_issue("skill-symlink-not-allowed", "skill path must not be a symlink", path))
        return SkillValidationResult(str(path), metadata, tuple(issues))
    if not path.is_dir():
        issues.append(_issue("skill-dir-missing", "skill path must be a directory", path))
        return SkillValidationResult(str(path), metadata, tuple(issues))
    _validate_no_symlinks(path, issues)
    if issues:
        return SkillValidationResult(str(path), metadata, tuple(issues))
    if not skill_file.is_file():
        issues.append(_issue("skill-md-missing", "skill directory must contain SKILL.md", skill_file))
        return SkillValidationResult(str(path), metadata, tuple(issues))

    try:
        metadata = normalize_skill_metadata(parse_skill_frontmatter(skill_file))
    except SkillFrontmatterError as e:
        issues.append(_issue("skill-frontmatter-invalid", str(e), skill_file))
        return SkillValidationResult(str(path), metadata, tuple(issues))
    except OSError as e:
        # An unreadable SKILL.md (e.g. PermissionError) is a skill-dir problem, not a
        # crash: report it as an issue so callers handle it as a finding instead of a
        # raw traceback escaping through build_skill_manifest/install.
        issues.append(_issue("skill-md-unreadable", f"SKILL.md could not be read: {e}", skill_file))
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
        if should_skip_skill_dir(child):
            continue
        results.append(validate_skill_dir(child))
    return results


def should_skip_skill_dir(path: Path) -> bool:
    """Return whether a skills-root child is metadata/reference-only."""
    return path.name.startswith(".") or path.name.startswith("_") or not path.is_dir()


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
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append(
                _issue("skill-frontmatter-field-required", f"frontmatter field '{field}' is required", skill_file)
            )
        elif not isinstance(value, str):
            issues.append(
                _issue(
                    "skill-frontmatter-field-type",
                    f"frontmatter field '{field}' must be a non-empty string; " f"got {type(value).__name__}={value!r}",
                    skill_file,
                )
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


def _validate_no_symlinks(skill_dir: Path, issues: list[SkillValidationIssue]) -> None:
    for root, dir_names, file_names in os.walk(skill_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        dir_names.sort()
        file_names.sort()
        for name in dir_names + file_names:
            path = root_path / name
            if path.is_symlink():
                issues.append(_issue("skill-symlink-not-allowed", "skill directories must not contain symlinks", path))
        dir_names[:] = [name for name in dir_names if not (root_path / name).is_symlink()]


def _issue(code: str, message: str, path: Path) -> SkillValidationIssue:
    return SkillValidationIssue(code=code, message=message, path=str(path))
