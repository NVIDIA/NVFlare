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

"""Read-only discovery of installed agent skills."""

from pathlib import Path
from typing import Optional

SKILL_FILE_NAME = "SKILL.md"
MAX_INSTALLED_SKILLS = 200
MAX_SKILL_FRONTMATTER_BYTES = 64 * 1024
_PROJECT_SKILL_DIRS = (".claude/skills", ".agents/skills")
_GLOBAL_SKILL_DIRS = ("~/.claude/skills", "~/.codex/skills")


def discover_installed_skills(target: Path) -> list[dict]:
    """Discover installed skills from known agent skill dirs."""
    skills: list[dict] = []
    seen_names: set[str] = set()
    for base, scope in _installed_skill_search_roots(target):
        for skill_dir in _iter_skill_dirs(base):
            if len(skills) >= MAX_INSTALLED_SKILLS:
                return skills
            skill_file = skill_dir / SKILL_FILE_NAME
            if skill_file.is_symlink() or not skill_file.is_file():
                continue
            frontmatter = _read_skill_frontmatter(skill_file)
            if frontmatter is None:
                continue
            name = frontmatter.get("name") or skill_dir.name
            if name in seen_names:
                continue
            seen_names.add(name)
            skills.append(
                {
                    "name": name,
                    "description": frontmatter.get("description", ""),
                    "scope": scope,
                    "source": _installed_skill_source(skill_dir),
                }
            )
    return skills


def _installed_skill_search_roots(target: Path) -> list[tuple[Path, str]]:
    roots: list[tuple[Path, str]] = []
    project_root = _project_root_for(target)
    if project_root is not None:
        for rel in _PROJECT_SKILL_DIRS:
            roots.append((project_root / rel, "project"))
    home = Path.home()
    for rel in _GLOBAL_SKILL_DIRS:
        roots.append((Path(rel).expanduser() if rel.startswith("~") else home / rel, "global"))
    return roots


def _project_root_for(target: Path) -> Optional[Path]:
    try:
        start = target if target.is_dir() and not target.is_symlink() else target.parent
        start = start.resolve()
        cwd = Path.cwd().resolve()
    except OSError:
        return None
    candidates = [start, *start.parents]
    for candidate in candidates:
        for rel in _PROJECT_SKILL_DIRS:
            if (candidate / rel).is_dir():
                return candidate
        if candidate == cwd:
            break
    return cwd


def _iter_skill_dirs(base: Path):
    if base.is_symlink() or not base.is_dir():
        return
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError:
        return
    for child in children:
        if child.is_symlink() or not child.is_dir():
            continue
        yield child


def _installed_skill_source(skill_dir: Path) -> str:
    try:
        return str(skill_dir.resolve(strict=False))
    except OSError:
        return str(skill_dir)


def _read_skill_frontmatter(skill_file: Path) -> Optional[dict]:
    """Parse the leading YAML frontmatter block for name/description only."""
    try:
        if skill_file.stat().st_size > MAX_SKILL_FRONTMATTER_BYTES:
            return None
        text = skill_file.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line[:1] in (" ", "\t") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in ("name", "description"):
            result[key] = _strip_scalar(value.strip())
    return result


def _strip_scalar(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value
