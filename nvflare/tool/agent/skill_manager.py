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

"""Native install/list support for NVFLARE-owned agent skills."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Optional

import nvflare
from nvflare.tool.agent.skill_manifest import (
    IGNORED_SKILL_FILE_NAMES,
    MANIFEST_FILE_NAME,
    build_skill_manifest,
    load_manifest,
    skill_tree_hash,
)

INSTALL_MANIFEST_FILE_NAME = ".nvflare_skill_install.json"
SUPPORTED_AGENT_TARGETS = ("codex", "claude")
BUNDLED_SKILLS_PACKAGE = "nvflare.tool.agent.bundled_skills"


@dataclass(frozen=True)
class SkillSource:
    source_type: str
    root: Path
    manifest: dict


def resolve_agent_target_dir(
    agent: str, *, target_dir: Optional[Path | str] = None, env: Optional[dict] = None
) -> Path:
    """Resolve a named agent target to its skill installation directory."""
    if target_dir:
        return _resolve_target_override(target_dir)

    env_map = env or os.environ
    if agent == "codex":
        codex_home = env_map.get("CODEX_HOME")
        if codex_home:
            return Path(codex_home).expanduser() / "skills"
        return Path.home() / ".codex" / "skills"
    if agent == "claude":
        return Path.home() / ".claude" / "skills"
    raise ValueError(f"unsupported agent target: {agent}")


def find_skill_source() -> SkillSource:
    """Find skills from an editable/source checkout first, then from the installed package bundle."""
    source_root = _source_checkout_root()
    if source_root:
        return SkillSource(
            source_type="editable",
            root=source_root,
            manifest=build_skill_manifest(source_root, source_type="editable", nvflare_version=nvflare.__version__),
        )

    bundle_root = Path(str(resources.files(BUNDLED_SKILLS_PACKAGE)))
    manifest_path = bundle_root / MANIFEST_FILE_NAME
    manifest = (
        load_manifest(manifest_path)
        if manifest_path.is_file()
        else build_skill_manifest(bundle_root, source_type="wheel")
    )
    return SkillSource(source_type="wheel", root=bundle_root, manifest=manifest)


def install_skills(
    *,
    agent: str,
    skill_name: Optional[str] = None,
    dry_run: bool = False,
    target_dir: Optional[Path | str] = None,
    source: Optional[SkillSource] = None,
) -> dict:
    """Plan or apply a native NVFLARE skill installation."""
    source = source or find_skill_source()
    target = resolve_agent_target_dir(agent, target_dir=target_dir)
    selected, missing = _select_skills(source.manifest, skill_name)
    plan = _install_plan(source, selected, target, agent=agent, requested_skill=skill_name)
    plan["missing"] = missing
    plan["applied"] = False

    if dry_run or missing:
        return plan

    target.mkdir(parents=True, exist_ok=True)
    for entry in plan["skills"]:
        try:
            if entry["action"] == "copy":
                _copy_skill(source.root / entry["relative_path"], Path(entry["target_path"]), entry, source)
                entry["status"] = "installed"
            elif entry["action"] == "replace":
                _replace_skill(
                    source.root / entry["relative_path"],
                    Path(entry["target_path"]),
                    Path(entry["backup_path"]),
                    entry,
                    source,
                )
                entry["status"] = "replaced"
            else:
                entry["status"] = "skipped"
        except Exception as e:
            error = _install_error(entry["name"], e)
            entry["status"] = "failed"
            entry["error"] = error
            plan["errors"].append(error)
    plan["applied"] = not plan["errors"]
    return plan


def list_skills(*, agent: str, target_dir: Optional[Path | str] = None, source: Optional[SkillSource] = None) -> dict:
    """List available packaged skills and installed managed skills for an agent target."""
    source = source or find_skill_source()
    target = resolve_agent_target_dir(agent, target_dir=target_dir)
    installed = []
    conflicts = []

    if target.is_dir():
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            if child.name.startswith(".") or not child.is_dir():
                continue
            install_manifest = _read_install_manifest(child)
            if install_manifest and install_manifest.get("managed_by") == "nvflare":
                installed.append(
                    {
                        "name": install_manifest.get("name", child.name),
                        "skill_version": install_manifest.get("skill_version"),
                        "source_hash": install_manifest.get("source_hash"),
                        "target_path": str(child),
                        "source_type": install_manifest.get("source_type"),
                    }
                )
            else:
                conflicts.append(
                    {
                        "skill": child.name,
                        "code": "external_install_detected",
                        "message": "target skill directory is not managed by nvflare",
                        "target_path": str(child),
                    }
                )

    return {
        "agent": agent,
        "target_path": str(target),
        "source": _source_summary(source),
        "available": source.manifest.get("skills", []),
        "installed": installed,
        "conflicts": conflicts,
    }


def _install_plan(
    source: SkillSource, skills: list[dict], target: Path, *, agent: str, requested_skill: Optional[str]
) -> dict:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    planned_skills = []
    conflicts = []
    for skill in skills:
        target_skill_dir = target / skill["name"]
        # version_delta: new, unknown external state, blocked local edit, same, or update.
        entry = {
            "name": skill["name"],
            "skill_version": skill.get("skill_version"),
            "source_hash": skill["source_hash"],
            "relative_path": skill["relative_path"],
            "target_path": str(target_skill_dir),
            "files": _files_to_copy(source.root / skill["relative_path"], target_skill_dir),
            "version_delta": "new",
        }
        if not target_skill_dir.exists():
            entry["action"] = "copy"
        else:
            install_manifest = _read_install_manifest(target_skill_dir)
            if not install_manifest or install_manifest.get("managed_by") != "nvflare":
                entry["action"] = "skip"
                entry["conflict"] = "external_install_detected"
                entry["version_delta"] = "unknown"
                conflicts.append(_conflict(skill["name"], "external_install_detected", target_skill_dir))
            elif skill_tree_hash(target_skill_dir, exclude_names={INSTALL_MANIFEST_FILE_NAME}) != install_manifest.get(
                "source_hash"
            ):
                entry["action"] = "skip"
                entry["conflict"] = "local_modifications_detected"
                entry["version_delta"] = "blocked"
                conflicts.append(_conflict(skill["name"], "local_modifications_detected", target_skill_dir))
            elif install_manifest.get("source_hash") == skill["source_hash"]:
                entry["action"] = "skip"
                entry["reason"] = "already_installed"
                entry["version_delta"] = "same"
            else:
                backup_path = target / ".nvflare_bak" / now / skill["name"]
                entry["action"] = "replace"
                entry["backup_path"] = str(backup_path)
                entry["version_delta"] = "update"
        planned_skills.append(entry)

    return {
        "agent": agent,
        "target_path": str(target),
        "requested_skill": requested_skill,
        "source": _source_summary(source),
        "available": source.manifest.get("skills", []),
        "skills": planned_skills,
        "conflicts": conflicts,
        "errors": [],
        "deprecated_skills_skipped": [],
    }


def _copy_skill(source_dir: Path, target_dir: Path, plan_entry: dict, source: SkillSource) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target_dir.name}.", dir=target_dir.parent) as temp_root:
        temp_skill_dir = Path(temp_root) / target_dir.name
        _stage_skill(source_dir, temp_skill_dir, plan_entry, source, installed_path=target_dir)
        if target_dir.exists():
            raise FileExistsError(f"target skill directory already exists: {target_dir}")
        _publish_staged_skill(temp_skill_dir, target_dir)


def _replace_skill(
    source_dir: Path, target_dir: Path, backup_path: Path, plan_entry: dict, source: SkillSource
) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target_dir.name}.", dir=target_dir.parent) as temp_root:
        temp_skill_dir = Path(temp_root) / target_dir.name
        _stage_skill(source_dir, temp_skill_dir, plan_entry, source, installed_path=target_dir)
        if not target_dir.exists():
            raise FileNotFoundError(f"target skill directory no longer exists: {target_dir}")
        if backup_path.exists():
            raise FileExistsError(f"backup skill directory already exists: {backup_path}")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(target_dir, backup_path)
        try:
            _publish_staged_skill(temp_skill_dir, target_dir)
        except Exception as publish_error:
            if not target_dir.exists() and backup_path.exists():
                try:
                    shutil.move(backup_path, target_dir)
                    _remove_empty_dir(backup_path.parent)
                except Exception as recovery_error:
                    publish_error.recovery_error = recovery_error
            raise


def _remove_empty_dir(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _install_error(skill_name: str, error: Exception) -> dict:
    result = {
        "skill": skill_name,
        "code": "skill_install_failed",
        "type": type(error).__name__,
        "message": str(error),
    }
    recovery_error = getattr(error, "recovery_error", None)
    if recovery_error is not None:
        result["recovery_error"] = {
            "type": type(recovery_error).__name__,
            "message": str(recovery_error),
        }
    return result


def _publish_staged_skill(staged_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        raise FileExistsError(f"target skill directory already exists: {target_dir}")
    os.replace(staged_dir, target_dir)


def _stage_skill(
    source_dir: Path, staged_dir: Path, plan_entry: dict, source: SkillSource, *, installed_path: Path
) -> None:
    symlink = _first_symlink_in_tree(source_dir)
    if symlink:
        raise ValueError(f"skill source must not contain symlinks: {symlink.relative_to(source_dir).as_posix()}")
    shutil.copytree(source_dir, staged_dir, ignore=shutil.ignore_patterns(*IGNORED_SKILL_FILE_NAMES))
    manifest = {
        "schema_version": "1",
        "managed_by": "nvflare",
        "name": plan_entry["name"],
        "skill_version": plan_entry.get("skill_version"),
        "nvflare_version": nvflare.__version__,
        "source_type": source.source_type,
        "source_hash": plan_entry["source_hash"],
        "installed_paths": [str(installed_path)],
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    (staged_dir / INSTALL_MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _select_skills(manifest: dict, skill_name: Optional[str]) -> tuple[list[dict], list[str]]:
    skills = manifest.get("skills", [])
    if not skill_name:
        return skills, []
    selected = [skill for skill in skills if skill.get("name") == skill_name]
    return selected, [] if selected else [skill_name]


def _source_summary(source: SkillSource) -> dict:
    return {
        "type": source.source_type,
        "root": str(source.root),
        "nvflare_version": source.manifest.get("nvflare_version"),
        "manifest_schema_version": source.manifest.get("schema_version"),
        "skill_count": len(source.manifest.get("skills", [])),
        "findings": source.manifest.get("findings", []),
    }


def _resolve_target_override(target_dir: Path | str) -> Path:
    target = Path(target_dir).expanduser()
    symlink = _first_symlink_component(target)
    if symlink:
        raise ValueError(f"agent skill target must not contain symlink components: {symlink}")
    return target.resolve(strict=False)


def _first_symlink_component(path: Path) -> Optional[Path]:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        if part in ("", "."):
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        if current.is_symlink():
            return current
    return None


def _first_symlink_in_tree(root_dir: Path) -> Optional[Path]:
    for root, dir_names, file_names in os.walk(root_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        dir_names.sort()
        file_names.sort()
        for name in dir_names + file_names:
            path = root_path / name
            if path.is_symlink():
                return path
        dir_names[:] = [name for name in dir_names if not (root_path / name).is_symlink()]
    return None


def _files_to_copy(source_dir: Path, target_dir: Path) -> list[dict]:
    files = []
    for root, dir_names, file_names in os.walk(source_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        dir_names.sort()
        file_names.sort()
        dir_names[:] = [name for name in dir_names if name != "__pycache__" and not (root_path / name).is_symlink()]
        for file_name in file_names:
            file_path = root_path / file_name
            if file_path.is_symlink():
                continue
            if file_path.suffix in {".pyc", ".pyo"}:
                continue
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(source_dir)
            files.append({"source": str(file_path), "target": str(target_dir / rel_path)})
    return files


def _read_install_manifest(skill_dir: Path) -> Optional[dict]:
    manifest_path = skill_dir / INSTALL_MANIFEST_FILE_NAME
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _conflict(skill_name: str, code: str, target_path: Path) -> dict:
    messages = {
        "external_install_detected": "target skill directory is not managed by nvflare",
        "local_modifications_detected": "managed skill content differs from its install manifest",
    }
    return {
        "skill": skill_name,
        "code": code,
        "message": messages.get(code, code),
        "target_path": str(target_path),
    }


def _source_checkout_root() -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    source_root = repo_root / "skills"
    if source_root.is_dir() and (repo_root / "pyproject.toml").is_file() and (repo_root / "setup.py").is_file():
        return source_root
    return None
