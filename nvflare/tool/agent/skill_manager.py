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
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources, util
from pathlib import Path
from typing import Optional

import nvflare
from nvflare.tool.agent.skill_manifest import (
    IGNORED_SKILL_FILE_NAMES,
    MANIFEST_FILE_NAME,
    SHARED_SKILL_REFERENCE_DIR,
    build_skill_manifest,
    load_manifest,
    skill_tree_hash,
)

INSTALL_MANIFEST_FILE_NAME = ".nvflare_skill_install.json"
SUPPORTED_AGENT_TARGETS = ("codex", "claude")
BUNDLED_SKILLS_PACKAGE = "nvflare.tool.agent.bundled_skills"
DEFAULT_INSTALL_LOCK_TTL_SECONDS = 300
INSTALL_LOCK_TIMESTAMP_FILE_NAME = "created_at_ns"


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
    try:
        _sync_shared_references(source, target)
    except Exception as e:
        error = _install_error(SHARED_SKILL_REFERENCE_DIR, e)
        plan["errors"].append(error)
        plan["applied"] = False
        return plan
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
    available = source.manifest.get("skills", [])
    available_names = {skill["name"] for skill in available}

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
            elif child.name in available_names:
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
        "available": available,
        "installed": installed,
        "conflicts": conflicts,
    }


def _install_plan(
    source: SkillSource, skills: list[dict], target: Path, *, agent: str, requested_skill: Optional[str]
) -> dict:
    planned_skills = []
    conflicts = []
    for skill in skills:
        source_skill_dir = source.root / skill["relative_path"]
        target_skill_dir = target / skill["name"]
        source_symlink = _first_symlink_in_tree(source_skill_dir)
        # The dry-run plan is advisory: source symlinks are checked again in
        # _stage_skill because the source tree could change before apply.
        # version_delta: new, unknown external state, blocked local edit, same, or update.
        entry = {
            "name": skill["name"],
            "skill_version": skill.get("skill_version"),
            "source_hash": skill["source_hash"],
            "relative_path": skill["relative_path"],
            "target_path": str(target_skill_dir),
            "files": [] if source_symlink else _files_to_copy(source_skill_dir, target_skill_dir),
            "version_delta": "new",
        }
        if source_symlink:
            entry["action"] = "skip"
            entry["conflict"] = "source_symlink_detected"
            entry["version_delta"] = "blocked"
            entry["source_issue"] = {
                "code": "source_symlink_detected",
                "message": "source skill directory contains a symlink",
                "source_path": str(source_skill_dir),
                "symlink_path": str(source_symlink),
            }
            conflicts.append(_source_conflict(skill["name"], source_skill_dir, source_symlink))
        elif not target_skill_dir.exists():
            entry["action"] = "copy"
        else:
            target_symlink = _first_symlink_in_tree(target_skill_dir)
            if target_symlink:
                entry["action"] = "skip"
                entry["conflict"] = "target_symlink_detected"
                entry["version_delta"] = "blocked"
                entry["target_issue"] = {
                    "code": "target_symlink_detected",
                    "message": "target skill directory contains a symlink",
                    "target_path": str(target_skill_dir),
                    "symlink_path": str(target_symlink),
                }
                conflicts.append(_target_symlink_conflict(skill["name"], target_skill_dir, target_symlink))
            else:
                install_manifest = _read_install_manifest(target_skill_dir)
                if not install_manifest or install_manifest.get("managed_by") != "nvflare":
                    entry["action"] = "skip"
                    entry["conflict"] = "external_install_detected"
                    entry["version_delta"] = "unknown"
                    conflicts.append(_conflict(skill["name"], "external_install_detected", target_skill_dir))
                else:
                    try:
                        installed_source_hash = skill_tree_hash(
                            target_skill_dir, exclude_names={INSTALL_MANIFEST_FILE_NAME}
                        )
                    except ValueError as e:
                        installed_source_hash = None
                        entry["target_issue"] = {
                            "code": "local_modifications_detected",
                            "message": str(e),
                            "target_path": str(target_skill_dir),
                        }
                    if installed_source_hash != install_manifest.get("source_hash"):
                        entry["action"] = "skip"
                        entry["conflict"] = "local_modifications_detected"
                        entry["version_delta"] = "blocked"
                        conflicts.append(_conflict(skill["name"], "local_modifications_detected", target_skill_dir))
                    elif install_manifest.get("source_hash") == skill["source_hash"]:
                        entry["action"] = "skip"
                        entry["reason"] = "already_installed"
                        entry["version_delta"] = "same"
                    else:
                        backup_path = _backup_path(target, skill["name"])
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
    with _skill_install_lock(target_dir):
        with tempfile.TemporaryDirectory(prefix=f".{target_dir.name}.", dir=target_dir.parent) as temp_root:
            temp_skill_dir = Path(temp_root) / target_dir.name
            _stage_skill(source_dir, temp_skill_dir, plan_entry, source, installed_path=target_dir)
            _publish_staged_skill(temp_skill_dir, target_dir)


def _replace_skill(
    source_dir: Path, target_dir: Path, backup_path: Path, plan_entry: dict, source: SkillSource
) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with _skill_install_lock(target_dir):
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


def _sync_shared_references(source: SkillSource, target: Path) -> None:
    source_shared = source.root / SHARED_SKILL_REFERENCE_DIR
    if not source_shared.is_dir():
        return
    source_hash = skill_tree_hash(source_shared)
    target_shared = target / SHARED_SKILL_REFERENCE_DIR
    plan_entry = {"name": SHARED_SKILL_REFERENCE_DIR, "source_hash": source_hash}
    if not target_shared.exists():
        _copy_skill(source_shared, target_shared, plan_entry, source)
        return
    if target_shared.is_symlink():
        raise ValueError(f"shared reference target must not be a symlink: {target_shared}")
    install_manifest = _read_install_manifest(target_shared)
    if not install_manifest or install_manifest.get("managed_by") != "nvflare":
        raise FileExistsError(f"shared reference target is not managed by nvflare: {target_shared}")
    installed_source_hash = skill_tree_hash(target_shared, exclude_names={INSTALL_MANIFEST_FILE_NAME})
    if installed_source_hash == source_hash and install_manifest.get("source_hash") == source_hash:
        return
    backup_path = _backup_path(target, target_shared.name)
    _replace_skill(source_shared, target_shared, backup_path, plan_entry, source)


def _backup_path(target: Path, skill_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return target / ".nvflare_bak" / f"{timestamp}-{time.time_ns()}" / skill_name


@contextmanager
def _skill_install_lock(target_dir: Path):
    lock_dir = target_dir.parent / f".{target_dir.name}.install.lock"
    try:
        _create_lock_dir(lock_dir)
    except FileExistsError as e:
        if _lock_dir_is_stale(lock_dir):
            shutil.rmtree(lock_dir)
            try:
                _create_lock_dir(lock_dir)
            except FileExistsError as retry_error:
                raise FileExistsError(
                    f"target skill directory is already being installed: {target_dir}"
                ) from retry_error
        else:
            raise FileExistsError(f"target skill directory is already being installed: {target_dir}") from e
    try:
        yield
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def _lock_ttl_seconds() -> int:
    value = os.environ.get("NVFLARE_AGENT_SKILL_INSTALL_LOCK_TTL_SECONDS")
    if value is None:
        return DEFAULT_INSTALL_LOCK_TTL_SECONDS
    try:
        return max(0, int(value))
    except ValueError:
        return DEFAULT_INSTALL_LOCK_TTL_SECONDS


def _lock_dir_is_stale(lock_dir: Path) -> bool:
    if lock_dir.is_symlink() or not lock_dir.is_dir():
        return False
    ttl_seconds = _lock_ttl_seconds()
    if ttl_seconds <= 0:
        return False
    lock_started_at = _read_lock_started_at(lock_dir)
    if lock_started_at is not None:
        return time.time() - lock_started_at > ttl_seconds
    try:
        age_seconds = time.time() - lock_dir.stat().st_mtime
    except OSError:
        return False
    return age_seconds > ttl_seconds


def _write_lock_timestamp(lock_dir: Path) -> None:
    (lock_dir / INSTALL_LOCK_TIMESTAMP_FILE_NAME).write_text(str(time.time_ns()), encoding="utf-8")


def _create_lock_dir(lock_dir: Path) -> None:
    lock_dir.mkdir()
    try:
        _write_lock_timestamp(lock_dir)
    except Exception:
        shutil.rmtree(lock_dir, ignore_errors=True)
        raise


def _read_lock_started_at(lock_dir: Path) -> float | None:
    timestamp_path = lock_dir / INSTALL_LOCK_TIMESTAMP_FILE_NAME
    try:
        text = timestamp_path.read_text(encoding="utf-8").strip()
        return int(text) / 1_000_000_000
    except (OSError, ValueError):
        return None


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
    os.rename(staged_dir, target_dir)


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
    if any(part == ".." for part in target.parts):
        raise ValueError(f"agent skill target must not contain parent directory traversal: {target}")
    logical = Path(os.path.abspath(os.path.normpath(str(target))))
    symlink = _first_disallowed_target_symlink_component(logical)
    if symlink is not None:
        raise ValueError(f"agent skill target must not contain symlink components: {symlink}")
    return target.resolve(strict=False)


def _target_system_symlink_aliases() -> tuple[Path, ...]:
    aliases = {Path("/tmp")}
    for value in (os.environ.get("TMPDIR"), tempfile.gettempdir()):
        if value:
            aliases.add(Path(os.path.abspath(os.path.normpath(value))))
    return tuple(sorted(aliases, key=lambda item: str(item)))


def _is_allowed_system_target_symlink(path: Path) -> bool:
    for alias in _target_system_symlink_aliases():
        try:
            alias.relative_to(path)
            return True
        except ValueError:
            pass
    return False


def _first_disallowed_target_symlink_component(path: Path) -> Optional[Path]:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for part in parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            if _is_allowed_system_target_symlink(current):
                continue
            return current
    return None


def _first_symlink_in_tree(root_dir: Path) -> Optional[Path]:
    if root_dir.is_symlink():
        return root_dir
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


def _source_conflict(skill_name: str, source_path: Path, symlink_path: Path) -> dict:
    return {
        "skill": skill_name,
        "code": "source_symlink_detected",
        "message": "source skill directory contains a symlink",
        "source_path": str(source_path),
        "symlink_path": str(symlink_path),
    }


def _target_symlink_conflict(skill_name: str, target_path: Path, symlink_path: Path) -> dict:
    return {
        "skill": skill_name,
        "code": "target_symlink_detected",
        "message": "target skill directory contains a symlink",
        "target_path": str(target_path),
        "symlink_path": str(symlink_path),
    }


def _source_checkout_root() -> Optional[Path]:
    spec = util.find_spec("nvflare")
    if spec is None or not spec.submodule_search_locations:
        return None
    repo_root = Path(next(iter(spec.submodule_search_locations))).resolve().parent
    source_root = repo_root / "skills"
    if source_root.is_dir() and (repo_root / "pyproject.toml").is_file():
        return source_root
    return None
