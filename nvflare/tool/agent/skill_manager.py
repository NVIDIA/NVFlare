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
import re
import secrets
import shutil
import stat
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
    MANIFEST_FILE_NAME,
    SHARED_SKILL_PACKAGING_EXCLUDE_NAMES,
    SHARED_SKILL_REFERENCE_DIR,
    SKILL_PACKAGING_EXCLUDE_NAMES,
    build_skill_manifest,
    load_manifest,
    skill_references_shared,
    skill_tree_hash,
    validate_manifest,
)

INSTALL_MANIFEST_FILE_NAME = ".nvflare_skill_install.json"
SUPPORTED_AGENT_TARGETS = ("codex", "claude")
BUNDLED_SKILLS_PACKAGE = "nvflare.tool.agent.bundled_skills"
DEFAULT_INSTALL_LOCK_TTL_SECONDS = 300
INSTALL_LOCK_TIMESTAMP_FILE_NAME = "created_at_ns"
INSTALL_LOCK_OWNER_FILE_NAME = "owner.json"
INSTALL_LOCK_LEASE_FILE_NAME = "lease"
INSTALL_LOCK_LEASE_VERSION = 1
MAX_INSTALL_LOCK_METADATA_BYTES = 16 * 1024
SHARED_INSTALL_ROOT_DIR = ".nvflare-shared"
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SkillSource:
    source_type: str
    root: Path
    manifest: dict


def resolve_agent_target_dir(
    agent: str, *, target_dir: Optional[Path | str] = None, env: Optional[dict] = None
) -> Path:
    """Resolve a named agent target to its skill installation directory.

    Explicit write targets, including CODEX_HOME-derived targets, use the same
    no-traversal/no-symlink policy as --target. The only symlink exception is
    the platform temp-directory alias handled in _resolve_target_override.
    """
    if target_dir:
        return _resolve_target_override(target_dir)

    env_map = env or os.environ
    if agent == "codex":
        codex_home = env_map.get("CODEX_HOME")
        if codex_home:
            return _resolve_target_override(Path(codex_home).expanduser() / "skills")
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

    bundle_root = _bundled_skills_root()
    manifest_path = bundle_root / MANIFEST_FILE_NAME
    manifest = (
        load_manifest(manifest_path)
        if manifest_path.is_file()
        else build_skill_manifest(bundle_root, source_type="wheel")
    )
    return SkillSource(source_type="wheel", root=bundle_root, manifest=manifest)


def _bundled_skills_root() -> Path:
    bundle_root = Path(str(resources.files(BUNDLED_SKILLS_PACKAGE)))
    if not bundle_root.is_dir():
        raise FileNotFoundError(
            f"bundled agent skills must be available from an unpacked filesystem package: {BUNDLED_SKILLS_PACKAGE}"
        )
    return bundle_root


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
    validate_manifest(source.manifest)
    target = resolve_agent_target_dir(agent, target_dir=target_dir)
    selected, missing = _select_skills(source.manifest, skill_name)
    plan = _install_plan(source, selected, target, agent=agent, requested_skill=skill_name)
    plan["missing"] = missing
    plan["applied"] = False

    if dry_run or missing:
        return plan

    try:
        target_identity = _prepare_install_target(target)
    except (OSError, ValueError) as e:
        error = _install_error(str(target), e)
        plan["errors"].append(error)
        plan["applied"] = False
        return plan
    legacy_shared_plan = plan.get("legacy_shared")
    if legacy_shared_plan and legacy_shared_plan.get("conflict"):
        error = {
            "skill": SHARED_SKILL_REFERENCE_DIR,
            "code": "legacy_shared_migration_required",
            "type": "SkillInstallConflict",
            "message": legacy_shared_plan["message"],
        }
        plan["errors"].append(error)
        for entry in plan["skills"]:
            entry["status"] = "blocked"
        return plan
    shared_plan = plan.get("shared")
    _assert_directory_identity(target, target_identity)
    if shared_plan and shared_plan.get("conflict"):
        error = {
            "skill": SHARED_SKILL_REFERENCE_DIR,
            "code": "shared_dependency_unavailable",
            "type": "SkillInstallConflict",
            "message": shared_plan.get("message", "versioned shared resources are unavailable"),
        }
        plan["errors"].append(error)
        for entry in plan["skills"]:
            if entry.get("shared_source_hash"):
                entry["status"] = "blocked"
    else:
        try:
            _sync_shared_references(source, target, shared_plan)
            if shared_plan:
                shared_plan["status"] = "installed" if shared_plan["action"] == "copy" else "skipped"
        except Exception as e:
            error = _install_error(SHARED_SKILL_REFERENCE_DIR, e)
            plan["errors"].append(error)
            if shared_plan:
                shared_plan["status"] = "failed"
                shared_plan["error"] = error
            for entry in plan["skills"]:
                if entry.get("shared_source_hash"):
                    entry["status"] = "blocked"
    for entry in plan["skills"]:
        if entry.get("status") == "blocked":
            continue
        try:
            _assert_directory_identity(target, target_identity)
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
    if not plan["errors"] and legacy_shared_plan and legacy_shared_plan.get("action") == "backup":
        try:
            _assert_directory_identity(target, target_identity)
            _backup_legacy_shared(legacy_shared_plan)
            legacy_shared_plan["status"] = "backed_up"
        except Exception as e:
            error = _install_error(SHARED_SKILL_REFERENCE_DIR, e)
            legacy_shared_plan["status"] = "failed"
            legacy_shared_plan["error"] = error
            plan["errors"].append(error)
    plan["applied"] = not plan["errors"]
    return plan


def list_skills(*, agent: str, target_dir: Optional[Path | str] = None, source: Optional[SkillSource] = None) -> dict:
    """List available packaged skills and installed managed skills for an agent target."""
    source = source or find_skill_source()
    validate_manifest(source.manifest)
    target = resolve_agent_target_dir(agent, target_dir=target_dir)
    installed = []
    conflicts = []
    errors = []
    available = source.manifest.get("skills", [])
    available_names = {skill["name"] for skill in available}
    available_category = {skill["name"]: skill.get("category") for skill in available}

    if target.is_dir():
        try:
            children = sorted(target.iterdir(), key=lambda p: p.name)
        except OSError as e:
            errors.append(_list_error(str(target), e))
            children = []
        for child in children:
            try:
                if child.name.startswith("."):
                    continue
                child_stat = child.lstat()
                if stat.S_ISLNK(child_stat.st_mode):
                    if child.name in available_names:
                        conflicts.append(_target_symlink_conflict(child.name, child, child))
                    continue
                if not stat.S_ISDIR(child_stat.st_mode):
                    continue
            except OSError as e:
                errors.append(_list_error(str(child), e))
                continue
            install_manifest = _read_install_manifest(child)
            if install_manifest and install_manifest.get("managed_by") == "nvflare":
                installed_name = install_manifest.get("name", child.name)
                installed.append(
                    {
                        "name": installed_name,
                        "skill_version": install_manifest.get("skill_version"),
                        "source_hash": install_manifest.get("source_hash"),
                        "shared_source_hash": install_manifest.get("shared_source_hash"),
                        "category": available_category.get(installed_name),
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
        "errors": errors,
    }


def _install_plan(
    source: SkillSource, skills: list[dict], target: Path, *, agent: str, requested_skill: Optional[str]
) -> dict:
    planned_skills = []
    conflicts = []
    dependencies = {}
    required_shared_hashes = set()
    for skill in skills:
        explicit_hash = skill.get("shared_source_hash")
        try:
            inferred_reference = skill_references_shared(source.root / skill["relative_path"])
        except (OSError, ValueError):
            # The normal source-safety planning below emits the structured
            # conflict; dependency inference must not mask that finding.
            inferred_reference = False
        references_shared = bool(explicit_hash) or inferred_reference
        dependencies[skill["name"]] = references_shared
        if explicit_hash:
            required_shared_hashes.add(explicit_hash)
    shared_plan = (
        _shared_install_plan(source, target, required_hashes=required_shared_hashes)
        if any(dependencies.values())
        else None
    )
    if shared_plan and shared_plan.get("conflict"):
        conflicts.append(
            {
                "skill": SHARED_SKILL_REFERENCE_DIR,
                "code": shared_plan["conflict"],
                "message": shared_plan["message"],
                "target_path": shared_plan["target_path"],
            }
        )
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
            "shared_source_hash": (
                skill.get("shared_source_hash") or (shared_plan or {}).get("source_hash")
                if dependencies[skill["name"]]
                else None
            ),
            "relative_path": skill["relative_path"],
            "target_path": str(target_skill_dir),
            "files": (
                []
                if source_symlink
                else _files_to_copy(source_skill_dir, target_skill_dir, exclude_names=SKILL_PACKAGING_EXCLUDE_NAMES)
            ),
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
                    except (OSError, ValueError) as e:
                        installed_source_hash = None
                        entry["target_issue"] = {
                            "code": "local_modifications_detected",
                            "message": str(e),
                            "error_type": type(e).__name__,
                            "target_path": str(target_skill_dir),
                        }
                    managed_installed_hash = install_manifest.get("installed_hash") or install_manifest.get(
                        "source_hash"
                    )
                    if installed_source_hash != managed_installed_hash:
                        entry["action"] = "skip"
                        entry["conflict"] = "local_modifications_detected"
                        entry["version_delta"] = "blocked"
                        conflicts.append(_conflict(skill["name"], "local_modifications_detected", target_skill_dir))
                    elif install_manifest.get("source_hash") == skill["source_hash"] and install_manifest.get(
                        "shared_source_hash"
                    ) == entry.get("shared_source_hash"):
                        entry["action"] = "skip"
                        entry["reason"] = "already_installed"
                        entry["version_delta"] = "same"
                    else:
                        backup_path = _backup_path(target, skill["name"])
                        entry["action"] = "replace"
                        entry["backup_path"] = str(backup_path)
                        entry["version_delta"] = "update"
        planned_skills.append(entry)

    legacy_shared_plan = _legacy_shared_install_plan(target, planned_skills)
    if legacy_shared_plan and legacy_shared_plan.get("conflict"):
        conflicts.append(
            {
                "skill": SHARED_SKILL_REFERENCE_DIR,
                "code": legacy_shared_plan["conflict"],
                "message": legacy_shared_plan["message"],
                "target_path": legacy_shared_plan["target_path"],
            }
        )

    return {
        "agent": agent,
        "target_path": str(target),
        "requested_skill": requested_skill,
        "source": _source_summary(source),
        "available": source.manifest.get("skills", []),
        "skills": planned_skills,
        "shared": shared_plan,
        "legacy_shared": legacy_shared_plan,
        "conflicts": conflicts,
        "errors": [],
        "deprecated_skills_skipped": [],
    }


def _legacy_shared_install_plan(target: Path, planned_skills: list[dict]) -> Optional[dict]:
    """Plan safe retirement of the old discoverable shared-skill directory."""

    legacy_dir = target / SHARED_SKILL_REFERENCE_DIR
    try:
        legacy_mode = legacy_dir.lstat().st_mode
    except FileNotFoundError:
        return None
    except OSError as e:
        return {
            "action": "skip",
            "conflict": "legacy_shared_unreadable",
            "message": f"legacy shared skill cannot be inspected: {e}",
            "target_path": str(legacy_dir),
        }

    entry = {"target_path": str(legacy_dir), "action": "backup"}
    if stat.S_ISLNK(legacy_mode) or not stat.S_ISDIR(legacy_mode):
        entry.update(
            action="skip",
            conflict="target_symlink_detected" if stat.S_ISLNK(legacy_mode) else "external_install_detected",
            message=f"legacy shared skill is not a safe managed directory: {legacy_dir}",
        )
        return entry

    install_manifest = _read_install_manifest(legacy_dir)
    if not install_manifest or install_manifest.get("managed_by") != "nvflare":
        entry.update(
            action="skip",
            conflict="external_install_detected",
            message=f"legacy shared skill is not managed by nvflare: {legacy_dir}",
        )
        return entry
    try:
        installed_hash = skill_tree_hash(legacy_dir, exclude_names={INSTALL_MANIFEST_FILE_NAME})
    except (OSError, ValueError) as e:
        installed_hash = None
        entry["target_issue"] = {"type": type(e).__name__, "message": str(e)}
    managed_hash = install_manifest.get("installed_hash") or install_manifest.get("source_hash")
    if installed_hash != managed_hash:
        entry.update(
            action="skip",
            conflict="local_modifications_detected",
            message=f"legacy shared skill has local modifications: {legacy_dir}",
        )
        return entry

    planned_by_name = {item["name"]: item for item in planned_skills}
    remaining_users = []
    if target.is_dir():
        for child in sorted(target.iterdir(), key=lambda path: path.name):
            if child == legacy_dir or child.name.startswith(".") or not child.is_dir() or child.is_symlink():
                continue
            try:
                uses_legacy_shared = skill_references_shared(child)
            except (OSError, ValueError):
                remaining_users.append(child.name)
                continue
            if not uses_legacy_shared:
                continue
            child_manifest = _read_install_manifest(child)
            installed_name = (
                child_manifest.get("name", child.name)
                if child_manifest and child_manifest.get("managed_by") == "nvflare"
                else child.name
            )
            replacement = planned_by_name.get(installed_name)
            if not replacement or replacement.get("action") != "replace":
                remaining_users.append(installed_name)

    if remaining_users:
        entry.update(
            action="skip",
            conflict="legacy_shared_in_use",
            message="legacy shared skill is still referenced by installs that are not safely replaceable: "
            + ", ".join(sorted(set(remaining_users))),
            dependent_skills=sorted(set(remaining_users)),
        )
        return entry

    entry.update(
        source_hash=managed_hash,
        backup_path=str(_backup_path(target, SHARED_SKILL_REFERENCE_DIR)),
    )
    return entry


def _backup_legacy_shared(plan_entry: dict) -> None:
    legacy_dir = Path(plan_entry["target_path"])
    backup_path = Path(plan_entry["backup_path"])
    parent_identity = _directory_identity(legacy_dir.parent)
    with _skill_install_lock(legacy_dir):
        if legacy_dir.is_symlink() or not legacy_dir.is_dir():
            raise ValueError(f"legacy shared skill is no longer a safe directory: {legacy_dir}")
        install_manifest = _read_install_manifest(legacy_dir)
        if not install_manifest or install_manifest.get("managed_by") != "nvflare":
            raise ValueError(f"legacy shared skill is no longer managed by nvflare: {legacy_dir}")
        installed_hash = skill_tree_hash(legacy_dir, exclude_names={INSTALL_MANIFEST_FILE_NAME})
        if installed_hash != plan_entry.get("source_hash"):
            raise ValueError(f"legacy shared skill changed after the install plan was created: {legacy_dir}")
        if backup_path.exists() or backup_path.is_symlink():
            raise FileExistsError(f"legacy shared skill backup already exists: {backup_path}")
        backup_root_identity = _prepare_backup_parent(legacy_dir.parent, backup_path)
        _assert_directory_identity(legacy_dir.parent, parent_identity)
        _assert_directory_identity(backup_path.parents[1], backup_root_identity)
        shutil.move(legacy_dir, backup_path)


def _copy_skill(source_dir: Path, target_dir: Path, plan_entry: dict, source: SkillSource) -> None:
    parent_identity = _directory_identity(target_dir.parent)
    with _skill_install_lock(target_dir):
        with tempfile.TemporaryDirectory(prefix=f".{target_dir.name}.", dir=target_dir.parent) as temp_root:
            temp_skill_dir = Path(temp_root) / target_dir.name
            _stage_skill(source_dir, temp_skill_dir, plan_entry, source, installed_path=target_dir)
            _assert_directory_identity(target_dir.parent, parent_identity)
            _publish_staged_skill(temp_skill_dir, target_dir)


def _replace_skill(
    source_dir: Path, target_dir: Path, backup_path: Path, plan_entry: dict, source: SkillSource
) -> None:
    parent_identity = _directory_identity(target_dir.parent)
    with _skill_install_lock(target_dir):
        with tempfile.TemporaryDirectory(prefix=f".{target_dir.name}.", dir=target_dir.parent) as temp_root:
            temp_skill_dir = Path(temp_root) / target_dir.name
            _stage_skill(source_dir, temp_skill_dir, plan_entry, source, installed_path=target_dir)
            if not target_dir.exists():
                raise FileNotFoundError(f"target skill directory no longer exists: {target_dir}")
            if backup_path.exists():
                raise FileExistsError(f"backup skill directory already exists: {backup_path}")
            backup_root_identity = _prepare_backup_parent(target_dir.parent, backup_path)
            _assert_directory_identity(target_dir.parent, parent_identity)
            _assert_directory_identity(backup_path.parents[1], backup_root_identity)
            shutil.move(target_dir, backup_path)
            try:
                _assert_directory_identity(target_dir.parent, parent_identity)
                _publish_staged_skill(temp_skill_dir, target_dir)
            except Exception as publish_error:
                if not target_dir.exists() and backup_path.exists():
                    try:
                        shutil.move(backup_path, target_dir)
                        _remove_empty_dir(backup_path.parent)
                    except Exception as recovery_error:
                        publish_error.recovery_error = recovery_error
                raise


def _prepare_backup_parent(target_root: Path, backup_path: Path) -> tuple[int, int]:
    backup_root = target_root / ".nvflare_bak"
    if backup_path.parents[1] != backup_root:
        raise ValueError(f"backup path escapes the managed backup root: {backup_path}")
    try:
        backup_root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    backup_root_identity = _private_directory_identity(backup_root)
    backup_path.parent.mkdir(mode=0o700)
    return backup_root_identity


def _shared_install_plan(source: SkillSource, target: Path, *, required_hashes: set[str]) -> Optional[dict]:
    """Plan an immutable, non-discoverable shared-resource snapshot."""

    source_shared = source.root / SHARED_SKILL_REFERENCE_DIR
    if not source_shared.is_dir():
        required_hash = next(iter(required_hashes), "missing")
        return {
            "name": SHARED_SKILL_REFERENCE_DIR,
            "source_hash": required_hash,
            "relative_path": SHARED_SKILL_REFERENCE_DIR,
            "target_path": str(target / SHARED_INSTALL_ROOT_DIR / required_hash),
            "version_delta": "blocked",
            "files": [],
            "action": "skip",
            "conflict": "shared_source_missing",
            "message": "selected skill requires shared resources, but the source tree does not contain them",
        }

    if len(required_hashes) > 1:
        return {
            "name": SHARED_SKILL_REFERENCE_DIR,
            "source_hash": "conflicting",
            "relative_path": SHARED_SKILL_REFERENCE_DIR,
            "target_path": str(target / SHARED_INSTALL_ROOT_DIR),
            "version_delta": "blocked",
            "files": [],
            "action": "skip",
            "conflict": "shared_dependency_hash_mismatch",
            "message": "selected skills declare different shared resource hashes",
        }

    source_symlink = _first_symlink_in_tree(source_shared)
    manifest_shared = source.manifest.get("shared")
    expected_hash = manifest_shared.get("source_hash") if isinstance(manifest_shared, dict) else None
    required_hash = next(iter(required_hashes), None)
    try:
        actual_hash = skill_tree_hash(
            source_shared,
            exclude_names=set(SHARED_SKILL_PACKAGING_EXCLUDE_NAMES),
        )
    except (OSError, ValueError) as e:
        actual_hash = None
        source_error = str(e)
    else:
        source_error = ""

    source_hash = expected_hash or actual_hash or "invalid"
    target_shared = target / SHARED_INSTALL_ROOT_DIR / source_hash
    entry = {
        "name": SHARED_SKILL_REFERENCE_DIR,
        "source_hash": source_hash,
        "relative_path": SHARED_SKILL_REFERENCE_DIR,
        "target_path": str(target_shared),
        "version_delta": "new",
        "files": [],
    }

    if source_symlink or actual_hash is None:
        entry.update(
            action="skip",
            conflict="source_symlink_detected" if source_symlink else "shared_source_unreadable",
            version_delta="blocked",
            message=(
                f"shared resource source contains a symlink: {source_symlink}"
                if source_symlink
                else f"shared resource source cannot be hashed: {source_error}"
            ),
        )
        return entry
    if expected_hash is not None and (
        not isinstance(expected_hash, str)
        or not SHA256_HEX_PATTERN.fullmatch(expected_hash)
        or expected_hash != actual_hash
    ):
        entry.update(
            action="skip",
            conflict="shared_manifest_hash_mismatch",
            version_delta="blocked",
            message="shared resource content does not match the source manifest",
        )
        return entry
    if required_hash is not None and required_hash != actual_hash:
        entry.update(
            action="skip",
            conflict="shared_dependency_hash_mismatch",
            version_delta="blocked",
            message="selected skill shared dependency does not match the verified shared resources",
        )
        return entry

    source_hash = actual_hash
    target_shared = target / SHARED_INSTALL_ROOT_DIR / source_hash
    entry["source_hash"] = source_hash
    entry["target_path"] = str(target_shared)
    entry["files"] = _files_to_copy(
        source_shared,
        target_shared,
        exclude_names=set(SHARED_SKILL_PACKAGING_EXCLUDE_NAMES),
    )

    hidden_root = target / SHARED_INSTALL_ROOT_DIR
    if hidden_root.is_symlink():
        entry.update(
            action="skip",
            conflict="target_symlink_detected",
            version_delta="blocked",
            message=f"shared resource root must not be a symlink: {hidden_root}",
        )
        return entry
    if not target_shared.exists():
        entry["action"] = "copy"
        return entry

    target_symlink = _first_symlink_in_tree(target_shared)
    if target_symlink:
        entry.update(
            action="skip",
            conflict="target_symlink_detected",
            version_delta="blocked",
            message=f"shared resource snapshot contains a symlink: {target_symlink}",
        )
        return entry

    install_manifest = _read_install_manifest(target_shared)
    if not install_manifest or install_manifest.get("managed_by") != "nvflare":
        entry.update(
            action="skip",
            conflict="external_install_detected",
            version_delta="blocked",
            message=f"shared resource snapshot is not managed by nvflare: {target_shared}",
        )
        return entry
    try:
        installed_hash = skill_tree_hash(target_shared, exclude_names={INSTALL_MANIFEST_FILE_NAME})
    except (OSError, ValueError) as e:
        installed_hash = None
        entry["target_issue"] = {"type": type(e).__name__, "message": str(e)}
    managed_hash = install_manifest.get("installed_hash") or install_manifest.get("source_hash")
    if installed_hash != managed_hash or install_manifest.get("source_hash") != source_hash:
        entry.update(
            action="skip",
            conflict="local_modifications_detected",
            version_delta="blocked",
            message=f"shared resource snapshot has local modifications: {target_shared}",
        )
        return entry

    entry.update(action="skip", reason="already_installed", version_delta="same")
    return entry


def _sync_shared_references(source: SkillSource, target: Path, plan_entry: Optional[dict]) -> None:
    """Install the planned shared snapshot without ever replacing an old one."""

    if not plan_entry or plan_entry.get("action") == "skip":
        return
    if plan_entry.get("action") != "copy":
        raise ValueError(f"unsupported shared resource action: {plan_entry.get('action')!r}")

    source_shared = source.root / SHARED_SKILL_REFERENCE_DIR
    source_hash = skill_tree_hash(
        source_shared,
        exclude_names=set(SHARED_SKILL_PACKAGING_EXCLUDE_NAMES),
    )
    if source_hash != plan_entry.get("source_hash"):
        raise ValueError("shared resource source changed after the install plan was created")

    target_shared = Path(plan_entry["target_path"])
    hidden_root = target_shared.parent
    if hidden_root.is_symlink():
        raise ValueError(f"shared resource root must not be a symlink: {hidden_root}")
    hidden_root.mkdir(mode=0o700, exist_ok=True)
    if hidden_root.is_symlink():
        raise ValueError(f"shared resource root became a symlink: {hidden_root}")
    hidden_root_identity = _private_directory_identity(hidden_root)

    with _skill_install_lock(target_shared):
        with tempfile.TemporaryDirectory(prefix=f".{source_hash}.", dir=hidden_root) as temp_root:
            staged_dir = Path(temp_root) / source_hash
            _stage_shared_snapshot(source_shared, staged_dir, plan_entry, source, installed_path=target_shared)
            _assert_directory_identity(hidden_root, hidden_root_identity)
            _publish_staged_skill(staged_dir, target_shared)


def _stage_shared_snapshot(
    source_dir: Path,
    staged_dir: Path,
    plan_entry: dict,
    source: SkillSource,
    *,
    installed_path: Path,
) -> None:
    symlink = _first_symlink_in_tree(source_dir)
    if symlink:
        raise ValueError(f"shared resource source must not contain symlinks: {symlink}")
    _copytree_no_follow(source_dir, staged_dir, exclude_names=SHARED_SKILL_PACKAGING_EXCLUDE_NAMES)
    if (staged_dir / "SKILL.md").exists():
        raise ValueError("shared resource snapshot must not contain SKILL.md")
    installed_hash = skill_tree_hash(staged_dir)
    if installed_hash != plan_entry["source_hash"]:
        raise ValueError("staged shared resource content does not match its planned hash")
    manifest = {
        "schema_version": "1",
        "managed_by": "nvflare",
        "name": SHARED_SKILL_REFERENCE_DIR,
        "nvflare_version": nvflare.__version__,
        "source_type": source.source_type,
        "source_hash": plan_entry["source_hash"],
        "installed_hash": installed_hash,
        "installed_paths": [str(installed_path)],
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    (staged_dir / INSTALL_MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _backup_path(target: Path, skill_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return target / ".nvflare_bak" / f"{timestamp}-{time.time_ns()}" / skill_name


@contextmanager
def _skill_install_lock(target_dir: Path):
    lock_dir = target_dir.parent / f".{target_dir.name}.install.lock"
    owner_token = secrets.token_hex(16)
    lease_fd = None
    owns_lock = False
    try:
        try:
            lease_fd = _create_lock_dir(lock_dir, owner_token)
            owns_lock = True
        except FileExistsError as e:
            try:
                observed_lock = lock_dir.lstat()
            except OSError:
                observed_lock = None
            lease_fd, lease_supported = _try_acquire_lock_lease(lock_dir)
            try:
                current_lock = lock_dir.lstat()
            except OSError:
                current_lock = None
            same_lock = bool(
                observed_lock
                and current_lock
                and (observed_lock.st_dev, observed_lock.st_ino) == (current_lock.st_dev, current_lock.st_ino)
            )
            owner = _read_lock_owner(lock_dir)
            versioned_lease = bool(
                lease_fd is not None and owner and owner.get("lease_version") == INSTALL_LOCK_LEASE_VERSION
            )
            stale = bool(
                same_lock
                and _lock_dir_is_stale(
                    lock_dir,
                    fallback_mtime=observed_lock.st_mtime,
                    lease_is_authoritative=versioned_lease,
                )
            )
            if lease_fd is None or not lease_supported or not stale:
                raise FileExistsError(f"target skill directory is already being installed: {target_dir}") from e
            # We own the OS lease for this inactive stale directory. Reuse it in
            # place instead of check-then-removing its pathname, which could delete
            # a replacement owner's lock.
            _write_lock_metadata(lock_dir, owner_token)
            owns_lock = True
        yield
    finally:
        try:
            if owns_lock:
                _remove_lock_if_owned(lock_dir, owner_token)
        finally:
            if lease_fd is not None:
                os.close(lease_fd)


def _lock_ttl_seconds() -> int:
    value = os.environ.get("NVFLARE_AGENT_SKILL_INSTALL_LOCK_TTL_SECONDS")
    if value is None:
        return DEFAULT_INSTALL_LOCK_TTL_SECONDS
    try:
        return max(0, int(value))
    except ValueError:
        return DEFAULT_INSTALL_LOCK_TTL_SECONDS


def _lock_dir_is_stale(
    lock_dir: Path,
    *,
    fallback_mtime: Optional[float] = None,
    lease_is_authoritative: bool = False,
) -> bool:
    if lock_dir.is_symlink() or not lock_dir.is_dir():
        return False
    ttl_seconds = _lock_ttl_seconds()
    if ttl_seconds <= 0:
        return False
    if not lease_is_authoritative:
        # Legacy locks did not hold an OS lease, so PID liveness remains the
        # only signal that an old-format owner may still be active. For a
        # versioned lock, successfully acquiring the flock is authoritative;
        # the recorded PID may already have been reused by an unrelated process.
        owner_pid = _read_lock_owner_pid(lock_dir)
        if owner_pid is not None and _process_is_alive(owner_pid):
            return False
    lock_started_at = _read_lock_started_at(lock_dir)
    if lock_started_at is not None:
        return time.time() - lock_started_at > ttl_seconds
    try:
        mtime = fallback_mtime if fallback_mtime is not None else lock_dir.stat().st_mtime
        age_seconds = time.time() - mtime
    except OSError:
        return False
    return age_seconds > ttl_seconds


def _create_lock_dir(lock_dir: Path, owner_token: str) -> Optional[int]:
    lock_dir.mkdir()
    lease_fd = None
    try:
        lease_fd, supported = _try_acquire_lock_lease(lock_dir)
        if supported and lease_fd is None:
            raise FileExistsError(f"could not acquire newly created install lock: {lock_dir}")
        _write_lock_metadata(lock_dir, owner_token)
        return lease_fd
    except Exception:
        if lease_fd is not None:
            os.close(lease_fd)
        shutil.rmtree(lock_dir, ignore_errors=True)
        raise


def _write_lock_metadata(lock_dir: Path, owner_token: str) -> None:
    dir_fd = _open_real_directory(lock_dir)
    try:
        _replace_lock_metadata_file(
            dir_fd,
            INSTALL_LOCK_TIMESTAMP_FILE_NAME,
            str(time.time_ns()),
        )
        _replace_lock_metadata_file(
            dir_fd,
            INSTALL_LOCK_OWNER_FILE_NAME,
            json.dumps(
                {
                    "token": owner_token,
                    "pid": os.getpid(),
                    "lease_version": INSTALL_LOCK_LEASE_VERSION,
                }
            ),
        )
    finally:
        os.close(dir_fd)


def _open_real_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        current = path.lstat()
        if not stat.S_ISDIR(opened.st_mode) or stat.S_ISLNK(current.st_mode):
            raise ValueError(f"expected a real directory: {path}")
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError(f"directory changed while it was opened: {path}")
        return fd
    except Exception:
        os.close(fd)
        raise


def _replace_lock_metadata_file(dir_fd: int, file_name: str, text: str) -> None:
    try:
        os.unlink(file_name, dir_fd=dir_fd)
    except FileNotFoundError:
        pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(file_name, flags, 0o600, dir_fd=dir_fd)
    try:
        payload = text.encode("utf-8")
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError(f"failed to write install lock metadata: {file_name}")
            offset += written
    finally:
        os.close(fd)


def _try_acquire_lock_lease(lock_dir: Path) -> tuple[Optional[int], bool]:
    try:
        import fcntl
    except ImportError:
        return None, False

    try:
        dir_fd = _open_real_directory(lock_dir)
    except (OSError, ValueError):
        return None, True

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(INSTALL_LOCK_LEASE_FILE_NAME, flags, 0o600, dir_fd=dir_fd)
    except OSError:
        os.close(dir_fd)
        return None, True
    os.close(dir_fd)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            os.close(fd)
            return None, True
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        return None, True
    return fd, True


def _read_lock_owner(lock_dir: Path) -> Optional[dict]:
    text = _read_lock_metadata_text(lock_dir, INSTALL_LOCK_OWNER_FILE_NAME)
    if text is None:
        return None
    try:
        owner = json.loads(text)
    except json.JSONDecodeError:
        return None
    return owner if isinstance(owner, dict) else None


def _read_lock_owner_pid(lock_dir: Path) -> Optional[int]:
    owner = _read_lock_owner(lock_dir)
    pid = owner.get("pid") if owner else None
    return pid if isinstance(pid, int) and pid > 0 else None


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, AttributeError):
        return False
    return True


def _remove_lock_if_owned(lock_dir: Path, owner_token: str) -> None:
    owner = _read_lock_owner(lock_dir)
    if not owner or owner.get("token") != owner_token:
        return
    shutil.rmtree(lock_dir, ignore_errors=True)


def _read_lock_started_at(lock_dir: Path) -> float | None:
    text = _read_lock_metadata_text(lock_dir, INSTALL_LOCK_TIMESTAMP_FILE_NAME)
    if text is None:
        return None
    try:
        return int(text.strip()) / 1_000_000_000
    except ValueError:
        return None


def _read_lock_metadata_text(lock_dir: Path, file_name: str) -> Optional[str]:
    try:
        dir_fd = _open_real_directory(lock_dir)
    except (OSError, ValueError):
        return None
    fd = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(file_name, flags, dir_fd=dir_fd)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_size > MAX_INSTALL_LOCK_METADATA_BYTES:
            return None
        chunks = []
        remaining = MAX_INSTALL_LOCK_METADATA_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_INSTALL_LOCK_METADATA_BYTES:
            return None
        return payload.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if fd is not None:
            os.close(fd)
        os.close(dir_fd)


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


def _list_error(target_path: str, error: Exception) -> dict:
    return {
        "target": target_path,
        "code": "skill_list_failed",
        "type": type(error).__name__,
        "message": str(error),
    }


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
    _copytree_no_follow(source_dir, staged_dir, exclude_names=SKILL_PACKAGING_EXCLUDE_NAMES)
    staged_source_hash = skill_tree_hash(staged_dir)
    if staged_source_hash != plan_entry["source_hash"]:
        raise ValueError("staged skill content does not match its source manifest hash")
    shared_source_hash = plan_entry.get("shared_source_hash")
    if shared_source_hash:
        if not isinstance(shared_source_hash, str) or not SHA256_HEX_PATTERN.fullmatch(shared_source_hash):
            raise ValueError("shared resource hash must be a lowercase SHA-256 digest")
        _rewrite_shared_references(staged_dir, shared_source_hash)
    installed_hash = skill_tree_hash(staged_dir)
    manifest = {
        "schema_version": "1",
        "managed_by": "nvflare",
        "name": plan_entry["name"],
        "skill_version": plan_entry.get("skill_version"),
        "nvflare_version": nvflare.__version__,
        "source_type": source.source_type,
        "source_hash": plan_entry["source_hash"],
        "installed_hash": installed_hash,
        "shared_source_hash": shared_source_hash,
        "installed_paths": [str(installed_path)],
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    (staged_dir / INSTALL_MANIFEST_FILE_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copytree_no_follow(source_dir: Path, target_dir: Path, *, exclude_names) -> None:
    """Copy a validated tree without following a link introduced during copy."""

    shutil.copytree(
        source_dir,
        target_dir,
        symlinks=True,
        ignore=shutil.ignore_patterns(*exclude_names),
        copy_function=_copy_regular_file_no_follow,
    )


def _copy_regular_file_no_follow(source_path, target_path):
    source_path = Path(source_path)
    target_path = Path(target_path)
    before = source_path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"skill source is not a regular file: {source_path}")
    if before.st_nlink != 1:
        raise ValueError(f"skill source must not be hard-linked: {source_path}")

    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source_path, read_flags)
    target_fd = None
    try:
        opened = os.fstat(source_fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"skill source changed before it could be copied: {source_path}")
        target_fd = os.open(target_path, write_flags, stat.S_IMODE(before.st_mode))
        os.fchmod(target_fd, stat.S_IMODE(before.st_mode))
        with (
            os.fdopen(source_fd, "rb", closefd=False) as source_stream,
            os.fdopen(target_fd, "wb", closefd=False) as target_stream,
        ):
            shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
        after = source_path.lstat()
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ValueError(f"skill source changed while it was being copied: {source_path}")
    except Exception:
        try:
            target_path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)
    return str(target_path)


def _rewrite_shared_references(skill_dir: Path, shared_source_hash: str) -> None:
    """Pin public-skill references to an immutable hidden shared snapshot."""

    replacement = f".nvflare-shared/{shared_source_hash}/"
    reference_pattern = re.compile(r"(?<![A-Za-z0-9_.-])nvflare-shared/")
    text_suffixes = {".md", ".txt", ".json", ".yaml", ".yml", ".py"}
    for root, dir_names, file_names in os.walk(skill_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        dir_names.sort()
        file_names.sort()
        for file_name in file_names:
            path = root_path / file_name
            if path.suffix.lower() not in text_suffixes or path.is_symlink() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(f"expected UTF-8 text while pinning shared references: {path}") from e
            updated = reference_pattern.sub(replacement, text)
            if updated != text:
                path.write_text(updated, encoding="utf-8")


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


def _prepare_install_target(target: Path) -> tuple[int, int]:
    """Create and verify an install root that other OS users cannot rewrite."""

    _create_secure_parent_chain(target.parent)
    _validate_install_parent_chain(target.parent)
    created = False
    try:
        target.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        pass
    target_stat = target.lstat()
    if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISDIR(target_stat.st_mode):
        raise ValueError(f"agent skill target must be a real directory: {target}")
    if hasattr(os, "geteuid") and target_stat.st_uid != os.geteuid():
        raise ValueError(f"agent skill target must be owned by the current user: {target}")
    if stat.S_IMODE(target_stat.st_mode) & 0o022:
        raise ValueError(f"agent skill target must not be group/world writable: {target}")
    if created:
        os.chmod(target, 0o700, follow_symlinks=False)
        target_stat = target.lstat()
    return target_stat.st_dev, target_stat.st_ino


def _create_secure_parent_chain(parent: Path) -> None:
    missing = []
    current = parent
    while True:
        try:
            current.lstat()
            break
        except FileNotFoundError:
            missing.append(current)
            if current == current.parent:
                raise ValueError(f"could not find an existing parent for agent skill target: {parent}")
            current = current.parent
    _validate_install_parent_chain(current)

    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        directory_stat = directory.lstat()
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError(f"agent skill target parent must be a real directory: {directory}")
        if hasattr(os, "geteuid") and directory_stat.st_uid != os.geteuid():
            raise ValueError(f"new agent skill target parent must be owned by the current user: {directory}")
        if stat.S_IMODE(directory_stat.st_mode) & 0o022:
            raise ValueError(f"new agent skill target parent must not be group/world writable: {directory}")


def _validate_install_parent_chain(parent: Path) -> None:
    absolute = parent if parent.is_absolute() else Path.cwd() / parent
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        current_stat = current.lstat()
        if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(current_stat.st_mode):
            raise ValueError(f"agent skill target parent must be a real directory: {current}")
        writable_by_others = stat.S_IMODE(current_stat.st_mode) & 0o022
        sticky = current_stat.st_mode & stat.S_ISVTX
        if writable_by_others and not sticky:
            raise ValueError(
                f"agent skill target parent is writable by other users without sticky protection: {current}"
            )


def _assert_directory_identity(path: Path, expected: tuple[int, int]) -> None:
    current = path.lstat()
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != expected
    ):
        raise ValueError(f"verified install directory changed during installation: {path}")


def _directory_identity(path: Path) -> tuple[int, int]:
    current = path.lstat()
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise ValueError(f"expected a real directory: {path}")
    return current.st_dev, current.st_ino


def _private_directory_identity(path: Path) -> tuple[int, int]:
    current = path.lstat()
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
        raise ValueError(f"expected a real directory: {path}")
    if hasattr(os, "geteuid") and current.st_uid != os.geteuid():
        raise ValueError(f"directory must be owned by the current user: {path}")
    if stat.S_IMODE(current.st_mode) & 0o022:
        raise ValueError(f"directory must not be group/world writable: {path}")
    return current.st_dev, current.st_ino


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
    var_alias = Path("/var")
    if var_alias.is_symlink() and any(_path_is_relative_to(alias, var_alias) for alias in aliases):
        aliases.add(var_alias)
    return tuple(sorted(aliases, key=lambda item: str(item)))


def _is_allowed_system_target_symlink(path: Path, *, target: Path) -> bool:
    for alias in _target_system_symlink_aliases():
        if path == alias and _path_is_relative_to(target, alias):
            return True
    return False


def _path_is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _first_disallowed_target_symlink_component(path: Path) -> Optional[Path]:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for part in parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            if _is_allowed_system_target_symlink(current, target=absolute):
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


def _files_to_copy(source_dir: Path, target_dir: Path, *, exclude_names: set[str]) -> list[dict]:
    files = []
    for root, dir_names, file_names in os.walk(source_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        dir_names.sort()
        file_names.sort()
        dir_names[:] = [name for name in dir_names if name not in exclude_names and not (root_path / name).is_symlink()]
        for file_name in file_names:
            if file_name in exclude_names:
                continue
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
