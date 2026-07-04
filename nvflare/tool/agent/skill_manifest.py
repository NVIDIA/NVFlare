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

"""Build and validate the manifest for NVFLARE-owned agent skills."""

import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Iterable, Optional

MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "1"
# Names that must never ship inside a skill even if present: byte-code caches,
# and eval suites, which belong in dev_tools/agent/skill_evals/ — fail closed so
# a stray skills/<skill>/evals/ cannot be bundled or installed and re-expose
# grading-oracle data.
SKILL_PACKAGING_EXCLUDE_NAMES = frozenset({"__pycache__", "*.pyc", "*.pyo", "evals"})
SHARED_SKILL_REFERENCE_DIR = "nvflare-shared"
SHARED_SKILL_PACKAGING_EXCLUDE_NAMES = frozenset({*SKILL_PACKAGING_EXCLUDE_NAMES, "SKILL.md"})
RESERVED_SKILL_FILE_NAMES = frozenset({".nvflare_skill_install.json"})
HASH_READ_CHUNK_BYTES = 1024 * 1024
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SHARED_REFERENCE_PATTERN = re.compile(rb"(?<![A-Za-z0-9_.-])nvflare-shared/")


class SkillManifestError(ValueError):
    """Manifest loading error surfaced through agent CLI structured error handling."""

    def __init__(self, code: str, message: str, hint: str = "", detail: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.detail = detail


def skill_tree_hash(skill_dir: Path, *, exclude_names: Optional[set[str]] = None) -> str:
    """Hash a skill directory by relative paths and file contents."""
    excluded = {"__pycache__"}
    if exclude_names:
        excluded.update(exclude_names)
    digest = hashlib.sha256()
    for file_path in _iter_skill_files(skill_dir, exclude_names=excluded):
        rel_path = file_path.relative_to(skill_dir).as_posix()
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        with _open_regular_nofollow(file_path) as stream:
            for chunk in iter(lambda: stream.read(HASH_READ_CHUNK_BYTES), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _open_regular_nofollow(file_path: Path):
    """Open a source file without following a last-component link or FIFO."""

    before = file_path.lstat()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(file_path, flags)
    try:
        opened = os.fstat(fd)
        current = file_path.lstat()
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"skill source is not a regular file: {file_path}")
        if opened.st_nlink != 1:
            raise ValueError(f"skill source must not be hard-linked: {file_path}")
        identity = (opened.st_dev, opened.st_ino)
        if identity != (before.st_dev, before.st_ino) or identity != (current.st_dev, current.st_ino):
            raise ValueError(f"skill source changed while it was being read: {file_path}")
        return os.fdopen(fd, "rb")
    except Exception:
        os.close(fd)
        raise


def build_skill_manifest(
    skills_root: Path | str,
    *,
    source_type: str,
    nvflare_version: str = "",
) -> dict:
    """Build the released-skill manifest for a skills source root."""
    root = Path(skills_root)
    skills = []
    findings = []
    source_hash_exclude_names = set(SKILL_PACKAGING_EXCLUDE_NAMES)
    shared = _build_shared_manifest(root)
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            if _should_skip_skill_dir(child):
                continue
            result = _validate_skill_dir(child)
            if not result.ok:
                findings.append(
                    {
                        "skill_dir": child.name,
                        "issues": [
                            {"code": issue.code, "message": issue.message, "path": issue.path}
                            for issue in result.issues
                        ],
                    }
                )
                continue
            metadata = dict(result.metadata)
            # NVFLARE custom fields live under the agentskills.io `metadata` map;
            # name/description remain top level. Validation (result.ok above)
            # guarantees the required sub-map fields are present for valid skills.
            sub = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
            try:
                source_hash = skill_tree_hash(child, exclude_names=source_hash_exclude_names)
            except (OSError, ValueError) as exc:
                raise SkillManifestError(
                    "AGENT_SKILL_MANIFEST_BUILD_FAILED",
                    f"Could not build skill manifest for skill source: {child}",
                    "Check the skill source tree for symlinks or unreadable files, then rebuild the NVFLARE skill bundle.",
                    detail=str(exc),
                ) from exc
            skill = {
                "name": metadata["name"],
                "skill_version": sub.get("skill_version", "0.0.0"),
                "min_flare_version": sub["min_flare_version"],
                "max_flare_version": sub.get("max_flare_version"),
                "blast_radius": sub["blast_radius"],
                "category": sub.get("category"),
                "source_hash": source_hash,
                "relative_path": child.name,
            }
            if shared and skill_references_shared(child):
                # Each installed skill records the exact immutable shared
                # snapshot it references. This avoids a mutable global shared
                # directory changing behavior underneath older skills.
                skill["shared_source_hash"] = shared["source_hash"]
            skills.append(skill)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_type": source_type,
        "nvflare_version": nvflare_version,
        "skills": skills,
        "findings": findings,
    }
    if shared:
        manifest["shared"] = shared
    return validate_manifest(manifest)


def _build_shared_manifest(root: Path) -> Optional[dict]:
    shared_root = root / SHARED_SKILL_REFERENCE_DIR
    if not shared_root.is_dir():
        return None
    try:
        source_hash = skill_tree_hash(
            shared_root,
            exclude_names=set(SHARED_SKILL_PACKAGING_EXCLUDE_NAMES),
        )
    except (OSError, ValueError) as exc:
        raise SkillManifestError(
            "AGENT_SKILL_MANIFEST_BUILD_FAILED",
            f"Could not build shared resource manifest: {shared_root}",
            "Check the shared resource tree for symlinks or unreadable files, then rebuild the NVFLARE skill bundle.",
            detail=str(exc),
        ) from exc
    return {
        "name": SHARED_SKILL_REFERENCE_DIR,
        "relative_path": SHARED_SKILL_REFERENCE_DIR,
        "source_hash": source_hash,
    }


def skill_references_shared(skill_dir: Path) -> bool:
    overlap_bytes = len(SHARED_SKILL_REFERENCE_DIR) + 2
    text_suffixes = {".md", ".txt", ".json", ".yaml", ".yml", ".py"}
    for file_path in _iter_skill_files(skill_dir, exclude_names=set(SKILL_PACKAGING_EXCLUDE_NAMES)):
        if file_path.suffix.lower() not in text_suffixes:
            continue
        tail = b""
        with _open_regular_nofollow(file_path) as stream:
            for chunk in iter(lambda: stream.read(HASH_READ_CHUNK_BYTES), b""):
                candidate = tail + chunk
                if SHARED_REFERENCE_PATTERN.search(candidate):
                    return True
                tail = candidate[-overlap_bytes:]
    return False


def _should_skip_skill_dir(path: Path) -> bool:
    # nvflare-shared is a real (spec-validated) skill dir, but it is shared
    # content copied into every install by _copy_shared_references_to_bundle, not
    # a user-selectable skill, so keep it out of the selectable skills manifest.
    return (
        path.name.startswith(".")
        or path.name.startswith("_")
        or path.name == SHARED_SKILL_REFERENCE_DIR
        or not path.is_dir()
    )


def write_manifest(manifest: dict, manifest_path: Path | str) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(manifest_path: Path | str) -> dict:
    path = Path(manifest_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillManifestError(
            "AGENT_SKILL_MANIFEST_READ_FAILED",
            f"Could not read skill manifest: {path}",
            "Rebuild or reinstall the NVFLARE agent skill bundle.",
            detail=str(exc),
        ) from exc
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SkillManifestError(
            "AGENT_SKILL_MANIFEST_INVALID_JSON",
            f"Skill manifest is not valid JSON: {path}",
            "Rebuild or reinstall the NVFLARE agent skill bundle.",
            detail=str(exc),
        ) from exc
    if not isinstance(manifest, dict):
        raise SkillManifestError(
            "AGENT_SKILL_MANIFEST_INVALID",
            f"Skill manifest must contain a JSON object: {path}",
            "Rebuild or reinstall the NVFLARE agent skill bundle.",
        )
    return validate_manifest(manifest, manifest_path=path)


def validate_manifest(manifest: dict, *, manifest_path: Optional[Path] = None) -> dict:
    """Validate install-relevant manifest fields before any path is joined."""

    location = f": {manifest_path}" if manifest_path is not None else ""

    def invalid(message: str) -> SkillManifestError:
        return SkillManifestError(
            "AGENT_SKILL_MANIFEST_INVALID",
            f"Skill manifest is invalid{location}",
            "Rebuild or reinstall the NVFLARE agent skill bundle.",
            detail=message,
        )

    if not isinstance(manifest, dict):
        raise invalid("manifest must be a JSON object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise invalid(f"unsupported schema_version: {manifest.get('schema_version')!r}")
    skills = manifest.get("skills")
    if not isinstance(skills, list):
        raise invalid("skills must be a list")
    if not isinstance(manifest.get("findings", []), list):
        raise invalid("findings must be a list")

    shared = manifest.get("shared")
    shared_hash = None
    if shared is not None:
        if not isinstance(shared, dict):
            raise invalid("shared must be an object")
        if shared.get("name") != SHARED_SKILL_REFERENCE_DIR:
            raise invalid(f"shared.name must be {SHARED_SKILL_REFERENCE_DIR!r}")
        if shared.get("relative_path") != SHARED_SKILL_REFERENCE_DIR:
            raise invalid(f"shared.relative_path must be {SHARED_SKILL_REFERENCE_DIR!r}")
        shared_hash = shared.get("source_hash")
        if not isinstance(shared_hash, str) or not SHA256_HEX_PATTERN.fullmatch(shared_hash):
            raise invalid("shared.source_hash must be a lowercase SHA-256 digest")

    names = set()
    relative_paths = set()
    for index, skill in enumerate(skills):
        if not isinstance(skill, dict):
            raise invalid(f"skills[{index}] must be an object")
        name = skill.get("name")
        relative_path = skill.get("relative_path")
        source_hash = skill.get("source_hash")
        if not isinstance(name, str) or not SKILL_NAME_PATTERN.fullmatch(name):
            raise invalid(f"skills[{index}].name is not a canonical skill name")
        if relative_path != name:
            raise invalid(f"skills[{index}].relative_path must exactly match its skill name")
        if not isinstance(source_hash, str) or not SHA256_HEX_PATTERN.fullmatch(source_hash):
            raise invalid(f"skills[{index}].source_hash must be a lowercase SHA-256 digest")
        if name in names:
            raise invalid(f"duplicate skill name: {name}")
        if relative_path in relative_paths:
            raise invalid(f"duplicate skill relative_path: {relative_path}")
        names.add(name)
        relative_paths.add(relative_path)

        dependency_hash = skill.get("shared_source_hash")
        if dependency_hash is not None:
            if not isinstance(dependency_hash, str) or not SHA256_HEX_PATTERN.fullmatch(dependency_hash):
                raise invalid(f"skills[{index}].shared_source_hash must be a lowercase SHA-256 digest")
            if shared_hash is None:
                raise invalid(f"skills[{index}] declares shared resources but the top-level shared entry is missing")
            if dependency_hash != shared_hash:
                raise invalid(f"skills[{index}].shared_source_hash does not match shared.source_hash")

    return manifest


def copy_released_skills_to_bundle(
    skills_root: Path | str,
    bundle_root: Path | str,
    *,
    nvflare_version: str = "",
) -> dict:
    """Copy valid released skills and write their manifest into a package bundle directory."""
    source_root = Path(skills_root)
    target_root = Path(bundle_root)
    target_identity = _clean_bundle_root(target_root)

    manifest = build_skill_manifest(
        source_root,
        source_type="wheel",
        nvflare_version=nvflare_version,
    )
    ignore_names = set(SKILL_PACKAGING_EXCLUDE_NAMES)
    _assert_bundle_root_identity(target_root, target_identity)
    _copy_shared_references_to_bundle(
        source_root,
        target_root,
        ignore_names=ignore_names,
        expected_hash=(manifest.get("shared") or {}).get("source_hash"),
    )
    for skill in manifest["skills"]:
        _assert_bundle_root_identity(target_root, target_identity)
        bundled_skill = target_root / skill["relative_path"]
        _copytree_no_follow(
            source_root / skill["relative_path"],
            bundled_skill,
            exclude_names=ignore_names,
        )
        bundled_hash = skill_tree_hash(bundled_skill, exclude_names=ignore_names)
        if bundled_hash != skill["source_hash"]:
            raise SkillManifestError(
                "AGENT_SKILL_MANIFEST_BUILD_FAILED",
                f"Bundled skill content changed while being copied: {skill['name']}",
                "Retry from a stable source checkout.",
            )
    _assert_bundle_root_identity(target_root, target_identity)
    write_manifest(manifest, target_root / MANIFEST_FILE_NAME)
    return manifest


def _copy_shared_references_to_bundle(
    source_root: Path,
    target_root: Path,
    *,
    ignore_names: set[str],
    expected_hash: Optional[str],
) -> None:
    shared_root = source_root / SHARED_SKILL_REFERENCE_DIR
    if not shared_root.is_dir():
        return
    # Validate that shared references contain no symlinks. The hash value is
    # not needed here; skill_tree_hash raises before copying unsafe content.
    shared_ignore_names = {*ignore_names, "SKILL.md"}
    skill_tree_hash(shared_root, exclude_names=shared_ignore_names)
    bundled_shared = target_root / SHARED_SKILL_REFERENCE_DIR
    _copytree_no_follow(
        shared_root,
        bundled_shared,
        exclude_names=shared_ignore_names,
    )
    # Shared content is an internal dependency, not a selectable skill.
    # Omitting SKILL.md keeps bundled resources non-discoverable.
    bundled_hash = skill_tree_hash(bundled_shared, exclude_names=shared_ignore_names)
    if expected_hash is not None and bundled_hash != expected_hash:
        raise SkillManifestError(
            "AGENT_SKILL_MANIFEST_BUILD_FAILED",
            "Bundled shared resource content changed while being copied.",
            "Retry from a stable source checkout.",
        )


def _copytree_no_follow(source_dir: Path, target_dir: Path, *, exclude_names: set[str]) -> None:
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
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"skill bundle source must be one regular, non-hard-linked file: {source_path}")
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    target_fd = os.open(target_path, write_flags, stat.S_IMODE(before.st_mode))
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(target_fd, stat.S_IMODE(before.st_mode))
        with (
            _open_regular_nofollow(source_path) as source_stream,
            os.fdopen(target_fd, "wb", closefd=False) as target_stream,
        ):
            shutil.copyfileobj(source_stream, target_stream, length=HASH_READ_CHUNK_BYTES)
        after = source_path.lstat()
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ValueError(f"skill bundle source changed while being copied: {source_path}")
    except Exception:
        try:
            target_path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(target_fd)
    return str(target_path)


def write_empty_skill_bundle(bundle_root: Path | str, *, nvflare_version: str = "") -> dict:
    """Write an empty package skill bundle manifest and remove bundled skill content."""
    target_root = Path(bundle_root)
    target_identity = _clean_bundle_root(target_root)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_type": "wheel",
        "nvflare_version": nvflare_version,
        "skills": [],
        "findings": [],
    }
    _assert_bundle_root_identity(target_root, target_identity)
    write_manifest(manifest, target_root / MANIFEST_FILE_NAME)
    return manifest


def _clean_bundle_root(target_root: Path) -> tuple[int, int]:
    logical_root = Path(os.path.abspath(os.path.normpath(str(target_root))))
    if logical_root == Path(logical_root.anchor):
        raise ValueError(f"skill bundle root must not be a filesystem root: {target_root}")
    _create_real_directory_chain(logical_root)
    root_fd = _open_real_directory(logical_root)
    try:
        root_stat = os.fstat(root_fd)
        identity = root_stat.st_dev, root_stat.st_ino
        for child_name in os.listdir(root_fd):
            try:
                child_stat = os.stat(child_name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if child_name == "__init__.py":
                if not stat.S_ISREG(child_stat.st_mode) or child_stat.st_nlink != 1:
                    raise ValueError("skill bundle package marker must be one regular, non-hard-linked file")
                continue
            if stat.S_ISDIR(child_stat.st_mode):
                _remove_directory_at(root_fd, child_name, expected_device=root_stat.st_dev)
            else:
                # unlink() removes regular files and links themselves, never a
                # link target outside the verified bundle root.
                os.unlink(child_name, dir_fd=root_fd)
        return identity
    finally:
        os.close(root_fd)


def _remove_directory_at(parent_fd: int, name: str, *, expected_device: int) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    child_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(child_fd)
        if not stat.S_ISDIR(opened.st_mode) or opened.st_dev != expected_device:
            raise ValueError(f"skill bundle cleanup refuses non-directory or mounted entry: {name}")
        for child_name in os.listdir(child_fd):
            try:
                child_stat = os.stat(child_name, dir_fd=child_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(child_stat.st_mode):
                _remove_directory_at(child_fd, child_name, expected_device=expected_device)
            else:
                os.unlink(child_name, dir_fd=child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _create_real_directory_chain(path: Path) -> None:
    missing = []
    current = path
    while True:
        try:
            current.lstat()
            break
        except FileNotFoundError:
            missing.append(current)
            if current == current.parent:
                raise ValueError(f"could not find an existing parent for skill bundle root: {path}")
            current = current.parent

    _validate_real_directory_chain(current)
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        mode = directory.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError(f"skill bundle path must contain only real directories: {directory}")


def _validate_real_directory_chain(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError(f"skill bundle path must contain only real directories: {current}")


def _open_real_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        current = path.lstat()
        if not stat.S_ISDIR(opened.st_mode) or stat.S_ISLNK(current.st_mode):
            raise ValueError(f"skill bundle root must be a real directory: {path}")
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError(f"skill bundle root changed while it was opened: {path}")
        if hasattr(os, "geteuid") and opened.st_uid != os.geteuid():
            raise ValueError(f"skill bundle root must be owned by the current user: {path}")
        if stat.S_IMODE(opened.st_mode) & 0o002:
            raise ValueError(f"skill bundle root must not be world writable: {path}")
        return fd
    except Exception:
        os.close(fd)
        raise


def _assert_bundle_root_identity(path: Path, expected: tuple[int, int]) -> None:
    logical_path = Path(os.path.abspath(os.path.normpath(str(path))))
    current = logical_path.lstat()
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != expected
    ):
        raise ValueError(f"verified skill bundle root changed during packaging: {path}")


def _iter_skill_files(skill_dir: Path, *, exclude_names: set[str]) -> Iterable[Path]:
    try:
        root_mode = skill_dir.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"skill directory cannot be inspected: {skill_dir}") from exc
    if stat.S_ISLNK(root_mode):
        raise ValueError("skill directory contains symlink: .")
    if not stat.S_ISDIR(root_mode):
        raise ValueError(f"skill directory is not a directory: {skill_dir}")
    for root, dir_names, file_names in os.walk(skill_dir, topdown=True, followlinks=False):
        root_path = Path(root)
        rel_root = root_path.relative_to(skill_dir)
        if any(part in exclude_names for part in rel_root.parts):
            dir_names[:] = []
            continue

        dir_names.sort()
        file_names.sort()
        for dir_name in list(dir_names):
            # Drop excluded dirs (e.g. __pycache__) before the symlink guard so a
            # symlinked byte-code cache does not block hashing valid skill trees.
            if dir_name in exclude_names:
                dir_names.remove(dir_name)
                continue
            dir_path = root_path / dir_name
            mode = dir_path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"skill directory contains symlink: {dir_path.relative_to(skill_dir).as_posix()}")
            if not stat.S_ISDIR(mode):
                raise ValueError(
                    "skill directory contains unsupported entry type: " f"{dir_path.relative_to(skill_dir).as_posix()}"
                )

        for file_name in file_names:
            file_path = root_path / file_name
            rel_path = file_path.relative_to(skill_dir)
            # Skip excluded and byte-code files before the symlink guard so a symlinked
            # __pycache__/.pyc entry does not raise on otherwise-valid skill trees.
            if any(part in exclude_names for part in rel_path.parts):
                continue
            if file_path.suffix in {".pyc", ".pyo"}:
                continue
            if file_name in RESERVED_SKILL_FILE_NAMES:
                raise ValueError(f"skill directory contains reserved installer file: {rel_path.as_posix()}")
            mode = file_path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"skill directory contains symlink: {rel_path.as_posix()}")
            if not stat.S_ISREG(mode):
                raise ValueError(f"skill directory contains unsupported entry type: {rel_path.as_posix()}")
            yield file_path


def _validate_skill_dir(skill_dir: Path):
    return _load_frontmatter_module().validate_skill_dir(skill_dir)


def _load_frontmatter_module():
    # setup.py loads this file before build isolation has all NVFLARE runtime
    # dependencies. Load the repo-local dev tool directly; dev_tools is a
    # development directory, not a shipped NVFLARE package.
    # TODO: remove this direct-loader workaround when skill packaging no longer
    # imports manifest building from setup.py/build isolation.
    module_name = "nvflare_agent_skill_frontmatter"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = Path(__file__).resolve().parents[3] / "dev_tools" / "agent" / "skills" / "checks" / "frontmatter.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load agent skill frontmatter validator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
    return module
