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
import shutil
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
HASH_READ_CHUNK_BYTES = 1024 * 1024


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
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(HASH_READ_CHUNK_BYTES), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


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
            skills.append(
                {
                    "name": metadata["name"],
                    "skill_version": sub.get("skill_version", "0.0.0"),
                    "min_flare_version": sub["min_flare_version"],
                    "max_flare_version": sub.get("max_flare_version"),
                    "blast_radius": sub["blast_radius"],
                    "category": sub.get("category"),
                    "source_hash": source_hash,
                    "relative_path": child.name,
                }
            )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_type": source_type,
        "nvflare_version": nvflare_version,
        "skills": skills,
        "findings": findings,
    }


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
    _clean_bundle_root(target_root)

    manifest = build_skill_manifest(
        source_root,
        source_type="wheel",
        nvflare_version=nvflare_version,
    )
    ignore_names = set(SKILL_PACKAGING_EXCLUDE_NAMES)
    _copy_shared_references_to_bundle(source_root, target_root, ignore_names=ignore_names)
    for skill in manifest["skills"]:
        shutil.copytree(
            source_root / skill["relative_path"],
            target_root / skill["relative_path"],
            ignore=shutil.ignore_patterns(*ignore_names),
        )
    write_manifest(manifest, target_root / MANIFEST_FILE_NAME)
    return manifest


def _copy_shared_references_to_bundle(source_root: Path, target_root: Path, *, ignore_names: set[str]) -> None:
    shared_root = source_root / SHARED_SKILL_REFERENCE_DIR
    if not shared_root.is_dir():
        return
    # Validate that shared references contain no symlinks. The hash value is
    # not needed here; skill_tree_hash raises before copying unsafe content.
    skill_tree_hash(shared_root, exclude_names=ignore_names)
    shutil.copytree(
        shared_root,
        target_root / SHARED_SKILL_REFERENCE_DIR,
        ignore=shutil.ignore_patterns(*ignore_names),
    )


def write_empty_skill_bundle(bundle_root: Path | str, *, nvflare_version: str = "") -> dict:
    """Write an empty package skill bundle manifest and remove bundled skill content."""
    target_root = Path(bundle_root)
    _clean_bundle_root(target_root)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_type": "wheel",
        "nvflare_version": nvflare_version,
        "skills": [],
        "findings": [],
    }
    write_manifest(manifest, target_root / MANIFEST_FILE_NAME)
    return manifest


def _clean_bundle_root(target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)

    for child in list(target_root.iterdir()):
        if child.name == "__init__.py":
            continue
        if child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _iter_skill_files(skill_dir: Path, *, exclude_names: set[str]) -> Iterable[Path]:
    if skill_dir.is_symlink():
        raise ValueError("skill directory contains symlink: .")
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
            if dir_path.is_symlink():
                raise ValueError(f"skill directory contains symlink: {dir_path.relative_to(skill_dir).as_posix()}")

        for file_name in file_names:
            file_path = root_path / file_name
            rel_path = file_path.relative_to(skill_dir)
            # Skip excluded and byte-code files before the symlink guard so a symlinked
            # __pycache__/.pyc entry does not raise on otherwise-valid skill trees.
            if any(part in exclude_names for part in rel_path.parts):
                continue
            if file_path.suffix in {".pyc", ".pyo"}:
                continue
            if file_path.is_symlink():
                raise ValueError(f"skill directory contains symlink: {rel_path.as_posix()}")
            if not file_path.is_file():
                continue
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
