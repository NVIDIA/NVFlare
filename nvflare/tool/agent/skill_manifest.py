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
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "1"
IGNORED_SKILL_FILE_NAMES = {"__pycache__", "*.pyc", "*.pyo"}
_FRONTMATTER_MODULE = None


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
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_skill_manifest(skills_root: Path | str, *, source_type: str, nvflare_version: str = "") -> dict:
    """Build the released-skill manifest for a skills source root."""
    root = Path(skills_root)
    skills = []
    findings = []
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            if child.name.startswith(".") or not child.is_dir():
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
            skills.append(
                {
                    "name": metadata["name"],
                    "skill_version": metadata.get("skill_version", "0.0.0"),
                    "min_flare_version": metadata["min_flare_version"],
                    "max_flare_version": metadata.get("max_flare_version"),
                    "blast_radius": metadata["blast_radius"],
                    "source_hash": skill_tree_hash(child),
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


def write_manifest(manifest: dict, manifest_path: Path | str) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(manifest_path: Path | str) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def copy_released_skills_to_bundle(
    skills_root: Path | str, bundle_root: Path | str, *, nvflare_version: str = ""
) -> dict:
    """Copy valid released skills and write their manifest into a package bundle directory."""
    source_root = Path(skills_root)
    target_root = Path(bundle_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for child in list(target_root.iterdir()):
        if child.name == "__init__.py":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    manifest = build_skill_manifest(source_root, source_type="wheel", nvflare_version=nvflare_version)
    for skill in manifest["skills"]:
        shutil.copytree(
            source_root / skill["relative_path"],
            target_root / skill["relative_path"],
            ignore=shutil.ignore_patterns(*IGNORED_SKILL_FILE_NAMES),
        )
    write_manifest(manifest, target_root / MANIFEST_FILE_NAME)
    return manifest


def _iter_skill_files(skill_dir: Path, *, exclude_names: set[str]) -> Iterable[Path]:
    for path in sorted(skill_dir.rglob("*"), key=lambda p: p.relative_to(skill_dir).as_posix()):
        if not path.is_file():
            continue
        if any(part in exclude_names for part in path.relative_to(skill_dir).parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def _validate_skill_dir(skill_dir: Path):
    global _FRONTMATTER_MODULE

    if _FRONTMATTER_MODULE is None:
        _FRONTMATTER_MODULE = _load_frontmatter_module()

    return _FRONTMATTER_MODULE.validate_skill_dir(skill_dir)


def _load_frontmatter_module():
    try:
        return importlib.import_module("nvflare.tool.agent_skill_checks.frontmatter")
    except ImportError:
        pass

    # setup.py loads this file before build isolation has all NVFLARE runtime
    # dependencies, so fall back to loading the validator file directly.
    module_name = "nvflare_agent_skill_frontmatter"
    module_path = Path(__file__).resolve().parents[1] / "agent_skill_checks" / "frontmatter.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load agent skill frontmatter validator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
