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

import json
import os
from importlib import resources

import pytest

from nvflare.tool.agent import bundled_skills
from nvflare.tool.agent.skill_manifest import build_skill_manifest, copy_released_skills_to_bundle, skill_tree_hash


def test_build_skill_manifest_includes_valid_skill(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")

    manifest = build_skill_manifest(tmp_path, source_type="editable", nvflare_version="2.8.0")

    assert manifest["schema_version"] == "1"
    assert manifest["source_type"] == "editable"
    assert manifest["nvflare_version"] == "2.8.0"
    assert manifest["findings"] == []
    assert manifest["skills"] == [
        {
            "name": "nvflare-test-skill",
            "skill_version": "0.0.0",
            "min_flare_version": "2.8.0",
            "max_flare_version": None,
            "blast_radius": "read_only",
            "source_hash": skill_tree_hash(skill_dir),
            "relative_path": "nvflare-test-skill",
        }
    ]


def test_skill_tree_hash_changes_when_skill_content_changes(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    first_hash = skill_tree_hash(skill_dir)

    skill_dir.joinpath("references").mkdir()
    skill_dir.joinpath("references", "notes.md").write_text("new reference\n", encoding="utf-8")

    assert skill_tree_hash(skill_dir) != first_hash


def test_build_skill_manifest_reports_invalid_skill_findings(tmp_path):
    skill_dir = tmp_path / "nvflare-invalid-skill"
    skill_dir.mkdir()
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n" "name: nvflare-invalid-skill\n" "description: Missing required fields.\n" "---\n",
        encoding="utf-8",
    )

    manifest = build_skill_manifest(tmp_path, source_type="editable", nvflare_version="2.8.0")

    assert manifest["skills"] == []
    assert manifest["findings"][0]["skill_dir"] == "nvflare-invalid-skill"
    assert manifest["findings"][0]["issues"][0]["code"] == "skill-frontmatter-field-required"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_build_skill_manifest_rejects_skill_symlinks(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside content\n", encoding="utf-8")
    skill_dir.joinpath("references").mkdir()
    skill_dir.joinpath("references", "outside-link.txt").symlink_to(outside_file)

    manifest = build_skill_manifest(tmp_path, source_type="editable", nvflare_version="2.8.0")

    assert manifest["skills"] == []
    assert manifest["findings"][0]["skill_dir"] == "nvflare-test-skill"
    assert manifest["findings"][0]["issues"][0]["code"] == "skill-symlink-not-allowed"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_skill_tree_hash_rejects_skill_symlinks(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside content\n", encoding="utf-8")
    skill_dir.joinpath("outside-link.txt").symlink_to(outside_file)

    with pytest.raises(ValueError, match="skill directory contains symlink"):
        skill_tree_hash(skill_dir)


def test_skill_tree_hash_ignores_python_cache_files(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    first_hash = skill_tree_hash(skill_dir)
    cache_dir = skill_dir / "__pycache__"
    cache_dir.mkdir()
    cache_dir.joinpath("SKILL.cpython-310.pyc").write_bytes(b"compiled cache")
    skill_dir.joinpath("cache.pyo").write_bytes(b"optimized cache")

    assert skill_tree_hash(skill_dir) == first_hash


def test_copy_released_skills_to_bundle_writes_manifest_and_files(tmp_path):
    source_root = tmp_path / "skills"
    bundle_root = tmp_path / "bundle"
    _write_skill(source_root, "nvflare-test-skill")

    manifest = copy_released_skills_to_bundle(source_root, bundle_root, nvflare_version="2.8.0")

    assert bundle_root.joinpath("manifest.json").is_file()
    assert bundle_root.joinpath("nvflare-test-skill", "SKILL.md").is_file()
    saved_manifest = json.loads(bundle_root.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    assert saved_manifest["skills"][0]["name"] == "nvflare-test-skill"


def test_copy_released_skills_to_bundle_cleans_existing_bundle_content(tmp_path):
    source_root = tmp_path / "skills"
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    bundle_root.joinpath("__init__.py").write_text("# keep package marker\n", encoding="utf-8")
    bundle_root.joinpath("stale.txt").write_text("stale\n", encoding="utf-8")
    stale_dir = bundle_root / "stale-skill"
    stale_dir.mkdir()
    stale_dir.joinpath("SKILL.md").write_text("stale\n", encoding="utf-8")
    _write_skill(source_root, "nvflare-test-skill")

    copy_released_skills_to_bundle(source_root, bundle_root, nvflare_version="2.8.0")

    assert bundle_root.joinpath("__init__.py").is_file()
    assert not bundle_root.joinpath("stale.txt").exists()
    assert not stale_dir.exists()
    assert bundle_root.joinpath("nvflare-test-skill", "SKILL.md").is_file()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_copy_released_skills_to_bundle_unlinks_stale_bundle_symlink(tmp_path):
    source_root = tmp_path / "skills"
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    stale_target = tmp_path / "stale-target"
    stale_target.mkdir()
    bundle_root.joinpath("stale-link").symlink_to(stale_target, target_is_directory=True)
    _write_skill(source_root, "nvflare-test-skill")

    copy_released_skills_to_bundle(source_root, bundle_root, nvflare_version="2.8.0")

    assert not bundle_root.joinpath("stale-link").exists()
    assert stale_target.is_dir()
    assert bundle_root.joinpath("nvflare-test-skill", "SKILL.md").is_file()


def test_bundled_skill_manifest_resource_exists():
    manifest = resources.files(bundled_skills).joinpath("manifest.json")

    assert manifest.is_file()
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema_version"] == "1"


def _write_skill(root, name):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Test skill fixture.\n"
        'min_flare_version: "2.8.0"\n'
        "blast_radius: read_only\n"
        "---\n"
        "\n"
        "# Test Skill\n",
        encoding="utf-8",
    )
    return skill_dir
