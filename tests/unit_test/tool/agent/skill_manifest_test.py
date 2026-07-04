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
import sys
from importlib import resources
from importlib.machinery import ModuleSpec
from types import SimpleNamespace

import pytest

from nvflare.tool.agent import bundled_skills, skill_manifest
from nvflare.tool.agent.skill_manifest import (
    SkillManifestError,
    build_skill_manifest,
    copy_released_skills_to_bundle,
    load_manifest,
    skill_tree_hash,
    write_empty_skill_bundle,
)


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
            "category": "Test",
            "source_hash": skill_tree_hash(skill_dir),
            "relative_path": "nvflare-test-skill",
        }
    ]


def test_build_skill_manifest_hash_is_stable_across_source_type(tmp_path):
    # Eval/QA content lives outside skills/, so there is no dev-vs-release
    # content split: the source hash and manifest are identical regardless of
    # source type, and no content_mode field is emitted.
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")

    editable = build_skill_manifest(tmp_path, source_type="editable", nvflare_version="2.8.0")
    wheel = build_skill_manifest(tmp_path, source_type="wheel", nvflare_version="2.8.0")

    assert "content_mode" not in editable
    assert "content_mode" not in wheel
    assert editable["skills"][0]["source_hash"] == skill_tree_hash(skill_dir)
    assert editable["skills"][0]["source_hash"] == wheel["skills"][0]["source_hash"]


def test_skill_tree_hash_changes_when_skill_content_changes(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    first_hash = skill_tree_hash(skill_dir)

    skill_dir.joinpath("references").mkdir()
    skill_dir.joinpath("references", "notes.md").write_text("new reference\n", encoding="utf-8")

    assert skill_tree_hash(skill_dir) != first_hash


def test_skill_tree_hash_reads_file_contents_in_chunks(tmp_path, monkeypatch):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")

    def fail_read_bytes(_path):
        raise AssertionError("skill_tree_hash should not load whole files with read_bytes")

    monkeypatch.setattr(type(skill_dir), "read_bytes", fail_read_bytes)

    assert isinstance(skill_tree_hash(skill_dir), str)


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


def test_build_skill_manifest_skips_shared_reference_dirs(tmp_path):
    _write_skill(tmp_path, "nvflare-test-skill")
    shared_dir = tmp_path / "nvflare-shared"
    shared_dir.mkdir()
    shared_dir.joinpath("reference.md").write_text("shared guidance\n", encoding="utf-8")

    manifest = build_skill_manifest(tmp_path, source_type="editable", nvflare_version="2.8.0")

    assert manifest["findings"] == []
    assert [skill["name"] for skill in manifest["skills"]] == ["nvflare-test-skill"]
    assert [skill["relative_path"] for skill in manifest["skills"]] == ["nvflare-test-skill"]


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


def test_build_skill_manifest_wraps_source_hash_errors(monkeypatch, tmp_path):
    _write_skill(tmp_path, "nvflare-test-skill")

    def fail_hash(_path, *, exclude_names=None):
        raise ValueError("skill directory contains symlink: late-link")

    monkeypatch.setattr(skill_manifest, "skill_tree_hash", fail_hash)

    with pytest.raises(SkillManifestError) as exc_info:
        build_skill_manifest(tmp_path, source_type="editable", nvflare_version="2.8.0")

    assert exc_info.value.code == "AGENT_SKILL_MANIFEST_BUILD_FAILED"
    assert "late-link" in exc_info.value.detail


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_skill_tree_hash_rejects_skill_symlinks(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside content\n", encoding="utf-8")
    skill_dir.joinpath("outside-link.txt").symlink_to(outside_file)

    with pytest.raises(ValueError, match="skill directory contains symlink"):
        skill_tree_hash(skill_dir)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_skill_tree_hash_rejects_symlink_skill_root(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    link_dir = tmp_path / "nvflare-link-skill"
    link_dir.symlink_to(skill_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="skill directory contains symlink"):
        skill_tree_hash(link_dir)


def test_skill_tree_hash_ignores_python_cache_files(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    first_hash = skill_tree_hash(skill_dir)
    cache_dir = skill_dir / "__pycache__"
    cache_dir.mkdir()
    cache_dir.joinpath("SKILL.cpython-310.pyc").write_bytes(b"compiled cache")
    skill_dir.joinpath("cache.pyo").write_bytes(b"optimized cache")

    assert skill_tree_hash(skill_dir) == first_hash


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_skill_tree_hash_ignores_symlinked_pycache_dir(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    first_hash = skill_tree_hash(skill_dir)
    real_cache = tmp_path / "cache-store"
    real_cache.mkdir()
    real_cache.joinpath("SKILL.cpython-310.pyc").write_bytes(b"compiled cache")
    skill_dir.joinpath("__pycache__").symlink_to(real_cache, target_is_directory=True)

    # A symlinked byte-code cache is excluded, not treated as a forbidden skill symlink.
    assert skill_tree_hash(skill_dir) == first_hash


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_skill_tree_hash_ignores_symlinked_pyc_file(tmp_path):
    skill_dir = _write_skill(tmp_path, "nvflare-test-skill")
    first_hash = skill_tree_hash(skill_dir)
    outside_pyc = tmp_path / "outside.pyc"
    outside_pyc.write_bytes(b"compiled cache")
    skill_dir.joinpath("SKILL.cpython-310.pyc").symlink_to(outside_pyc)

    assert skill_tree_hash(skill_dir) == first_hash


def test_validate_skill_dir_uses_current_frontmatter_loader(monkeypatch, tmp_path):
    first = SimpleNamespace(validate_skill_dir=lambda _path: "first")
    second = SimpleNamespace(validate_skill_dir=lambda _path: "second")

    monkeypatch.setattr(skill_manifest, "_load_frontmatter_module", lambda: first)
    assert skill_manifest._validate_skill_dir(tmp_path) == "first"

    monkeypatch.setattr(skill_manifest, "_load_frontmatter_module", lambda: second)
    assert skill_manifest._validate_skill_dir(tmp_path) == "second"


def test_load_frontmatter_module_does_not_cache_failed_fallback_module(monkeypatch):
    module_name = "nvflare_agent_skill_frontmatter"
    sys.modules.pop(module_name, None)

    def raise_import_error(_name):
        raise ImportError("force direct loader fallback")

    class FailingLoader:
        def create_module(self, _spec):
            return None

        def exec_module(self, module):
            module.partial_state = True
            raise ModuleNotFoundError("No module named 'yaml'")

    monkeypatch.setattr(skill_manifest.importlib, "import_module", raise_import_error)
    monkeypatch.setattr(
        skill_manifest.importlib.util,
        "spec_from_file_location",
        lambda name, _path: ModuleSpec(name, FailingLoader()),
    )

    with pytest.raises(ModuleNotFoundError, match="yaml"):
        skill_manifest._load_frontmatter_module()

    assert module_name not in sys.modules


def test_copy_released_skills_to_bundle_excludes_stray_eval_dir(tmp_path):
    # Fail-closed: a stray skills/<skill>/evals/ must never be bundled, so
    # grading-oracle data cannot re-enter installed skills.
    source_root = tmp_path / "skills"
    bundle_root = tmp_path / "bundle"
    skill_dir = _write_skill(source_root, "nvflare-test-skill")
    skill_dir.joinpath("evals").mkdir()
    skill_dir.joinpath("evals", "evals.json").write_text("{}\n", encoding="utf-8")

    copy_released_skills_to_bundle(source_root, bundle_root, nvflare_version="2.8.0")

    assert bundle_root.joinpath("nvflare-test-skill", "SKILL.md").is_file()
    assert not bundle_root.joinpath("nvflare-test-skill", "evals").exists()


def test_copy_released_skills_to_bundle_copies_runtime_content(tmp_path):
    source_root = tmp_path / "skills"
    bundle_root = tmp_path / "bundle"
    _write_skill(source_root, "nvflare-test-skill")

    manifest = copy_released_skills_to_bundle(source_root, bundle_root, nvflare_version="2.8.0")

    assert bundle_root.joinpath("manifest.json").is_file()
    assert bundle_root.joinpath("nvflare-test-skill", "SKILL.md").is_file()
    saved_manifest = json.loads(bundle_root.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    assert "content_mode" not in saved_manifest
    assert saved_manifest["skills"][0]["name"] == "nvflare-test-skill"
    assert saved_manifest["skills"][0]["source_hash"] == skill_tree_hash(bundle_root / "nvflare-test-skill")


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


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_copy_released_skills_to_bundle_rejects_shared_reference_symlink(tmp_path):
    source_root = tmp_path / "skills"
    bundle_root = tmp_path / "bundle"
    _write_skill(source_root, "nvflare-test-skill")
    shared_dir = source_root / "nvflare-shared"
    shared_dir.mkdir()
    outside_file = tmp_path / "outside.md"
    outside_file.write_text("external shared content\n", encoding="utf-8")
    shared_dir.joinpath("outside-link.md").symlink_to(outside_file)

    with pytest.raises(ValueError, match="skill directory contains symlink"):
        copy_released_skills_to_bundle(source_root, bundle_root, nvflare_version="2.8.0")

    assert not bundle_root.joinpath("nvflare-shared", "outside-link.md").exists()


def test_write_empty_skill_bundle_writes_empty_manifest_and_cleans_existing_content(tmp_path):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    bundle_root.joinpath("__init__.py").write_text("# keep package marker\n", encoding="utf-8")
    bundle_root.joinpath("stale.txt").write_text("stale\n", encoding="utf-8")
    stale_skill = bundle_root / "nvflare-stale-skill"
    stale_skill.mkdir()
    stale_skill.joinpath("SKILL.md").write_text("stale\n", encoding="utf-8")

    manifest = write_empty_skill_bundle(bundle_root, nvflare_version="2.8.0")

    assert bundle_root.joinpath("__init__.py").is_file()
    assert not bundle_root.joinpath("stale.txt").exists()
    assert not stale_skill.exists()
    assert manifest == {
        "schema_version": "1",
        "source_type": "wheel",
        "nvflare_version": "2.8.0",
        "skills": [],
        "findings": [],
    }
    assert json.loads(bundle_root.joinpath("manifest.json").read_text(encoding="utf-8")) == manifest


def test_bundled_skill_manifest_resource_exists():
    manifest = resources.files(bundled_skills).joinpath("manifest.json")

    assert manifest.is_file()
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema_version"] == "1"


def test_load_manifest_wraps_read_and_json_errors(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(SkillManifestError) as read_exc:
        load_manifest(missing)
    assert read_exc.value.code == "AGENT_SKILL_MANIFEST_READ_FAILED"

    corrupt = tmp_path / "manifest.json"
    corrupt.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(SkillManifestError) as json_exc:
        load_manifest(corrupt)
    assert json_exc.value.code == "AGENT_SKILL_MANIFEST_INVALID_JSON"

    corrupt.write_text("[]\n", encoding="utf-8")
    with pytest.raises(SkillManifestError) as shape_exc:
        load_manifest(corrupt)
    assert shape_exc.value.code == "AGENT_SKILL_MANIFEST_INVALID"


def _write_skill(root, name):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Test skill fixture.\n"
        "metadata:\n"
        '  author: "Test Author <test-author@nvidia.com>"\n'
        '  min_flare_version: "2.8.0"\n'
        "  blast_radius: read_only\n"
        "  category: Test\n"
        "---\n"
        "\n"
        "# Test Skill\n",
        encoding="utf-8",
    )
    return skill_dir
