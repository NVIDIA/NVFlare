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
from importlib import resources

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
