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
import shutil
from pathlib import Path

import pytest

from nvflare.tool.agent import skill_manager
from nvflare.tool.agent.skill_manager import (
    INSTALL_MANIFEST_FILE_NAME,
    SkillSource,
    find_skill_source,
    install_skills,
    list_skills,
    resolve_agent_target_dir,
)
from nvflare.tool.agent.skill_manifest import build_skill_manifest, write_manifest


def test_resolve_codex_target_uses_codex_home(tmp_path):
    target = resolve_agent_target_dir("codex", env={"CODEX_HOME": str(tmp_path / "codex-home")})

    assert target == tmp_path / "codex-home" / "skills"


def test_resolve_claude_target_uses_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    target = resolve_agent_target_dir("claude")

    assert target == tmp_path / ".claude" / "skills"


def test_resolve_target_override_skips_agent_home_resolution(tmp_path):
    target = resolve_agent_target_dir("codex", target_dir=tmp_path / "custom-target", env={"CODEX_HOME": "ignored"})

    assert target == tmp_path / "custom-target"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_resolve_target_override_rejects_symlink_target(tmp_path):
    actual_target = tmp_path / "actual-target"
    actual_target.mkdir()
    link_target = tmp_path / "link-target"
    link_target.symlink_to(actual_target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink components"):
        resolve_agent_target_dir("codex", target_dir=link_target)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_resolve_target_override_rejects_symlink_parent(tmp_path):
    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    link_parent = tmp_path / "link-parent"
    link_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink components"):
        resolve_agent_target_dir("codex", target_dir=link_parent / "skills")


def test_resolve_unsupported_agent_raises_value_error():
    try:
        resolve_agent_target_dir("unsupported")
    except ValueError as e:
        assert "unsupported agent target" in str(e)
    else:
        assert False, "unsupported agent target should raise ValueError"


def test_find_skill_source_does_not_misclassify_site_packages_skills(monkeypatch, tmp_path):
    fake_site_packages = tmp_path / "site-packages"
    fake_module = fake_site_packages / "nvflare" / "tool" / "agent" / "skill_manager.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("# fake installed module\n", encoding="utf-8")
    _write_skill(fake_site_packages / "skills", "unrelated-skill")
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    write_manifest(
        {
            "schema_version": "1",
            "source_type": "wheel",
            "nvflare_version": "2.8.0",
            "skills": [],
            "findings": [],
        },
        bundle_root / "manifest.json",
    )
    monkeypatch.setattr(skill_manager, "__file__", str(fake_module))
    monkeypatch.setattr(skill_manager.resources, "files", lambda _package: bundle_root)

    source = find_skill_source()

    assert source.source_type == "wheel"
    assert source.root == bundle_root


def test_install_skills_dry_run_reports_plan_without_copying(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"

    plan = install_skills(agent="codex", dry_run=True, target_dir=target, source=source)

    assert plan["applied"] is False
    assert plan["skills"][0]["action"] == "copy"
    assert plan["skills"][0]["files"]
    assert not target.exists()


def test_install_skills_installs_all_by_default(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"

    plan = install_skills(agent="codex", target_dir=target, source=source)

    skill_dir = target / "nvflare-test-skill"
    assert plan["applied"] is True
    assert skill_dir.joinpath("SKILL.md").is_file()
    install_manifest = json.loads(skill_dir.joinpath(INSTALL_MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    assert install_manifest["managed_by"] == "nvflare"
    assert install_manifest["name"] == "nvflare-test-skill"


def test_install_skills_reports_missing_named_skill(tmp_path):
    source = _skill_source(tmp_path)

    plan = install_skills(agent="codex", skill_name="nvflare-missing", target_dir=tmp_path / "target", source=source)

    assert plan["applied"] is False
    assert plan["missing"] == ["nvflare-missing"]
    assert plan["skills"] == []


def test_install_skills_preserves_external_target_directory(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    external = target / "nvflare-test-skill"
    external.mkdir(parents=True)
    external.joinpath("SKILL.md").write_text("external content\n", encoding="utf-8")

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is True
    assert plan["skills"][0]["action"] == "skip"
    assert plan["conflicts"][0]["code"] == "external_install_detected"
    assert external.joinpath("SKILL.md").read_text(encoding="utf-8") == "external content\n"


def test_install_skills_is_idempotent_for_same_managed_version(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is True
    assert plan["skills"][0]["action"] == "skip"
    assert plan["skills"][0]["reason"] == "already_installed"


def test_install_skills_preserves_modified_managed_install(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)
    skill_file = target / "nvflare-test-skill" / "SKILL.md"
    skill_file.write_text(skill_file.read_text(encoding="utf-8") + "\n# User Edit\n", encoding="utf-8")

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is True
    assert plan["skills"][0]["action"] == "skip"
    assert plan["skills"][0]["conflict"] == "local_modifications_detected"
    assert plan["conflicts"][0]["code"] == "local_modifications_detected"
    assert "# User Edit" in skill_file.read_text(encoding="utf-8")


def test_install_skills_replaces_unmodified_managed_install_with_backup(tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "nvflare-test-skill", heading="First Skill")
    source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)

    _write_skill(root, "nvflare-test-skill", heading="Second Skill")
    updated_source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )
    plan = install_skills(agent="codex", target_dir=target, source=updated_source)

    skill_plan = plan["skills"][0]
    assert skill_plan["action"] == "replace"
    assert skill_plan["status"] == "replaced"
    assert skill_plan["version_delta"] == "update"
    assert "Second Skill" in (target / "nvflare-test-skill" / "SKILL.md").read_text(encoding="utf-8")
    backup_runs = list((target / ".nvflare_bak").iterdir())
    assert len(backup_runs) == 1
    backup_file = backup_runs[0] / "nvflare-test-skill" / "SKILL.md"
    assert "First Skill" in backup_file.read_text(encoding="utf-8")


def test_install_skills_replace_copy_error_keeps_existing_install(monkeypatch, tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "nvflare-test-skill", heading="First Skill")
    source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)

    _write_skill(root, "nvflare-test-skill", heading="Second Skill")
    updated_source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )

    def copytree_with_failure(src, dst, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(skill_manager.shutil, "copytree", copytree_with_failure)

    plan = install_skills(agent="codex", target_dir=target, source=updated_source)

    assert plan["applied"] is False
    assert plan["errors"] == [
        {
            "skill": "nvflare-test-skill",
            "code": "skill_install_failed",
            "type": "OSError",
            "message": "disk full",
        }
    ]
    assert "First Skill" in (target / "nvflare-test-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert not (target / ".nvflare_bak").exists()


def test_install_skills_replace_publish_error_restores_existing_install(monkeypatch, tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "nvflare-test-skill", heading="First Skill")
    source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)

    _write_skill(root, "nvflare-test-skill", heading="Second Skill")
    updated_source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )

    def publish_with_failure(src, dst):
        raise OSError("publish failed")

    monkeypatch.setattr(skill_manager, "_publish_staged_skill", publish_with_failure)

    plan = install_skills(agent="codex", target_dir=target, source=updated_source)

    assert plan["applied"] is False
    assert plan["errors"] == [
        {
            "skill": "nvflare-test-skill",
            "code": "skill_install_failed",
            "type": "OSError",
            "message": "publish failed",
        }
    ]
    assert "First Skill" in (target / "nvflare-test-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert "Second Skill" not in (target / "nvflare-test-skill" / "SKILL.md").read_text(encoding="utf-8")
    assert not any((target / ".nvflare_bak").iterdir())


def test_install_skills_replace_reports_publish_error_when_recovery_fails(monkeypatch, tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "nvflare-test-skill", heading="First Skill")
    source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)

    _write_skill(root, "nvflare-test-skill", heading="Second Skill")
    updated_source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )
    real_move = shutil.move

    def publish_with_failure(src, dst):
        raise OSError("publish failed")

    def move_with_recovery_failure(src, dst, *args, **kwargs):
        if ".nvflare_bak" in Path(src).parts:
            raise OSError("restore failed")
        return real_move(src, dst, *args, **kwargs)

    monkeypatch.setattr(skill_manager, "_publish_staged_skill", publish_with_failure)
    monkeypatch.setattr(skill_manager.shutil, "move", move_with_recovery_failure)

    plan = install_skills(agent="codex", target_dir=target, source=updated_source)

    assert plan["applied"] is False
    assert plan["errors"] == [
        {
            "skill": "nvflare-test-skill",
            "code": "skill_install_failed",
            "type": "OSError",
            "message": "publish failed",
            "recovery_error": {"type": "OSError", "message": "restore failed"},
        }
    ]
    assert not (target / "nvflare-test-skill").exists()
    assert any((target / ".nvflare_bak").iterdir())


def test_install_skills_reports_copy_error_and_continues(monkeypatch, tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "nvflare-a-skill")
    _write_skill(root, "nvflare-failing-skill")
    _write_skill(root, "nvflare-z-skill")
    source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )
    target = tmp_path / "target"
    real_copytree = shutil.copytree

    def copytree_with_failure(src, dst, *args, **kwargs):
        if src.name == "nvflare-failing-skill":
            raise OSError("disk full")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(skill_manager.shutil, "copytree", copytree_with_failure)

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is False
    assert plan["errors"] == [
        {
            "skill": "nvflare-failing-skill",
            "code": "skill_install_failed",
            "type": "OSError",
            "message": "disk full",
        }
    ]
    assert (target / "nvflare-a-skill" / "SKILL.md").is_file()
    assert not (target / "nvflare-failing-skill").exists()
    assert (target / "nvflare-z-skill" / "SKILL.md").is_file()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_install_skills_rejects_source_symlink_before_copytree(tmp_path):
    root = tmp_path / "skills"
    skill_dir = _write_skill(root, "nvflare-test-skill")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside content\n", encoding="utf-8")
    skill_dir.joinpath("outside-link.txt").symlink_to(outside_file)
    source = SkillSource(
        source_type="editable",
        root=root,
        manifest={
            "schema_version": "1",
            "source_type": "editable",
            "nvflare_version": "2.8.0",
            "skills": [
                {
                    "name": "nvflare-test-skill",
                    "skill_version": "0.0.0",
                    "source_hash": "fake-source-hash",
                    "relative_path": "nvflare-test-skill",
                }
            ],
            "findings": [],
        },
    )

    plan = install_skills(agent="codex", target_dir=tmp_path / "target", source=source)

    assert plan["applied"] is False
    assert plan["errors"][0]["type"] == "ValueError"
    assert "skill source must not contain symlinks" in plan["errors"][0]["message"]
    assert not (tmp_path / "target" / "nvflare-test-skill").exists()


def test_list_skills_reports_available_installed_and_external_conflicts(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)
    (target / "external-skill").mkdir()

    data = list_skills(agent="codex", target_dir=target, source=source)

    assert data["available"][0]["name"] == "nvflare-test-skill"
    assert data["installed"][0]["name"] == "nvflare-test-skill"
    assert data["conflicts"][0]["skill"] == "external-skill"


def test_conflict_falls_back_to_code_for_unknown_conflict(tmp_path):
    conflict = skill_manager._conflict("nvflare-test-skill", "future_conflict", tmp_path / "target")

    assert conflict["code"] == "future_conflict"
    assert conflict["message"] == "future_conflict"


def _skill_source(tmp_path):
    root = tmp_path / "skills"
    _write_skill(root, "nvflare-test-skill")
    return SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )


def _write_skill(root, name, heading="Test Skill"):
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Test skill fixture.\n"
        'min_flare_version: "2.8.0"\n'
        "blast_radius: read_only\n"
        "---\n"
        "\n"
        f"# {heading}\n",
        encoding="utf-8",
    )
    return skill_dir
