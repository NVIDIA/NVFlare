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
import tempfile
from pathlib import Path

import pytest

from nvflare.tool.agent import skill_manager
from nvflare.tool.agent.skill_manager import (
    INSTALL_LOCK_TIMESTAMP_FILE_NAME,
    INSTALL_MANIFEST_FILE_NAME,
    SkillSource,
    find_skill_source,
    install_skills,
    list_skills,
    resolve_agent_target_dir,
)
from nvflare.tool.agent.skill_manifest import build_skill_manifest, skill_tree_hash, write_manifest


def test_resolve_codex_target_uses_codex_home(tmp_path):
    target = resolve_agent_target_dir("codex", env={"CODEX_HOME": str(tmp_path / "codex-home")})

    assert target == tmp_path / "codex-home" / "skills"


def test_resolve_codex_home_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="parent directory traversal"):
        resolve_agent_target_dir("codex", env={"CODEX_HOME": str(tmp_path / "target" / ".." / "codex-home")})


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_resolve_codex_home_rejects_symlink_component(tmp_path):
    actual_home = tmp_path / "actual-codex-home"
    actual_home.mkdir()
    link_home = tmp_path / "codex-home-link"
    link_home.symlink_to(actual_home, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink components"):
        resolve_agent_target_dir("codex", env={"CODEX_HOME": str(link_home)})


def test_resolve_claude_target_uses_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    target = resolve_agent_target_dir("claude")

    assert target == tmp_path / ".claude" / "skills"


def test_resolve_target_override_skips_agent_home_resolution(tmp_path):
    target = resolve_agent_target_dir("codex", target_dir=tmp_path / "custom-target", env={"CODEX_HOME": "ignored"})

    assert target == tmp_path / "custom-target"


def test_resolve_target_override_accepts_system_temp_alias(tmp_path):
    target = Path(tempfile.gettempdir()) / f"nvflare-skill-target-{tmp_path.name}"

    resolved = resolve_agent_target_dir("codex", target_dir=target, env={"CODEX_HOME": "ignored"})

    assert resolved == target.resolve(strict=False)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_resolve_target_override_allows_only_requested_temp_alias(monkeypatch, tmp_path):
    actual_temp = tmp_path / "actual-temp"
    actual_temp.mkdir()
    alias = tmp_path / "temp-alias"
    alias.symlink_to(actual_temp, target_is_directory=True)
    monkeypatch.setattr(skill_manager, "_target_system_symlink_aliases", lambda: (alias,))

    allowed = resolve_agent_target_dir("codex", target_dir=alias / "skills", env={"CODEX_HOME": "ignored"})

    assert allowed == (actual_temp / "skills").resolve(strict=False)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_resolve_target_override_rejects_temp_alias_symlink_ancestor(monkeypatch, tmp_path):
    actual_root = tmp_path / "actual-root"
    actual_temp = actual_root / "folders" / "T"
    actual_temp.mkdir(parents=True)
    link_root = tmp_path / "link-root"
    link_root.symlink_to(actual_root, target_is_directory=True)
    alias = link_root / "folders" / "T"
    monkeypatch.setattr(skill_manager, "_target_system_symlink_aliases", lambda: (alias,))

    with pytest.raises(ValueError, match="symlink components"):
        resolve_agent_target_dir("codex", target_dir=alias / "skills", env={"CODEX_HOME": "ignored"})


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


def test_resolve_target_override_rejects_parent_traversal(tmp_path):
    with pytest.raises(ValueError, match="parent directory traversal"):
        resolve_agent_target_dir("codex", target_dir=tmp_path / "target" / ".." / "skills")


def test_resolve_unsupported_agent_raises_value_error():
    try:
        resolve_agent_target_dir("unsupported")
    except ValueError as e:
        assert "unsupported agent target" in str(e)
    else:
        assert False, "unsupported agent target should raise ValueError"


def test_find_skill_source_does_not_misclassify_site_packages_skills(monkeypatch, tmp_path):
    fake_site_packages = tmp_path / "site-packages"
    fake_package = fake_site_packages / "nvflare"
    fake_package.mkdir(parents=True)
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
    monkeypatch.setattr(
        skill_manager.util,
        "find_spec",
        lambda _name: type("Spec", (), {"submodule_search_locations": [str(fake_package)]})(),
    )
    monkeypatch.setattr(skill_manager.resources, "files", lambda _package: bundle_root)

    source = find_skill_source()

    assert source.source_type == "wheel"
    assert source.root == bundle_root


def test_find_skill_source_requires_unpacked_bundle(monkeypatch, tmp_path):
    fake_site_packages = tmp_path / "site-packages"
    fake_package = fake_site_packages / "nvflare"
    fake_package.mkdir(parents=True)
    archive_path = tmp_path / "nvflare.zip" / "nvflare" / "tool" / "agent" / "bundled_skills"
    monkeypatch.setattr(
        skill_manager.util,
        "find_spec",
        lambda _name: type("Spec", (), {"submodule_search_locations": [str(fake_package)]})(),
    )
    monkeypatch.setattr(skill_manager.resources, "files", lambda _package: archive_path)

    with pytest.raises(FileNotFoundError, match="unpacked filesystem package"):
        find_skill_source()


def test_find_skill_source_accepts_pyproject_checkout_without_setup_py(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    fake_package = repo_root / "nvflare"
    fake_package.mkdir(parents=True)
    repo_root.joinpath("pyproject.toml").write_text("[project]\nname='nvflare'\n", encoding="utf-8")
    _write_skill(repo_root / "skills", "nvflare-test-skill")
    monkeypatch.setattr(
        skill_manager.util,
        "find_spec",
        lambda _name: type("Spec", (), {"submodule_search_locations": [str(fake_package)]})(),
    )

    source = find_skill_source()

    assert source.source_type == "editable"
    assert source.root == repo_root / "skills"


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


def test_install_skills_copies_runtime_content_excluding_caches_and_evals(tmp_path):
    # Runtime installs copy SKILL.md and references, but fail closed on cache
    # files and any stray eval suite that belongs under dev_tools/agent/skill_evals/.
    root = tmp_path / "skills"
    skill_dir = _write_skill(root, "nvflare-test-skill")
    skill_dir.joinpath("references").mkdir()
    skill_dir.joinpath("references", "notes.md").write_text("reference\n", encoding="utf-8")
    cache_dir = skill_dir.joinpath("__pycache__")
    cache_dir.mkdir()
    cache_dir.joinpath("stale.pyc").write_text("cached\n", encoding="utf-8")
    evals_dir = skill_dir.joinpath("evals")
    evals_dir.mkdir()
    evals_dir.joinpath("evals.json").write_text("{}\n", encoding="utf-8")
    source = SkillSource(
        source_type="wheel",
        root=root,
        manifest=build_skill_manifest(root, source_type="wheel", nvflare_version="2.8.0"),
    )
    target = tmp_path / "target"

    dry_run_plan = install_skills(agent="codex", target_dir=target, source=source, dry_run=True)
    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert all(
        "__pycache__" not in Path(file_plan["source"]).parts and "evals" not in Path(file_plan["source"]).parts
        for file_plan in dry_run_plan["skills"][0]["files"]
    )
    installed = target / "nvflare-test-skill"
    assert plan["applied"] is True
    assert installed.joinpath("SKILL.md").is_file()
    assert installed.joinpath("references", "notes.md").is_file()
    assert not installed.joinpath("__pycache__").exists()
    assert not installed.joinpath("evals").exists()
    assert (
        skill_tree_hash(installed, exclude_names={INSTALL_MANIFEST_FILE_NAME})
        == source.manifest["skills"][0]["source_hash"]
    )


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


def test_install_skills_preserves_modified_shared_references(tmp_path):
    source = _skill_source(tmp_path, shared_heading="Shared v1")
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)
    shared_file = target / "_shared" / "conversion-workflow.md"
    shared_file.write_text("# User Edit\n", encoding="utf-8")

    updated_source = _skill_source(tmp_path, shared_heading="Shared v2")
    plan = install_skills(agent="codex", target_dir=target, source=updated_source)

    assert plan["applied"] is False
    assert plan["errors"] == [
        {
            "skill": "_shared",
            "code": "skill_install_failed",
            "type": "FileExistsError",
            "message": f"shared reference target has local modifications: {target / '_shared'}",
        }
    ]
    assert shared_file.read_text(encoding="utf-8") == "# User Edit\n"


def test_install_skills_updates_unmodified_shared_references(tmp_path):
    source = _skill_source(tmp_path, shared_heading="Shared v1")
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)

    updated_source = _skill_source(tmp_path, shared_heading="Shared v2")
    plan = install_skills(agent="codex", target_dir=target, source=updated_source)

    assert plan["applied"] is True
    assert "# Shared v2" in (target / "_shared" / "conversion-workflow.md").read_text(encoding="utf-8")


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


def test_install_plan_reports_unreadable_managed_install_as_structured_conflict(tmp_path, monkeypatch):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)
    original_hash = skill_manager.skill_tree_hash

    def fake_skill_tree_hash(path, *args, **kwargs):
        if Path(path) == target / "nvflare-test-skill":
            raise PermissionError("permission denied")
        return original_hash(path, *args, **kwargs)

    monkeypatch.setattr(skill_manager, "skill_tree_hash", fake_skill_tree_hash)

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is True
    assert plan["errors"] == []
    assert plan["skills"][0]["action"] == "skip"
    assert plan["skills"][0]["conflict"] == "local_modifications_detected"
    assert plan["skills"][0]["target_issue"]["error_type"] == "PermissionError"
    assert plan["conflicts"][0]["code"] == "local_modifications_detected"


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


def test_skill_install_lock_ignores_stale_lock_removal_race(monkeypatch, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    lock_dir = tmp_path / ".target.install.lock"
    lock_dir.mkdir()
    real_rmtree = shutil.rmtree
    calls = []

    monkeypatch.setattr(skill_manager, "_lock_dir_is_stale", lambda path: path == lock_dir)

    def rmtree_with_race(path, *args, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("ignore_errors") is True
        real_rmtree(path, ignore_errors=True)

    monkeypatch.setattr(skill_manager.shutil, "rmtree", rmtree_with_race)

    with skill_manager._skill_install_lock(target):
        assert lock_dir.is_dir()

    assert calls == [{"ignore_errors": True}, {"ignore_errors": True}]


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


def test_install_skills_reports_target_mkdir_error(monkeypatch, tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "denied" / "target"

    def mkdir_denied(self, *args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(skill_manager.Path, "mkdir", mkdir_denied)

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is False
    assert len(plan["errors"]) == 1
    error = plan["errors"][0]
    assert error["code"] == "skill_install_failed"
    assert error["type"] == "PermissionError"
    assert error["message"] == "permission denied"
    assert not target.exists()


def test_install_skills_fails_when_same_skill_install_lock_exists(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    (target / ".nvflare-test-skill.install.lock").mkdir()

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is False
    assert plan["errors"][0]["type"] == "FileExistsError"
    assert "already being installed" in plan["errors"][0]["message"]
    assert not (target / "nvflare-test-skill").exists()


def test_install_skills_recovers_stale_install_lock(monkeypatch, tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    lock = target / ".nvflare-test-skill.install.lock"
    lock.mkdir()
    stale_time = 1_000_000
    os.utime(lock, (stale_time, stale_time))
    monkeypatch.setenv("NVFLARE_AGENT_SKILL_INSTALL_LOCK_TTL_SECONDS", "1")

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is True
    assert plan["skills"][0]["status"] == "installed"
    assert (target / "nvflare-test-skill" / "SKILL.md").is_file()
    assert not lock.exists()


def test_install_skills_recovers_stale_install_lock_from_timestamp(monkeypatch, tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    lock = target / ".nvflare-test-skill.install.lock"
    lock.mkdir()
    lock.joinpath(INSTALL_LOCK_TIMESTAMP_FILE_NAME).write_text("0", encoding="utf-8")
    monkeypatch.setenv("NVFLARE_AGENT_SKILL_INSTALL_LOCK_TTL_SECONDS", "1")

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is True
    assert plan["skills"][0]["status"] == "installed"
    assert (target / "nvflare-test-skill" / "SKILL.md").is_file()
    assert not lock.exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_install_skills_dry_run_reports_source_symlink_conflict(tmp_path):
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
    target = tmp_path / "target"

    plan = install_skills(agent="codex", target_dir=target, source=source, dry_run=True)

    assert plan["applied"] is False
    assert plan["errors"] == []
    assert plan["skills"][0]["action"] == "skip"
    assert plan["skills"][0]["conflict"] == "source_symlink_detected"
    assert plan["skills"][0]["files"] == []
    assert plan["skills"][0]["source_issue"]["symlink_path"] == str(skill_dir / "outside-link.txt")
    assert plan["conflicts"] == [
        {
            "skill": "nvflare-test-skill",
            "code": "source_symlink_detected",
            "message": "source skill directory contains a symlink",
            "source_path": str(skill_dir),
            "symlink_path": str(skill_dir / "outside-link.txt"),
        }
    ]
    assert not (target / "nvflare-test-skill").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_install_skills_skips_source_symlink_before_copytree(tmp_path):
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
    target = tmp_path / "target"

    plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is True
    assert plan["errors"] == []
    assert plan["skills"][0]["action"] == "skip"
    assert plan["skills"][0]["status"] == "skipped"
    assert plan["conflicts"][0]["code"] == "source_symlink_detected"
    assert not (target / "nvflare-test-skill").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_stage_skill_rejects_source_symlink_before_copytree(tmp_path):
    root = tmp_path / "skills"
    skill_dir = _write_skill(root, "nvflare-test-skill")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside content\n", encoding="utf-8")
    skill_dir.joinpath("outside-link.txt").symlink_to(outside_file)
    source = SkillSource(source_type="editable", root=root, manifest={})

    with pytest.raises(ValueError, match="skill source must not contain symlinks"):
        skill_manager._stage_skill(
            skill_dir,
            tmp_path / "staged",
            {"name": "nvflare-test-skill", "source_hash": "fake-source-hash"},
            source,
            installed_path=tmp_path / "target" / "nvflare-test-skill",
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_install_skills_reports_source_symlink_root_conflict(tmp_path):
    actual_root = tmp_path / "actual-skills"
    skill_dir = _write_skill(actual_root, "nvflare-test-skill")
    root = tmp_path / "skills"
    root.mkdir()
    source_link = root / "nvflare-test-skill"
    source_link.symlink_to(skill_dir, target_is_directory=True)
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

    plan = install_skills(agent="codex", target_dir=tmp_path / "target", source=source, dry_run=True)

    assert plan["skills"][0]["action"] == "skip"
    assert plan["skills"][0]["conflict"] == "source_symlink_detected"
    assert plan["conflicts"][0]["symlink_path"] == str(source_link)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_stage_skill_rejects_source_symlink_root(tmp_path):
    actual_root = tmp_path / "actual-skills"
    skill_dir = _write_skill(actual_root, "nvflare-test-skill")
    source_link = tmp_path / "nvflare-test-skill"
    source_link.symlink_to(skill_dir, target_is_directory=True)
    source = SkillSource(source_type="editable", root=tmp_path, manifest={})

    with pytest.raises(ValueError, match="skill source must not contain symlinks"):
        skill_manager._stage_skill(
            source_link,
            tmp_path / "staged",
            {"name": "nvflare-test-skill", "source_hash": "fake-source-hash"},
            source,
            installed_path=tmp_path / "target" / "nvflare-test-skill",
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_install_skills_reports_installed_skill_nested_symlink_conflict(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside content\n", encoding="utf-8")
    symlink = target / "nvflare-test-skill" / "outside-link.txt"
    symlink.symlink_to(outside_file)

    plan = install_skills(agent="codex", target_dir=target, source=source, dry_run=True)

    assert plan["skills"][0]["action"] == "skip"
    assert plan["skills"][0]["conflict"] == "target_symlink_detected"
    assert plan["skills"][0]["target_issue"]["symlink_path"] == str(symlink)
    assert plan["conflicts"] == [
        {
            "skill": "nvflare-test-skill",
            "code": "target_symlink_detected",
            "message": "target skill directory contains a symlink",
            "target_path": str(target / "nvflare-test-skill"),
            "symlink_path": str(symlink),
        }
    ]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_install_skills_reports_installed_skill_root_symlink_conflict(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)
    installed_skill = target / "nvflare-test-skill"
    actual_skill = tmp_path / "actual-installed-skill"
    shutil.move(installed_skill, actual_skill)
    installed_skill.symlink_to(actual_skill, target_is_directory=True)

    plan = install_skills(agent="codex", target_dir=target, source=source, dry_run=True)

    assert plan["skills"][0]["action"] == "skip"
    assert plan["skills"][0]["conflict"] == "target_symlink_detected"
    assert plan["conflicts"][0]["symlink_path"] == str(installed_skill)


def test_list_skills_reports_available_installed_and_ignores_unrelated_external_skills(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    install_skills(agent="codex", target_dir=target, source=source)
    (target / "external-skill").mkdir()

    data = list_skills(agent="codex", target_dir=target, source=source)

    assert data["available"][0]["name"] == "nvflare-test-skill"
    assert data["installed"][0]["name"] == "nvflare-test-skill"
    assert data["conflicts"] == []
    assert data["errors"] == []


def test_list_skills_reports_unreadable_target_as_structured_error(tmp_path, monkeypatch):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    original_iterdir = type(target).iterdir

    def fake_iterdir(path):
        if path == target:
            raise PermissionError("permission denied")
        return original_iterdir(path)

    monkeypatch.setattr(type(target), "iterdir", fake_iterdir)

    data = list_skills(agent="codex", target_dir=target, source=source)

    assert data["installed"] == []
    assert data["conflicts"] == []
    assert data["errors"] == [
        {
            "target": str(target),
            "code": "skill_list_failed",
            "type": "PermissionError",
            "message": "permission denied",
        }
    ]


def test_list_skills_reports_unreadable_child_as_structured_error(tmp_path, monkeypatch):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    child = target / "nvflare-test-skill"
    child.mkdir()
    original_lstat = type(child).lstat

    def fake_lstat(path):
        if path == child:
            raise PermissionError("permission denied")
        return original_lstat(path)

    monkeypatch.setattr(type(child), "lstat", fake_lstat)

    data = list_skills(agent="codex", target_dir=target, source=source)

    assert data["installed"] == []
    assert data["conflicts"] == []
    assert data["errors"] == [
        {
            "target": str(child),
            "code": "skill_list_failed",
            "type": "PermissionError",
            "message": "permission denied",
        }
    ]


def test_list_skills_flags_name_overlap_external_skill_as_conflict(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    external = target / "nvflare-test-skill"
    external.mkdir(parents=True)
    external.joinpath("SKILL.md").write_text("external content\n", encoding="utf-8")

    data = list_skills(agent="codex", target_dir=target, source=source)

    assert data["installed"] == []
    assert data["conflicts"] == [
        {
            "skill": "nvflare-test-skill",
            "code": "external_install_detected",
            "message": "target skill directory is not managed by nvflare",
            "target_path": str(external),
        }
    ]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_list_skills_reports_name_overlap_symlink_without_following(tmp_path):
    source = _skill_source(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    external = tmp_path / "external-managed-skill"
    external.mkdir()
    external.joinpath(INSTALL_MANIFEST_FILE_NAME).write_text(
        json.dumps(
            {
                "managed_by": "nvflare",
                "name": "nvflare-test-skill",
                "skill_version": "9.9.9",
                "source_hash": "external-hash",
                "source_type": "external",
            }
        ),
        encoding="utf-8",
    )
    linked_skill = target / "nvflare-test-skill"
    linked_skill.symlink_to(external, target_is_directory=True)

    data = list_skills(agent="codex", target_dir=target, source=source)

    assert data["installed"] == []
    assert data["conflicts"] == [
        {
            "skill": "nvflare-test-skill",
            "code": "target_symlink_detected",
            "message": "target skill directory contains a symlink",
            "target_path": str(linked_skill),
            "symlink_path": str(linked_skill),
        }
    ]


def test_conflict_falls_back_to_code_for_unknown_conflict(tmp_path):
    conflict = skill_manager._conflict("nvflare-test-skill", "future_conflict", tmp_path / "target")

    assert conflict["code"] == "future_conflict"
    assert conflict["message"] == "future_conflict"


def _skill_source(tmp_path, *, shared_heading=None):
    root = tmp_path / "skills"
    _write_skill(root, "nvflare-test-skill")
    if shared_heading:
        _write_shared_reference(root, shared_heading)
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
        "metadata:\n"
        '  min_flare_version: "2.8.0"\n'
        "  blast_radius: read_only\n"
        "  category: Test\n"
        "---\n"
        "\n"
        f"# {heading}\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_shared_reference(root, heading):
    shared_dir = root / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.joinpath("conversion-workflow.md").write_text(f"# {heading}\n", encoding="utf-8")
    return shared_dir
