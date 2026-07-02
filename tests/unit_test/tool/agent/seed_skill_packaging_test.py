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

import importlib.util
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from nvflare.tool.agent.inspector import inspect_path
from nvflare.tool.agent.skill_manager import SkillSource, install_skills, list_skills
from nvflare.tool.agent.skill_manifest import build_skill_manifest, copy_released_skills_to_bundle

SEED_SKILLS = {
    "nvflare-orient",
    "nvflare-convert-pytorch",
    "nvflare-convert-lightning",
    "nvflare-diagnose-job",
}


def test_seed_skill_manifest_includes_public_skills_and_skips_shared_references():
    skills_root = _repo_root() / "skills"

    manifest = build_skill_manifest(skills_root, source_type="editable", nvflare_version="2.8.0")

    names = {skill["name"] for skill in manifest["skills"]}
    assert manifest["findings"] == []
    assert SEED_SKILLS.issubset(names)
    assert "_shared" not in names
    assert all(skill["relative_path"] != "_shared" for skill in manifest["skills"])


def test_inspector_recommendations_are_available_in_seed_manifest(tmp_path):
    skills_root = _repo_root() / "skills"
    manifest = build_skill_manifest(skills_root, source_type="editable", nvflare_version="2.8.0")
    available_names = {skill["name"] for skill in manifest["skills"]}

    cases = {
        "pytorch": ("import torch\n", ["nvflare-convert-pytorch"]),
        "lightning": (
            "import pytorch_lightning as pl\n\nclass Net(pl.LightningModule):\n    pass\n",
            ["nvflare-convert-lightning"],
        ),
    }
    for name, (source, expected_skills) in cases.items():
        script = tmp_path / f"{name}.py"
        script.write_text(source, encoding="utf-8")

        data = inspect_path(script)

        recommended_skills = data["skill_selection"]["recommended_skills"]
        assert recommended_skills == expected_skills
        assert set(recommended_skills) <= available_names


def test_convert_pytorch_eval_requires_declared_primary_metric_alignment():
    evals_path = _repo_root() / "dev_tools" / "agent" / "skill_evals" / "nvflare-convert-pytorch" / "evals.json"
    data = json.loads(evals_path.read_text(encoding="utf-8"))
    case = next(item for item in data["evals"] if item["id"] == "pytorch-convert-basic")
    behaviors = {
        item["id"]: item["description"] for item in case["nvflare"]["mandatory_behavior"] if isinstance(item, dict)
    }

    description = behaviors["align-primary-metric"]
    assert "job documentation, task guidance, or source project guidance" in description
    assert "FL recipe/global metric" in description
    assert "FLModel.metrics" in description
    assert "reports that scalar as validation evidence" in description


def test_seed_bundle_copy_excludes_eval_suites(tmp_path):
    skills_root = _repo_root() / "skills"
    bundle_root = tmp_path / "bundle"

    manifest = copy_released_skills_to_bundle(skills_root, bundle_root, nvflare_version="2.8.0")

    names = {skill["name"] for skill in manifest["skills"]}
    assert SEED_SKILLS.issubset(names)
    assert (bundle_root / "_shared" / "conversion-workflow.md").is_file()
    _assert_convert_pytorch_payload(bundle_root / "nvflare-convert-pytorch")
    _assert_convert_lightning_payload(bundle_root / "nvflare-convert-lightning")
    _assert_diagnose_runtime_payload(bundle_root / "nvflare-diagnose-job")
    # Eval suites live in dev_tools/agent/skill_evals/, never in the bundle.
    _assert_analysis_payload_filtered(bundle_root / "nvflare-diagnose-job")
    assert not bundle_root.joinpath("nvflare-convert-pytorch", "evals").exists()


def test_seed_skills_install_into_codex_and_claude_temp_targets(tmp_path):
    source = _seed_skill_source()
    codex_target = tmp_path / "codex-home" / "skills"
    claude_target = tmp_path / "home" / ".claude" / "skills"

    codex_plan = install_skills(agent="codex", target_dir=codex_target, source=source)
    claude_plan = install_skills(agent="claude", target_dir=claude_target, source=source)

    assert codex_plan["applied"] is True
    assert claude_plan["applied"] is True
    assert SEED_SKILLS.issubset({entry["name"] for entry in codex_plan["skills"]})
    assert SEED_SKILLS.issubset({entry["name"] for entry in claude_plan["skills"]})
    assert codex_target.joinpath("_shared", "conversion-workflow.md").is_file()
    assert claude_target.joinpath("_shared", "conversion-workflow.md").is_file()
    _assert_convert_pytorch_payload(codex_target / "nvflare-convert-pytorch")
    _assert_convert_pytorch_payload(claude_target / "nvflare-convert-pytorch")
    _assert_convert_lightning_payload(codex_target / "nvflare-convert-lightning")
    _assert_convert_lightning_payload(claude_target / "nvflare-convert-lightning")
    _assert_diagnose_runtime_payload(codex_target / "nvflare-diagnose-job")
    _assert_diagnose_runtime_payload(claude_target / "nvflare-diagnose-job")
    _assert_analysis_payload_filtered(codex_target / "nvflare-diagnose-job")
    _assert_analysis_payload_filtered(claude_target / "nvflare-diagnose-job")

    codex_list = list_skills(agent="codex", target_dir=codex_target, source=source)
    claude_list = list_skills(agent="claude", target_dir=claude_target, source=source)
    assert SEED_SKILLS.issubset({skill["name"] for skill in codex_list["installed"]})
    assert SEED_SKILLS.issubset({skill["name"] for skill in claude_list["installed"]})


def test_released_seed_skills_install_without_analysis_fixtures(tmp_path):
    skills_root = _repo_root() / "skills"
    bundle_root = tmp_path / "bundle"
    manifest = copy_released_skills_to_bundle(skills_root, bundle_root, nvflare_version="2.8.0")
    source = SkillSource(source_type="wheel", root=bundle_root, manifest=manifest)
    target = tmp_path / "target"

    plan = install_skills(agent="codex", target_dir=target, source=source)
    repeat_plan = install_skills(agent="codex", target_dir=target, source=source)

    assert plan["applied"] is True
    assert repeat_plan["applied"] is True
    assert all(
        entry["action"] == "skip" and entry.get("reason") == "already_installed" for entry in repeat_plan["skills"]
    )
    listed = list_skills(agent="codex", target_dir=target, source=source)
    assert SEED_SKILLS.issubset({skill["name"] for skill in listed["installed"]})
    _assert_diagnose_runtime_payload(target / "nvflare-diagnose-job")
    _assert_analysis_payload_filtered(target / "nvflare-diagnose-job")
    _assert_analysis_payload_filtered(target / "nvflare-convert-pytorch")
    _assert_analysis_payload_filtered(target / "nvflare-convert-lightning")
    _assert_runtime_markdown_references_resolve(target)


def test_seed_skills_dry_run_selects_all_seed_skills_without_copying(tmp_path):
    source = _seed_skill_source()
    target = tmp_path / "target"

    plan = install_skills(agent="codex", target_dir=target, source=source, dry_run=True)

    assert plan["applied"] is False
    assert SEED_SKILLS.issubset({entry["name"] for entry in plan["skills"]})
    # Eval fixtures are not part of the shipped skill source, so a dry-run plan
    # copies only runtime files (SKILL.md, references), never eval logs.
    assert all(
        "evals" not in Path(file_plan["source"]).parts for entry in plan["skills"] for file_plan in entry["files"]
    )
    assert not target.exists()


@pytest.mark.xdist_group(name="setup_py_packaging")
def test_setup_build_py_can_disable_packaged_agent_skills(tmp_path):
    repo_root = _repo_root()
    build_lib = tmp_path / "build_lib"
    env = os.environ.copy()
    env["NVFLARE_PACKAGE_AGENT_SKILLS"] = "0"

    result = subprocess.run(
        [sys.executable, "setup.py", "build_py", "--build-lib", str(build_lib)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    bundle_root = build_lib / "nvflare" / "tool" / "agent" / "bundled_skills"
    manifest = json.loads(bundle_root.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert bundle_root.joinpath("__init__.py").is_file()
    assert manifest["source_type"] == "wheel"
    assert manifest["skills"] == []
    assert manifest["findings"] == []
    for skill_name in SEED_SKILLS:
        assert not bundle_root.joinpath(skill_name).exists()


@pytest.mark.xdist_group(name="setup_py_packaging")
def test_setup_build_py_packages_seed_skills_without_eval_suites(tmp_path):
    repo_root = _repo_root()
    build_lib = tmp_path / "build_lib"
    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, "setup.py", "build_py", "--build-lib", str(build_lib)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    bundle_root = build_lib / "nvflare" / "tool" / "agent" / "bundled_skills"
    manifest = json.loads(bundle_root.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_type"] == "wheel"
    assert "content_mode" not in manifest
    assert SEED_SKILLS.issubset({skill["name"] for skill in manifest["skills"]})
    _assert_diagnose_runtime_payload(bundle_root / "nvflare-diagnose-job")
    _assert_analysis_payload_filtered(bundle_root / "nvflare-diagnose-job")
    _assert_analysis_payload_filtered(bundle_root / "nvflare-convert-pytorch")
    _assert_analysis_payload_filtered(bundle_root / "nvflare-convert-lightning")
    _assert_runtime_markdown_references_resolve(bundle_root)

    target = tmp_path / "target"
    source = SkillSource(source_type="wheel", root=bundle_root, manifest=manifest)
    install_plan = install_skills(agent="codex", target_dir=target, source=source)
    listed = list_skills(agent="codex", target_dir=target, source=source)
    assert install_plan["applied"] is True
    assert SEED_SKILLS.issubset({skill["name"] for skill in listed["installed"]})
    _assert_runtime_markdown_references_resolve(target)


@pytest.mark.xdist_group(name="setup_py_packaging")
def test_setup_bdist_wheel_no_skills_build_has_distinct_filename(tmp_path):
    if not _bdist_wheel_available():
        pytest.skip("bdist_wheel command is not installed in this test environment")

    repo_root = _repo_root()
    dist_dir = tmp_path / "dist"
    env = os.environ.copy()
    env["NVFLARE_PACKAGE_AGENT_SKILLS"] = "0"

    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(dist_dir)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1
    assert "-1no_skills-" in wheels[0].name
    with zipfile.ZipFile(wheels[0]) as wheel_file:
        names = set(wheel_file.namelist())
        manifest_name = "nvflare/tool/agent/bundled_skills/manifest.json"
        assert manifest_name in names
        manifest = json.loads(wheel_file.read(manifest_name).decode("utf-8"))
        assert manifest["skills"] == []
        for skill_name in SEED_SKILLS:
            assert f"nvflare/tool/agent/bundled_skills/{skill_name}/SKILL.md" not in names


def _seed_skill_source() -> SkillSource:
    skills_root = _repo_root() / "skills"
    return SkillSource(
        source_type="editable",
        root=skills_root,
        manifest=build_skill_manifest(skills_root, source_type="editable", nvflare_version="2.8.0"),
    )


def _bdist_wheel_available() -> bool:
    for module_name in ("setuptools.command.bdist_wheel", "wheel.bdist_wheel"):
        try:
            if importlib.util.find_spec(module_name) is not None:
                return True
        except ModuleNotFoundError:
            continue
    return False


def _assert_convert_pytorch_payload(skill_dir: Path) -> None:
    assert skill_dir.joinpath("SKILL.md").is_file()
    assert skill_dir.joinpath("references", "job-validation.md").is_file()
    assert skill_dir.joinpath("references", "pytorch-client-api-conversion.md").is_file()
    assert skill_dir.joinpath("references", "recipe-selection.md").is_file()

    packaged_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            skill_dir / "SKILL.md",
            skill_dir / "references" / "job-validation.md",
            skill_dir / "references" / "pytorch-client-api-conversion.md",
        ]
    )
    assert "conversion-workflow.md" in packaged_text

    # Runnable templates the references promise must actually ship (not be
    # silently stripped by a future packaging change).
    assert skill_dir.joinpath("references", "templates", "client_with_eval.py").is_file()

    shared_conversion = skill_dir.parent / "_shared" / "conversion-workflow.md"
    assert shared_conversion.is_file()
    assert "Do not require `rg` to be installed" in shared_conversion.read_text(encoding="utf-8")
    # Shared custom-aggregation template must ship so a Lightning-only install
    # can still adapt it.
    assert skill_dir.parent.joinpath("_shared", "templates", "aggregator.py").is_file()


def _assert_convert_lightning_payload(skill_dir: Path) -> None:
    assert skill_dir.joinpath("SKILL.md").is_file()
    assert skill_dir.joinpath("references", "lightning-detection.md").is_file()
    assert skill_dir.joinpath("references", "lightning-conversion.md").is_file()
    assert skill_dir.joinpath("references", "lightning-validation.md").is_file()
    assert skill_dir.joinpath("references", "lightning-ddp-and-tracking.md").is_file()

    packaged_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            skill_dir / "SKILL.md",
            skill_dir / "references" / "lightning-conversion.md",
        ]
    )
    assert "flare.patch(trainer)" in packaged_text
    assert "conversion-workflow.md" in packaged_text

    # Runnable Lightning template the reference promises must actually ship.
    assert skill_dir.joinpath("references", "templates", "lightning_client.py").is_file()

    shared_conversion = skill_dir.parent / "_shared" / "conversion-workflow.md"
    assert shared_conversion.is_file()


def _assert_diagnose_runtime_payload(skill_dir: Path) -> None:
    assert skill_dir.joinpath("SKILL.md").is_file()
    assert skill_dir.joinpath("references", "evidence-collection.md").is_file()
    assert skill_dir.joinpath("references", "failure-patterns.md").is_file()


def _assert_analysis_payload_filtered(skill_dir: Path) -> None:
    # Eval suites are never packaged into a skill; they live in dev_tools/agent/skill_evals/.
    assert not skill_dir.joinpath("evals").exists()


def _assert_runtime_markdown_references_resolve(root: Path) -> None:
    missing = []
    for markdown_path in sorted(root.rglob("*.md")):
        rel_parts = markdown_path.relative_to(root).parts
        if "evals" in rel_parts:
            continue
        text = markdown_path.read_text(encoding="utf-8")
        for ref in sorted(set(re.findall(r"`([^`]+\.md)`", text))):
            ref_path = Path(ref)
            if ref_path.is_absolute():
                continue
            if not (markdown_path.parent / ref_path).resolve(strict=False).is_file():
                missing.append(f"{markdown_path.relative_to(root).as_posix()} -> {ref}")
    assert missing == []


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
