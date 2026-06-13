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

import hashlib
import json
import os
from pathlib import Path


def test_common_write_json_preserves_existing_file_on_replace_failure(tmp_path, monkeypatch):
    from skills.harness import common

    target = tmp_path / "record.json"
    target.write_text('{"status": "old"}', encoding="utf-8")

    def fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(common.os, "replace", fail_replace)

    try:
        common.write_json(target, {"status": "new"})
    except OSError as exc:
        assert "replace failed" in str(exc)
    else:
        raise AssertionError("replace failure should propagate")

    assert target.read_text(encoding="utf-8") == '{"status": "old"}'
    assert list(tmp_path.glob(".record.json.*.tmp")) == []


def test_record_identity_prefers_direct_skill_and_case():
    from skills.harness.record_identity import record_case, record_skill

    record = {
        "skill": "direct-skill",
        "case_id": "direct-case",
        "skill_discovery": {"selected_skill": "fallback-skill", "selected_case_id": "fallback-case"},
    }

    assert record_skill(record) == "direct-skill"
    assert record_case(record) == "direct-case"


def test_record_identity_falls_back_to_skill_discovery():
    from skills.harness.record_identity import record_case, record_skill

    record = {"skill_discovery": {"selected_skill": "fallback-skill", "selected_case_id": "fallback-case"}}

    assert record_skill(record) == "fallback-skill"
    assert record_case(record) == "fallback-case"


def test_collect_report_artifacts_skips_symlinks(tmp_path):
    from skills.harness.artifacts import collect_report_artifacts

    root = tmp_path / "results"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    root.joinpath("leak.txt").symlink_to(outside)
    root.joinpath("kept.txt").write_text("normal\n", encoding="utf-8")

    artifacts = collect_report_artifacts(root)

    assert {item["relative_path"] for item in artifacts} == {"kept.txt"}


def test_collect_report_artifacts_does_not_traverse_symlinked_directories(tmp_path):
    if not hasattr(os, "symlink"):
        return
    from skills.harness.artifacts import collect_report_artifacts

    root = tmp_path / "results"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.joinpath("leak.txt").write_text("secret\n", encoding="utf-8")
    root.joinpath("linked").symlink_to(outside, target_is_directory=True)
    root.joinpath("kept.txt").write_text("normal\n", encoding="utf-8")

    artifacts = collect_report_artifacts(root)

    assert {item["relative_path"] for item in artifacts} == {"kept.txt"}


def test_workspace_artifact_capture_does_not_traverse_symlinked_directories(tmp_path):
    if not hasattr(os, "symlink"):
        return
    from skills.harness.artifacts import capture_workspace_delta, write_workspace_baseline

    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    workspace.joinpath("kept.py").write_text("print('kept')\n", encoding="utf-8")
    outside.joinpath("leak.py").write_text("print('secret')\n", encoding="utf-8")
    workspace.joinpath("linked").symlink_to(outside, target_is_directory=True)
    baseline_path = tmp_path / "baseline.json"

    write_workspace_baseline(workspace, baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    capture_workspace_delta(
        workspace,
        tmp_path / "empty_baseline.json",
        tmp_path / "delta",
        tmp_path / "delta_manifest.json",
        tmp_path / "nvflare",
        include_runtime_artifacts=False,
    )
    manifest = json.loads((tmp_path / "delta_manifest.json").read_text(encoding="utf-8"))

    assert set(baseline["files"]) == {"kept.py"}
    assert {item["path"] for item in manifest["final_files"]} == {"kept.py"}
    assert {item["path"] for item in manifest["changed_files"]} == {"kept.py"}


def test_workspace_artifact_json_writes_use_atomic_helper(tmp_path, monkeypatch):
    from skills.harness import artifacts

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("kept.py").write_text("print('kept')\n", encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    manifest_path = tmp_path / "delta_manifest.json"
    calls = []

    def fake_write_json_atomic(path, value):
        calls.append((Path(path).name, value))

    monkeypatch.setattr(artifacts, "write_json_atomic", fake_write_json_atomic)

    artifacts.write_workspace_baseline(workspace, baseline_path)
    artifacts.capture_workspace_delta(
        workspace,
        tmp_path / "empty_baseline.json",
        tmp_path / "delta",
        manifest_path,
        tmp_path / "nvflare",
        include_runtime_artifacts=False,
    )

    assert [name for name, _value in calls] == ["baseline.json", "delta_manifest.json"]
    assert calls[0][1]["files"]["kept.py"]["size_bytes"] > 0
    assert calls[1][1]["final_file_manifest_count"] == 1


def test_runtime_artifact_capture_preserves_generic_workspace_json(tmp_path):
    from skills.harness.artifacts import capture_workspace_delta, write_workspace_baseline

    workspace = tmp_path / "workspace"
    runtime_root = tmp_path / "nvflare_workspaces"
    workspace.mkdir()
    workspace.joinpath("client.py").write_text("print('client')\n", encoding="utf-8")
    runtime_metric = runtime_root / "job_a" / "server" / "simulate_job" / "metrics" / "custom_results.json"
    runtime_metric.parent.mkdir(parents=True)
    runtime_metric.write_text(
        json.dumps({"server": {"metric": "auroc", "score": 0.8123}}) + "\n",
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    write_workspace_baseline(workspace, baseline_path)

    capture_workspace_delta(
        workspace,
        baseline_path,
        tmp_path / "delta",
        tmp_path / "delta_manifest.json",
        tmp_path / "agent_runtime",
        extra_runtime_artifact_sources=[("runtime_workspaces", runtime_root)],
    )
    manifest = json.loads((tmp_path / "delta_manifest.json").read_text(encoding="utf-8"))

    assert [item["path"] for item in manifest["runtime_artifacts"]] == [
        "runtime_workspaces/job_a/server/simulate_job/metrics/custom_results.json"
    ]


def test_metric_artifact_parser_reads_generic_json_shape(tmp_path):
    from skills.harness.metric_artifacts import validation_metric_from_workspace_delta_manifest

    delta = tmp_path / "delta"
    artifact = (
        delta / "runtime_artifacts" / "runtime_workspaces" / "job_a" / "server" / "metrics" / "custom_results.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps({"server": {"metric": "auroc", "score": 0.8123}}) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "delta_dir": str(delta),
        "runtime_artifacts": [
            {
                "path": "runtime_workspaces/job_a/server/metrics/custom_results.json",
                "artifact_path": "runtime_artifacts/runtime_workspaces/job_a/server/metrics/custom_results.json",
            }
        ],
    }

    metric = validation_metric_from_workspace_delta_manifest(manifest, tmp_path / "delta_manifest.json", "AUROC")

    assert metric["name"] == "AUROC"
    assert metric["value"] == 0.8123
    assert metric["source"] == "metrics_artifact"


def test_metric_artifact_parser_prefers_runtime_artifacts_over_changed_files(tmp_path):
    from skills.harness.metric_artifacts import validation_metric_from_workspace_delta_manifest

    delta = tmp_path / "delta"
    changed_artifact = (
        delta / "changed_files" / "fl_workspace" / "job_a" / "server" / "simulate_job" / "metrics" / "summary.json"
    )
    runtime_artifact = delta / "runtime_artifacts" / "server" / "simulate_job" / "metrics" / "summary.json"
    changed_artifact.parent.mkdir(parents=True)
    runtime_artifact.parent.mkdir(parents=True)
    changed_artifact.write_text(json.dumps({"server": {"metric": "auroc", "score": 0.1111}}), encoding="utf-8")
    runtime_artifact.write_text(json.dumps({"server": {"metric": "auroc", "score": 0.8123}}), encoding="utf-8")
    manifest = {
        "delta_dir": str(delta),
        "changed_files": [
            {
                "path": "fl_workspace/job_a/server/simulate_job/metrics/summary.json",
                "artifact_path": "changed_files/fl_workspace/job_a/server/simulate_job/metrics/summary.json",
            }
        ],
        "runtime_artifacts": [
            {
                "path": "server/simulate_job/metrics/summary.json",
                "artifact_path": "runtime_artifacts/server/simulate_job/metrics/summary.json",
            }
        ],
    }

    metric = validation_metric_from_workspace_delta_manifest(manifest, tmp_path / "delta_manifest.json", "AUROC")

    assert metric["value"] == 0.8123
    assert "/runtime_artifacts/" in metric["source_path"]


def test_parse_usage_activity_scans_hints_from_raw_line_without_json_reserialize(tmp_path, monkeypatch):
    from skills.harness import events

    events_path = tmp_path / "events.jsonl"
    events_path.write_text(json.dumps({"type": "turn", "message": "read SKILL.md"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(events.json, "dumps", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unused")))

    _usage, activity = events.parse_usage_and_activity_data(events_path)

    assert activity["hint_counts"]["skill_md"] == 1


def test_parse_usage_activity_caps_command_list(tmp_path):
    from skills.harness import events

    events_path = tmp_path / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as stream:
        for index in range(events.MAX_ACTIVITY_COMMANDS + 3):
            stream.write(json.dumps({"type": "turn", "cmd": f"python job.py --round {index}"}) + "\n")

    _usage, activity = events.parse_usage_and_activity_data(events_path)

    assert activity["command_count"] == events.MAX_ACTIVITY_COMMANDS + 3
    assert len(activity["commands"]) == events.MAX_ACTIVITY_COMMANDS
    assert activity["unique_command_count"] == events.MAX_ACTIVITY_COMMANDS + 3
    assert activity["commands_truncated"] is True


def test_parse_usage_activity_counts_usage_objects_without_retaining_them(tmp_path):
    from skills.harness import events

    events_path = tmp_path / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as stream:
        for index in range(1205):
            stream.write(json.dumps({"type": "turn", "usage": {"total_tokens": index}}) + "\n")

    usage, _activity = events.parse_usage_and_activity_data(events_path)

    assert usage["usage_objects_seen"] == 1205


def test_parse_usage_activity_bounds_distinct_tracking_structures(tmp_path, monkeypatch):
    from skills.harness import events

    monkeypatch.setattr(events, "MAX_TRACKED_EVENT_TYPES", 2)
    monkeypatch.setattr(events, "MAX_TRACKED_COMMAND_PREFIXES", 2)
    monkeypatch.setattr(events, "MAX_TRACKED_UNIQUE_COMMANDS", 2)
    events_path = tmp_path / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as stream:
        for index in range(5):
            stream.write(
                json.dumps(
                    {
                        "type": f"event-{index}",
                        "cmd": f"tool-{index} --arg",
                        "input_tokens": index,
                        "output_tokens": index + 10,
                    }
                )
                + "\n"
            )

    usage, activity = events.parse_usage_and_activity_data(events_path)

    assert usage["max_input_tokens"] == 4
    assert usage["max_output_tokens"] == 14
    assert len(activity["event_types"]) == 2
    assert activity["event_types_truncated"] is True
    assert len(activity["command_prefix_counts"]) == 2
    assert activity["command_prefix_counts_truncated"] is True
    assert activity["unique_command_count"] == 2
    assert activity["unique_commands_truncated"] is True


def test_prepare_input_workspace_rejects_symlink_escaping_input(tmp_path):
    from skills.harness.container.agent_run import AgentRunConfig, prepare_input_workspace

    job = tmp_path / "job"
    job.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    job.joinpath("escape.txt").symlink_to(outside)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=job,
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=tmp_path / "run",
        prompt_source=tmp_path / "prompt.txt",
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="test",
        agent_model="test-model",
        agent_home=tmp_path / ".agent",
        agent_model_was_explicit=False,
    )

    try:
        prepare_input_workspace(config)
    except RuntimeError as exc:
        assert "symlink escapes input directory" in str(exc)
    else:
        raise AssertionError("job input symlinks that escape the input directory should be rejected")


def test_prepare_input_workspace_dereferences_safe_symlinks(tmp_path):
    if not hasattr(os, "symlink"):
        return
    from skills.harness.container.agent_run import AgentRunConfig, prepare_input_workspace

    job = tmp_path / "job"
    job.mkdir()
    job.joinpath("source.txt").write_text("inside\n", encoding="utf-8")
    job.joinpath("link.txt").symlink_to(job / "source.txt")
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=job,
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=tmp_path / "run",
        prompt_source=tmp_path / "prompt.txt",
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="test",
        agent_model="test-model",
        agent_home=tmp_path / ".agent",
        agent_model_was_explicit=False,
    )

    prepare_input_workspace(config)

    copied_input = config.run_input_dir / "link.txt"
    copied_workspace = config.run_workspace_dir / "link.txt"
    assert copied_input.read_text(encoding="utf-8") == "inside\n"
    assert copied_workspace.read_text(encoding="utf-8") == "inside\n"
    assert not copied_input.is_symlink()
    assert not copied_workspace.is_symlink()


def test_validate_input_symlinks_does_not_traverse_symlinked_directory_loop(tmp_path):
    if not hasattr(os, "symlink"):
        return
    from skills.harness.container.agent_run import validate_input_symlinks

    job = tmp_path / "job"
    job.mkdir()
    job.joinpath("loop").symlink_to(job, target_is_directory=True)

    validate_input_symlinks(job)


def test_prepare_prompt_hashes_copied_prompt_file(tmp_path):
    from skills.harness.container.agent_run import AgentRunConfig, prepare_prompt

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("convert this job\n", encoding="utf-8")
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=tmp_path / "run",
        prompt_source=prompt,
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="test",
        agent_model="test-model",
        agent_home=tmp_path / ".agent",
        agent_model_was_explicit=False,
    )

    prepare_prompt(config)

    copied = config.prompt_file_path.read_bytes()
    metadata = json.loads((result_dir / "prompt_metadata.json").read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(copied).hexdigest()
    assert metadata["template_sha256"] == expected_hash
    assert metadata["prompt_sha256"] == expected_hash
    assert metadata["template_bytes"] == len(copied)
    assert metadata["prompt_bytes"] == len(copied)
    assert metadata["verbatim_copy"] is True


def test_setup_skill_availability_allows_missing_optional_metadata(tmp_path):
    from skills.harness.container.agent_run import AgentRunConfig, setup_skill_availability

    codex_home = tmp_path / ".codex"
    result_dir = tmp_path / "results"
    skill_dir = codex_home / "skills" / "nvflare-convert-pytorch"
    skill_dir.mkdir(parents=True)
    result_dir.mkdir()

    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=tmp_path / "run",
        prompt_source=tmp_path / "prompt.txt",
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="codex",
        agent_model="test-model",
        agent_home=codex_home,
        agent_model_was_explicit=True,
    )

    setup_skill_availability(config)

    state = json.loads((result_dir / "skills_state.json").read_text(encoding="utf-8"))
    missing = json.loads((result_dir / "skills_metadata_missing.json").read_text(encoding="utf-8"))
    assert state["status"] == "prepared"
    assert state["skills_enabled"] is True
    assert sorted(Path(item).name for item in missing["missing"]) == [
        "skills_build_install.json",
        "skills_list.json",
    ]
    assert not (result_dir / "skills_build_install.json").exists()


def test_skill_exposure_carries_launch_args_and_environment(tmp_path):
    from skills.harness.agents.base import SkillExposureSpec
    from skills.harness.container.skills import apply_skill_exposure

    skill_root = tmp_path / "skills"
    (skill_root / "nvflare-convert-pytorch").mkdir(parents=True)
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    result = apply_skill_exposure(
        spec=SkillExposureSpec(
            mechanism_type="launch_flag",
            container_home=tmp_path,
            skill_root=skill_root,
            launch_args=["--add-dir", str(skill_root)],
            environment={"AGENT_SKILLS_DIR": str(skill_root)},
        ),
        skills_enabled=True,
        result_dir=result_dir,
        sdk_image_kind="test-skills",
    )

    state = json.loads((result_dir / "skills_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "prepared"
    assert result.status == "prepared"
    assert result.launch_args == ["--add-dir", str(skill_root)]
    assert result.environment == {"AGENT_SKILLS_DIR": str(skill_root)}


def test_skill_exposure_action_timeout_writes_failure_state(tmp_path, monkeypatch):
    import subprocess

    from skills.harness.container import skills

    result_dir = tmp_path / "results"
    result_dir.mkdir()

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output="partial output\n")

    monkeypatch.setattr(skills.subprocess, "run", timeout_run)

    try:
        skills.run_exposure_action(["agent", "skills", "install"], result_dir, "setup_action", {})
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("timed out exposure action should abort skill exposure")

    state = json.loads((result_dir / "skills_state.json").read_text(encoding="utf-8"))
    output = (result_dir / "skills_setup_action_output.txt").read_text(encoding="utf-8")
    assert state["reason"] == "action_timeout"
    assert state["action_name"] == "setup_action"
    assert state["timeout_seconds"] == skills.EXPOSURE_ACTION_TIMEOUT_SECONDS
    assert state["output_ref"].endswith("skills_setup_action_output.txt")
    assert output == "partial output\n"


def test_skill_exposure_rejects_skill_root_outside_container_home(tmp_path):
    from skills.harness.agents.base import SkillExposureSpec
    from skills.harness.container.skills import apply_skill_exposure

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    outside = tmp_path / "outside"

    try:
        apply_skill_exposure(
            spec=SkillExposureSpec(
                mechanism_type="preinstalled_home",
                container_home=tmp_path / "agent_home",
                skill_root=outside,
            ),
            skills_enabled=False,
            result_dir=result_dir,
            sdk_image_kind="test-baseline",
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("out-of-scope skill_root should fail before removal")

    state = json.loads((result_dir / "skills_state.json").read_text(encoding="utf-8"))
    assert state["reason"] == "skill_root_outside_container_home"
    assert state["skill_root"] == str(outside)


def test_skill_exposure_rejects_metadata_file_outside_container_home(tmp_path):
    from skills.harness.agents.base import SkillExposureSpec
    from skills.harness.container.skills import apply_skill_exposure

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    container_home = tmp_path / "agent_home"
    skill_root = container_home / "skills"
    (skill_root / "nvflare-convert-pytorch").mkdir(parents=True)
    outside_metadata = tmp_path / "outside.json"
    outside_metadata.write_text("{}\n", encoding="utf-8")

    try:
        apply_skill_exposure(
            spec=SkillExposureSpec(
                mechanism_type="preinstalled_home",
                container_home=container_home,
                skill_root=skill_root,
                metadata_files=[outside_metadata],
            ),
            skills_enabled=True,
            result_dir=result_dir,
            sdk_image_kind="test-skills",
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("metadata files outside container_home should not be copied to results")

    state = json.loads((result_dir / "skills_state.json").read_text(encoding="utf-8"))
    assert state["reason"] == "metadata_file_outside_container_home"
    assert state["metadata_file"] == str(outside_metadata)


def test_skill_exposure_requires_container_home_when_skill_root_is_configured(tmp_path):
    from skills.harness.agents.base import SkillExposureSpec
    from skills.harness.container.skills import apply_skill_exposure

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    skill_root = tmp_path / "skills"

    try:
        apply_skill_exposure(
            spec=SkillExposureSpec(mechanism_type="preinstalled_home", skill_root=skill_root),
            skills_enabled=False,
            result_dir=result_dir,
            sdk_image_kind="test-baseline",
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("configured skill_root without container_home should fail before removal")

    state = json.loads((result_dir / "skills_state.json").read_text(encoding="utf-8"))
    assert state["reason"] == "container_home_required_for_skill_root"


def test_skill_exposure_rejects_bundled_root_outside_workspace(tmp_path):
    from skills.harness.agents.base import SkillExposureSpec
    from skills.harness.container.skills import apply_skill_exposure

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    container_home = tmp_path / "home"
    skill_root = container_home / "skills"

    try:
        apply_skill_exposure(
            spec=SkillExposureSpec(
                mechanism_type="preinstalled_home",
                container_home=container_home,
                skill_root=skill_root,
                disable_packaged_source=True,
            ),
            skills_enabled=False,
            result_dir=result_dir,
            sdk_image_kind="test-baseline",
            bundled_skills_root=lambda: str(tmp_path / "unsafe-bundled-source"),
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("out-of-scope bundled skill root should fail before removal")

    state = json.loads((result_dir / "skills_state.json").read_text(encoding="utf-8"))
    assert state["reason"] == "bundled_skill_source_outside_workspace"


def test_copy_optional_metadata_files_preserves_generic_names(tmp_path):
    from skills.harness.container.agent_run import copy_optional_metadata_files

    source_dir = tmp_path / "source"
    result_dir = tmp_path / "results"
    source_dir.mkdir()
    result_dir.mkdir()
    (source_dir / "skills_list.json").write_text('{"installed": []}\n', encoding="utf-8")

    payload = copy_optional_metadata_files(
        source_dir,
        result_dir,
        ("skills_list.json", "skills_build_install.json"),
    )

    assert (result_dir / "skills_list.json").read_text(encoding="utf-8") == '{"installed": []}\n'
    assert payload["copied"] == [
        {
            "source": str(source_dir / "skills_list.json"),
            "target": str(result_dir / "skills_list.json"),
        }
    ]
    assert payload["missing"] == [str(source_dir / "skills_build_install.json")]


def test_copy_optional_metadata_files_rejects_path_traversal(tmp_path):
    from skills.harness.container.agent_run import copy_optional_metadata_files

    source_dir = tmp_path / "source"
    result_dir = tmp_path / "results"
    source_dir.mkdir()
    result_dir.mkdir()

    try:
        copy_optional_metadata_files(source_dir, result_dir, ("../unsafe.json",))
    except ValueError as exc:
        assert "path separators" in str(exc)
    else:
        raise AssertionError("metadata file names must not traverse outside the source dir")


def test_available_skill_names_skips_symlinked_skill_directories(tmp_path, monkeypatch):
    from skills.harness import records

    agent_home = tmp_path / "agent-home"
    skills_root = agent_home / "skills"
    outside = tmp_path / "outside-skills"
    skills_root.mkdir(parents=True)
    outside.mkdir()
    (skills_root / "nvflare-real").mkdir()
    try:
        (skills_root / "nvflare-linked").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        return
    monkeypatch.setenv("BENCHMARK_AGENT_HOME", str(agent_home))

    assert records.available_skill_names() == {"nvflare-real"}


def test_infer_from_events_scores_installed_skill_names_in_single_pass(monkeypatch):
    from skills.harness import records

    monkeypatch.setattr(records, "available_skill_names", lambda: {"nvflare-a", "nvflare-b"})

    inferred = records.infer_from_events("nvflare-a did work. nvflare-b helped. nvflare-b finished.")

    assert inferred["skill"] == "nvflare-b"
    assert inferred["skill_source"] == "installed_skill_name_seen_in_events"


def test_infer_from_events_caps_installed_skill_name_candidates(monkeypatch):
    from skills.harness import records

    skill_names = {f"nvflare-skill-{index:03d}" for index in range(75)}
    monkeypatch.setattr(records, "available_skill_names", lambda: skill_names)

    inferred = records.infer_from_events("nvflare-skill-001 did work.")

    assert inferred["parser_warnings"]
    assert "capped at 50 candidates" in inferred["parser_warnings"][0]


def test_infer_from_events_uses_case_id_boundaries(monkeypatch):
    from skills.harness import records

    monkeypatch.setattr(records, "available_skill_names", lambda: {"nvflare-a"})
    monkeypatch.setattr(records, "eval_case_ids_for_skill", lambda _skill: ["basic", "basic-v2"])

    inferred = records.infer_from_events("nvflare-a ran case basic-v2 successfully.")

    assert inferred["case_id"] == "basic-v2"


def test_eval_case_ids_for_skill_skips_oversized_evals_json(tmp_path, monkeypatch):
    from skills.harness import records

    agent_home = tmp_path / "agent-home"
    evals_dir = agent_home / "skills" / "nvflare-a" / "evals"
    evals_dir.mkdir(parents=True)
    with evals_dir.joinpath("evals.json").open("wb") as stream:
        stream.truncate(records.MAX_EVALS_FILE_BYTES + 1)
    monkeypatch.setenv("BENCHMARK_AGENT_HOME", str(agent_home))
    monkeypatch.setattr(
        records,
        "load_json",
        lambda _path, default=None: (_ for _ in ()).throw(AssertionError("oversized evals should not load")),
    )

    assert records.eval_case_ids_for_skill("nvflare-a") == []


def test_load_text_caps_bytes(tmp_path):
    from skills.harness.records import load_text

    path = tmp_path / "events.jsonl"
    path.write_bytes(b"abcdef")

    assert load_text(path, max_bytes=3) == "abc"


def test_synthesize_agent_record_caps_event_text_for_identity(tmp_path, monkeypatch):
    from skills.harness import records

    monkeypatch.setattr(records, "MAX_EVENTS_TEXT_BYTES", 8)
    monkeypatch.setattr(records, "available_skill_names", lambda: {"nvflare-large"})
    monkeypatch.setattr(records, "eval_case_ids_for_skill", lambda _skill: [])
    records_dir = tmp_path / "records"
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    usage_path = tmp_path / "usage.json"
    activity_path = tmp_path / "activity.json"
    events_path = tmp_path / "agent_events.jsonl"
    last_message_path = tmp_path / "agent_last_message.txt"
    workspace_delta_path = tmp_path / "workspace_delta_manifest.json"
    usage_path.write_text("{}\n", encoding="utf-8")
    activity_path.write_text("{}\n", encoding="utf-8")
    events_path.write_text("prefixxx nvflare-large should not be read\n", encoding="utf-8")
    last_message_path.write_text("done\n", encoding="utf-8")
    workspace_delta_path.write_text("{}\n", encoding="utf-8")
    agent_record_path = records_dir / "with_skills_agent_record.json"

    records.synthesize_agent_record(
        records.AgentRecordSynthesisInputs(
            agent_record_path=agent_record_path,
            records_dir=records_dir,
            events_path=events_path,
            usage_path=usage_path,
            activity_path=activity_path,
            last_message_path=last_message_path,
            input_dir=input_dir,
            mode="with_skills",
            elapsed_seconds=1,
            agent_exit=0,
            skills_enabled=True,
            skill_run_mode="with_skills",
            agent="codex",
            agent_model="test-model",
            run_start_time_ns=0,
            workspace_delta_manifest_path=workspace_delta_path,
        )
    )

    record = json.loads(agent_record_path.read_text(encoding="utf-8"))
    assert "skill" not in record
    assert record["event_identity_inference"]["skill"] == ""


def test_iter_json_records_enforces_file_count_limit(tmp_path, monkeypatch):
    from skills.harness import records

    for index in range(3):
        (tmp_path / f"record-{index}.json").write_text(
            json.dumps({"skill": "nvflare-test", "case_id": f"case-{index}"}),
            encoding="utf-8",
        )
    monkeypatch.setattr(records, "MAX_JSON_RECORD_FILES", 1)

    loaded = list(records.iter_json_records(tmp_path))

    assert len(loaded) == 1


def test_iter_json_records_skips_oversized_json_files(tmp_path, monkeypatch):
    from skills.harness import records

    small = tmp_path / "small.json"
    small.write_text(json.dumps({"skill": "nvflare-test"}), encoding="utf-8")
    (tmp_path / "large.json").write_text(
        json.dumps({"skill": "nvflare-large-value-that-exceeds-the-limit"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(records, "MAX_JSON_RECORD_FILE_BYTES", small.stat().st_size)

    loaded = list(records.iter_json_records(tmp_path))

    assert [record["skill"] for _path, record in loaded] == ["nvflare-test"]


def test_iter_json_records_file_count_limit_ignores_oversized_skips(tmp_path, monkeypatch):
    from skills.harness import records

    small = tmp_path / "small.json"
    small.write_text(json.dumps({"skill": "nvflare-test"}), encoding="utf-8")
    oversized = tmp_path / "oversized.json"
    oversized.write_text(json.dumps({"skill": "nvflare-oversized", "payload": "too-large"}), encoding="utf-8")
    monkeypatch.setattr(records, "MAX_JSON_RECORD_FILES", 1)
    monkeypatch.setattr(records, "MAX_JSON_RECORD_FILE_BYTES", small.stat().st_size)

    loaded = list(records.iter_json_records(tmp_path))

    assert [record["skill"] for _path, record in loaded] == ["nvflare-test"]


def test_iter_json_records_does_not_traverse_symlinked_directories(tmp_path):
    if not hasattr(os, "symlink"):
        return
    from skills.harness import records

    outside = tmp_path / "outside"
    outside.mkdir()
    outside.joinpath("external.json").write_text(json.dumps({"skill": "external"}), encoding="utf-8")
    records_root = tmp_path / "records"
    records_root.mkdir()
    records_root.joinpath("local.json").write_text(json.dumps({"skill": "local"}), encoding="utf-8")
    records_root.joinpath("linked").symlink_to(outside, target_is_directory=True)

    loaded = list(records.iter_json_records(records_root))

    assert [record["skill"] for _path, record in loaded] == ["local"]


def test_login_shell_runtime_probe_uses_configured_venv_path(monkeypatch):
    from skills.harness.container import agent_run

    class ProbeResult:
        returncode = 0
        stdout = "\n".join(
            [
                "PATH=/custom/venv/bin:/usr/bin",
                "python=/custom/venv/bin/python",
                "sdk_import_name=example_sdk",
                "sdk_import_version=9.9",
            ]
        )

    class VersionResult:
        returncode = 0
        stdout = "Example SDK 9.9\n"

    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        return ProbeResult() if len(calls) == 1 else VersionResult()

    monkeypatch.setenv("BENCHMARK_CONTAINER_VENV_DIR", "/custom/venv")
    monkeypatch.setenv("SDK_IMPORT_NAME", "example_sdk")
    monkeypatch.setenv("SDK_VERSION_COMMAND", "example-sdk --version")
    monkeypatch.setattr(agent_run.subprocess, "run", fake_run)

    probe = agent_run.login_shell_runtime_probe()

    assert probe["ok"] is True
    assert probe["expected_python"] == "/custom/venv/bin/python"
    assert probe["sdk_import_name"] == "example_sdk"
    assert probe["sdk_import_version"] == "9.9"
    assert calls[1] == ["example-sdk", "--version"]
    assert probe["sdk_version_exit_code"] == 0
    assert probe["sdk_version_output"] == "Example SDK 9.9"


def test_login_shell_runtime_probe_does_not_shell_interpolate_sdk_version_command(monkeypatch):
    from skills.harness.container import agent_run

    class ProbeResult:
        returncode = 0
        stdout = "\n".join(["PATH=/custom/venv/bin:/usr/bin", "python=/custom/venv/bin/python"])

    class VersionResult:
        returncode = 0
        stdout = "version output\n"

    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        return ProbeResult() if len(calls) == 1 else VersionResult()

    monkeypatch.setenv("BENCHMARK_CONTAINER_VENV_DIR", "/custom/venv")
    monkeypatch.setenv("SDK_VERSION_COMMAND", "example-sdk --version; rm -rf /workspace")
    monkeypatch.setattr(agent_run.subprocess, "run", fake_run)

    probe = agent_run.login_shell_runtime_probe()

    assert calls[0] == ["/bin/bash", "-lc", calls[0][2]]
    assert "rm -rf" not in calls[0][2]
    assert calls[1] == ["example-sdk", "--version;", "rm", "-rf", "/workspace"]
    assert probe["sdk_version_output"] == "version output"


def test_runtime_metadata_skips_login_shell_probe_when_launch_does_not_require_it(tmp_path, monkeypatch):
    from skills.harness.agents.base import AgentLaunchSpec
    from skills.harness.container import agent_run
    from skills.harness.container.agent_run import AgentRunConfig, persist_container_runtime_metadata

    result_dir = tmp_path / "results"
    agent_home = tmp_path / ".agent"
    result_dir.mkdir()
    agent_home.mkdir()
    (agent_home / "sdk_wheel_metadata.json").write_text(
        json.dumps({"sdk_name": "example", "variant": "skills"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=tmp_path / "run",
        prompt_source=tmp_path / "prompt.txt",
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="test",
        agent_model="test-model",
        agent_home=agent_home,
        agent_model_was_explicit=False,
    )
    launch = AgentLaunchSpec(
        argv=["agent", "run"],
        cwd=tmp_path,
        prompt_file=tmp_path / "prompt.txt",
        prompt_input_mode="stdin",
        stdout_events_dest=result_dir / "agent_events.jsonl",
        stderr_dest=result_dir / "agent_stderr.txt",
        final_message_dest=result_dir / "agent_last_message.txt",
        login_shell=False,
    )
    monkeypatch.setattr(agent_run, "command_output", lambda _command: None)

    def fail_probe():
        raise AssertionError("login shell probe should be skipped")

    monkeypatch.setattr(agent_run, "login_shell_runtime_probe", fail_probe)

    persist_container_runtime_metadata(config, launch)

    metadata = json.loads((result_dir / "runtime_image.json").read_text(encoding="utf-8"))
    assert metadata["login_shell_required"] is False
    assert metadata["login_shell_runtime_probe"]["reason"] == "skipped_adapter_does_not_use_login_shell"
    assert metadata["sdk_wheel_metadata"]["sdk_name"] == "example"
    assert (result_dir / "sdk_wheel_metadata.json").is_file()


def test_finalize_timing_uses_named_lifecycle_epochs(tmp_path):
    from skills.harness.common import write_json
    from skills.harness.timing import LifecycleEpochs, finalize_timing

    summary_path = tmp_path / "run_summary.json"
    record_path = tmp_path / "record.json"
    activity_path = tmp_path / "agent_activity.json"
    timing_path = tmp_path / "timing.json"
    write_json(summary_path, {"process_metrics": {}})
    write_json(record_path, {"process_metrics": {}})
    write_json(activity_path, {"event_count": 3, "command_count": 2})

    finalize_timing(
        summary_path,
        record_path,
        timing_path,
        activity_path,
        LifecycleEpochs(
            script_start=10,
            skill_availability_start=11,
            skill_availability_end=13,
            input_copy_start=13,
            input_copy_end=17,
            prompt_prep_start=18,
            prompt_prep_end=20,
            agent_start=21,
            agent_end=31,
            post_process_start=31,
            post_process_end=35,
            report_outcome_start=36,
            report_outcome_end=37,
            script_end=40,
        ),
    )

    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert timing["phase_seconds"] == {
        "container_elapsed_seconds": 30,
        "setup_elapsed_seconds": 11,
        "skill_exposure_elapsed_seconds": 2,
        "input_copy_elapsed_seconds": 4,
        "prompt_prepare_elapsed_seconds": 2,
        "agent_elapsed_seconds": 10,
        "post_process_elapsed_seconds": 4,
        "report_elapsed_seconds": 1,
    }
    assert summary["activity"]["event_count"] == 3
    assert summary["activity"]["command_count"] == 2
    assert record["process_metrics"]["phase_seconds"]["agent_elapsed_seconds"] == 10


def test_finalize_timing_creates_missing_output_parents(tmp_path):
    from skills.harness.common import write_json
    from skills.harness.timing import LifecycleEpochs, finalize_timing

    summary_path = tmp_path / "run_summary.json"
    timing_path = tmp_path / "timing.json"
    activity_path = tmp_path / "agent_activity.json"
    record_path = tmp_path / "missing_parent" / "record.json"
    write_json(summary_path, {"process_metrics": {}})
    write_json(activity_path, {})

    finalize_timing(
        summary_path,
        record_path,
        timing_path,
        activity_path,
        LifecycleEpochs(
            script_start=10,
            skill_availability_start=11,
            skill_availability_end=13,
            input_copy_start=13,
            input_copy_end=17,
            prompt_prep_start=18,
            prompt_prep_end=20,
            agent_start=21,
            agent_end=31,
            post_process_start=31,
            post_process_end=35,
            report_outcome_start=36,
            report_outcome_end=37,
            script_end=40,
        ),
    )

    assert record_path.is_file()
    assert json.loads(record_path.read_text(encoding="utf-8"))["process_metrics"]["agent_elapsed_seconds"] == 10
    assert json.loads(timing_path.read_text(encoding="utf-8"))["phase_seconds"]["agent_elapsed_seconds"] == 10


def test_atomic_json_writer_cleans_staged_temp_file_when_dump_fails(tmp_path, monkeypatch):
    from skills.harness import timing

    def fail_json_dump(*_args, **_kwargs):
        raise TypeError("cannot serialize")

    monkeypatch.setattr(timing.json, "dump", fail_json_dump)

    try:
        timing._write_json_files_atomic({tmp_path / "out.json": {"bad": object()}})
    except TypeError:
        pass
    else:
        raise AssertionError("json dump failure should propagate")

    assert list(tmp_path.iterdir()) == []


def test_write_failure_record_outputs_early_failure_artifacts(tmp_path):
    from skills.harness.container.agent_run import write_failure_record

    result_dir = tmp_path / "results"
    records_dir = result_dir / "records"

    exit_code = write_failure_record(
        result_dir=result_dir,
        records_dir=records_dir,
        mode="with_skills",
        exit_code=2,
        error_type="RuntimeError",
        message="prompt missing",
        phase="input_validation",
        agent="codex",
        agent_model="test-model",
        skills_enabled=True,
    )

    record = json.loads((records_dir / "with_skills_record.json").read_text(encoding="utf-8"))
    early = json.loads((result_dir / "early_failure.json").read_text(encoding="utf-8"))
    summary = json.loads((result_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert exit_code == 2
    assert record["harness_failure"] is True
    assert record["harness_error"]["phase"] == "input_validation"
    assert record["final_container_exit_code"] == 2
    assert early["record_path"] == str(records_dir / "with_skills_record.json")
    assert summary["harness_failure"] is True
    assert summary["final_container_exit_code"] == 2


def test_write_failure_record_defaults_to_unknown_agent(tmp_path):
    from skills.harness.container.agent_run import write_failure_record

    result_dir = tmp_path / "results"
    records_dir = result_dir / "records"

    write_failure_record(
        result_dir=result_dir,
        records_dir=records_dir,
        mode="with_skills",
        exit_code=2,
        error_type="RuntimeError",
        message="config failed",
        phase="config",
    )

    record = json.loads((records_dir / "with_skills_record.json").read_text(encoding="utf-8"))
    assert record["agent"] == "unknown"
    assert record["process_metrics"]["agent_elapsed_seconds"] == 0


def test_merge_harness_failure_preserves_existing_record(tmp_path):
    from skills.harness.common import write_json
    from skills.harness.container.agent_run import AgentRunConfig, merge_harness_failure

    result_dir = tmp_path / "results"
    records_dir = result_dir / "records"
    records_dir.mkdir(parents=True)
    final_record = records_dir / "with_skills_record.json"
    write_json(
        final_record,
        {
            "mode": "with_skills",
            "agent_process_passed": True,
            "agent_process_exit_code": 0,
            "process_metrics": {"elapsed_seconds": 12},
        },
    )
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=records_dir,
        run_root=tmp_path / "run",
        prompt_source=tmp_path / "prompt.txt",
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="codex",
        agent_model="test-model",
        agent_home=tmp_path / ".codex",
        agent_model_was_explicit=True,
    )

    exit_code = merge_harness_failure(config, RuntimeError("post process failed"), 1, "post_process")

    record = json.loads(final_record.read_text(encoding="utf-8"))
    late = json.loads((result_dir / "late_harness_failure.json").read_text(encoding="utf-8"))
    summary = json.loads((result_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert record["agent_process_passed"] is True
    assert record["process_metrics"]["elapsed_seconds"] == 12
    assert record["harness_error"]["phase"] == "post_process"
    assert record["harness_errors"][0]["message"] == "post process failed"
    assert late["preserved_existing_record"] is True
    assert summary["harness_failure"] is True
    assert summary["harness_errors"][0]["phase"] == "post_process"


def test_pair_result_root_cleanup_removes_legacy_eval_artifacts(tmp_path):
    from skills.harness.host.runner import clean_pair_result_root

    result_root = tmp_path / "result"
    result_root.mkdir()
    for name in ("with_skills_eval_on", "with_skills_eval_off", "process_eval_runs", "without_skills", "with_skills"):
        path = result_root / name
        path.mkdir()
        path.joinpath("old.txt").write_text("stale\n", encoding="utf-8")
    result_root.joinpath("comprehensive_report.md").write_text("Benchmark Metrics Comparison\n", encoding="utf-8")
    result_root.joinpath("metrics_summary.json").write_text("{}\n", encoding="utf-8")
    result_root.joinpath("user_note.txt").write_text("keep\n", encoding="utf-8")

    clean_pair_result_root(result_root)

    assert not result_root.joinpath("with_skills_eval_on").exists()
    assert not result_root.joinpath("with_skills_eval_off").exists()
    assert not result_root.joinpath("process_eval_runs").exists()
    assert not result_root.joinpath("without_skills").exists()
    assert not result_root.joinpath("with_skills").exists()
    assert not result_root.joinpath("comprehensive_report.md").exists()
    assert not result_root.joinpath("metrics_summary.json").exists()
    assert result_root.joinpath("user_note.txt").read_text(encoding="utf-8") == "keep\n"
