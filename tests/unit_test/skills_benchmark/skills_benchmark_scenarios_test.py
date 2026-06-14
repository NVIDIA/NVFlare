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
from pathlib import Path

import yaml


def write_prompt_and_job(tmp_path: Path) -> tuple[Path, Path]:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Convert this job with the workflow named in the benchmark prompt.\n", encoding="utf-8")
    job = tmp_path / "ames"
    job.mkdir(exist_ok=True)
    job.joinpath("README.md").write_text("Synthetic job fixture.\n", encoding="utf-8")
    return prompt, job


def base_scenario(tmp_path: Path) -> dict:
    prompt, job = write_prompt_and_job(tmp_path)
    return {
        "name": "ci smoke scaffold",
        "prompt": prompt.name,
        "agents": [{"name": "codex", "models": ["gpt-test"]}],
        "comparison": {"type": "mode_ablation", "modes": ["without_skills", "with_skills"]},
        "workflows": [{"name": "SCAFFOLD"}],
        "jobs": [{"path": job.name, "scale": "small"}],
    }


def test_mode_ablation_expands_modes_and_target_record_paths(tmp_path):
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    compilation = compile_scenario(raw, base_dir=tmp_path)
    scenario = compilation.scenario
    run_plan = compilation.run_plan
    entries = run_plan["entries"]

    expected_prompt_hash = hashlib.sha256((tmp_path / "prompt.txt").read_bytes()).hexdigest()
    assert scenario["name"] == "ci smoke scaffold"
    assert scenario["prompt"]["sha256"] == expected_prompt_hash
    assert scenario["reproducibility"]["prompt_hash"] == expected_prompt_hash
    assert scenario["reproducibility"]["image_targets"]["codex"]["skills"] == "agent-skills-benchmark:codex-skills"
    assert scenario["reproducibility"]["agent_versions"]["codex"]["AGENT_CLI_NAME"] == "codex"
    assert scenario["reproducibility"]["agent_versions"]["codex"]["AGENT_VERSION_COMMAND"] == "codex --version"
    assert scenario["reproducibility"]["wheel_variants"] == [
        "local_wheel_with_preinstalled_skills",
        "local_wheel_without_packaged_skills",
    ]
    assert run_plan["run_count"] == 2
    assert run_plan["comparison_group_count"] == 1
    assert [entry["mode"] for entry in entries] == [
        "without_skills",
        "with_skills",
    ]
    assert {entry["prompt_hash"] for entry in entries} == {expected_prompt_hash}
    assert entries[0]["record_dir"] == (
        "records/agent=codex/model=gpt_test/workflow=scaffold/job=ames/mode=without_skills"
    )
    assert entries[1]["skills_enabled"] is True
    assert run_plan["comparison_groups"][0]["compared_run_ids"] == ["run_00001", "run_00002"]


def test_compile_scenario_file_writes_scenario_and_run_plan(tmp_path):
    from skills.harness.scenarios import compile_scenario_file

    raw = base_scenario(tmp_path)
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output_dir = tmp_path / "compiled"

    compilation = compile_scenario_file(scenario_path)
    compilation.write(output_dir)

    scenario_json = json.loads((output_dir / "scenario.json").read_text(encoding="utf-8"))
    run_plan_json = json.loads((output_dir / "run_plan.json").read_text(encoding="utf-8"))
    assert scenario_json["source_path"] == str(scenario_path.resolve())
    assert run_plan_json["source_path"] == str(scenario_path.resolve())
    assert run_plan_json["run_count"] == 2


def test_scenario_reproducibility_records_profile_image_targets(tmp_path):
    from skills.harness.scenarios import compile_scenario

    scenario = compile_scenario(base_scenario(tmp_path), base_dir=tmp_path).scenario

    assert scenario["reproducibility"]["image_targets"]["codex"] == {
        "skills": "agent-skills-benchmark:codex-skills",
        "baseline": "agent-skills-benchmark:codex-baseline",
        "report": "agent-skills-benchmark:codex-skills",
    }


def test_prompt_path_must_stay_inside_scenario_directory(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    base_dir = tmp_path / "scenario"
    base_dir.mkdir()
    raw = base_scenario(base_dir)
    outside_prompt = tmp_path / "outside_prompt.txt"
    outside_prompt.write_text("secret prompt\n", encoding="utf-8")
    raw["prompt"] = str(outside_prompt)

    try:
        compile_scenario(raw, base_dir=base_dir)
    except ScenarioValidationError as exc:
        assert "Prompt file must stay within scenario directory" in str(exc)
    else:
        raise AssertionError("absolute prompt paths outside the scenario directory should be rejected")


def test_prompt_template_renders_only_explicit_variables(tmp_path):
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    template = tmp_path / "prompt_template.txt"
    template.write_text("Convert {job_name} with metric {metric}.\n", encoding="utf-8")
    raw["prompt"] = {
        "template": template.name,
        "variables": {"job_name": "ames", "metric": "AUROC"},
    }

    compilation = compile_scenario(raw, base_dir=tmp_path)
    prompt = compilation.scenario["prompt"]
    rendered_bytes = b"Convert ames with metric AUROC.\n"
    expected_hash = hashlib.sha256(rendered_bytes).hexdigest()

    assert prompt["source_type"] == "template"
    assert prompt["source_path"] == str(template.resolve())
    assert prompt["sha256"] == expected_hash
    assert prompt["rendered_sha256"] == expected_hash
    assert prompt["bytes"] == len(rendered_bytes)
    assert prompt["path"] == str(template.resolve())
    assert not (tmp_path / ".agent_benchmark").exists()
    assert {entry["prompt_hash"] for entry in compilation.run_plan["entries"]} == {expected_hash}

    result_root = tmp_path / "results"
    materialized = compilation.write(result_root)
    written_scenario = json.loads((result_root / "scenario.json").read_text(encoding="utf-8"))
    written_run_plan = json.loads((result_root / "run_plan.json").read_text(encoding="utf-8"))
    rendered_path = Path(written_scenario["prompt"]["path"])

    assert rendered_path.is_relative_to((result_root / ".agent_benchmark" / "rendered_prompts").resolve())
    assert rendered_path.read_bytes() == rendered_bytes
    assert "_rendered_text" not in written_scenario["prompt"]
    assert {entry["prompt_source"] for entry in written_run_plan["entries"]} == {str(rendered_path)}
    assert {entry["prompt_source"] for entry in materialized.run_plan["entries"]} == {str(rendered_path)}

    second_root = tmp_path / "second-results"
    second_materialized = materialized.write(second_root)
    second_scenario = json.loads((second_root / "scenario.json").read_text(encoding="utf-8"))
    second_run_plan = json.loads((second_root / "run_plan.json").read_text(encoding="utf-8"))
    second_rendered_path = Path(second_scenario["prompt"]["path"])
    assert second_rendered_path.is_relative_to((second_root / ".agent_benchmark" / "rendered_prompts").resolve())
    assert second_rendered_path.read_bytes() == rendered_bytes
    assert {entry["prompt_source"] for entry in second_run_plan["entries"]} == {str(second_rendered_path)}
    assert {entry["prompt_source"] for entry in second_materialized.run_plan["entries"]} == {str(second_rendered_path)}


def test_prompt_path_with_variables_is_rendered_as_template(tmp_path):
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Convert {job_name}.\n", encoding="utf-8")
    raw["prompt"] = {"path": prompt.name, "variables": {"job_name": "ames"}}

    compilation = compile_scenario(raw, base_dir=tmp_path)

    assert compilation.scenario["prompt"]["source_type"] == "template"
    assert compilation.scenario["prompt"]["path"] == str(prompt.resolve())
    materialized = compilation.write(tmp_path / "results")
    assert Path(materialized.scenario["prompt"]["path"]).read_text(encoding="utf-8") == "Convert ames.\n"


def test_inline_prompt_template_string_is_rejected(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["prompt"] = "Convert {job_name}."

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "prompt must be a file path, not an inline template string" in str(exc)
        assert "prompt.template" in str(exc)
    else:
        raise AssertionError("inline prompt template strings should be rejected with a targeted error")


def test_execute_run_plan_materializes_template_prompt_under_result_root(tmp_path, monkeypatch):
    from skills.harness.common import write_json
    from skills.harness.host import runner
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    raw["comparison"] = {"type": "one", "mode": "with_skills"}
    raw["path_budget"] = 400
    template = tmp_path / "prompt_template.txt"
    template.write_text("Convert {job_name}.\n", encoding="utf-8")
    raw["prompt"] = {"template": template.name, "variables": {"job_name": "ames"}}
    compilation = compile_scenario(raw, base_dir=tmp_path)
    result_root = tmp_path / "results"
    observed_prompt_paths = []

    def fake_run_case(config, *, logs=(), prefix=None):
        observed_prompt_paths.append(config.prompt_path)
        records_dir = config.result_dir / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            config.result_dir / "run_summary.json",
            {
                "agent_process_passed": True,
                "final_container_exit_code": 0,
                "source_input_immutable_policy": {"status": "pass"},
                "required_validation_metric_status": "not_required",
            },
        )
        write_json(config.result_dir / "container_exit_code.json", {"exit_code": 0})
        write_json(
            records_dir / f"{config.mode}_record.json",
            {
                "agent_process_passed": True,
                "final_container_exit_code": 0,
                "source_input_immutable_policy": {"status": "pass"},
                "process_metrics": {"agent_elapsed_seconds": 1},
            },
        )
        return 0

    monkeypatch.setattr(runner, "run_case_safely", fake_run_case)
    monkeypatch.setattr(runner, "inspect_docker_image", lambda image: (True, ""))

    runner.execute_run_plan(compilation, result_root=result_root)

    assert not (tmp_path / ".agent_benchmark").exists()
    assert len(observed_prompt_paths) == 1
    rendered_prompt = observed_prompt_paths[0]
    assert rendered_prompt.is_relative_to((result_root / ".agent_benchmark" / "rendered_prompts").resolve())
    assert rendered_prompt.read_text(encoding="utf-8") == "Convert ames.\n"
    run_plan = json.loads((result_root / "run_plan.json").read_text(encoding="utf-8"))
    assert run_plan["entries"][0]["prompt_source"] == str(rendered_prompt)


def test_prompt_template_rejects_missing_unsafe_and_non_scalar_variables(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    template = tmp_path / "prompt_template.txt"
    template.write_text("Convert {job.name} with {metric}.\n", encoding="utf-8")
    raw = base_scenario(tmp_path)
    raw["prompt"] = {"template": template.name, "variables": {"metric": "AUROC"}}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "attribute or index access" in str(exc)
    else:
        raise AssertionError("prompt templates should reject attribute access")

    template.write_text("Convert {job_name} with {metric}.\n", encoding="utf-8")
    raw["prompt"] = {"template": template.name, "variables": {"job_name": "ames"}}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "missing variable" in str(exc)
        assert "metric" in str(exc)
    else:
        raise AssertionError("prompt templates should reject missing variables")

    raw["prompt"] = {"template": template.name, "variables": {"job_name": "ames", "metric": ["AUROC"]}}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "prompt.variables.metric" in str(exc)
        assert "scalar" in str(exc)
    else:
        raise AssertionError("prompt templates should reject non-scalar variables")


def test_prompt_template_literal_brace_errors_explain_escaping(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    template = tmp_path / "prompt_template.txt"
    template.write_text('Return JSON like {"metric": "AUROC"}.\n', encoding="utf-8")
    raw["prompt"] = {"template": template.name}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        message = str(exc)
        assert "literal braces" in message
        assert "{{" in message
        assert "}}" in message
    else:
        raise AssertionError("prompt templates should explain literal brace escaping")

    template.write_text('Return JSON like {{"metric": "AUROC"}} for {job_name}.\n', encoding="utf-8")
    raw["prompt"] = {"template": template.name, "variables": {"job_name": "ames"}}

    compilation = compile_scenario(raw, base_dir=tmp_path)

    assert compilation.scenario["prompt"]["_rendered_text"] == 'Return JSON like {"metric": "AUROC"} for ames.\n'


def test_model_comparison_expands_comparison_models(tmp_path):
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    raw["agents"] = [{"name": "codex"}]
    raw["comparison"] = {
        "type": "model_comparison",
        "agent": "codex",
        "mode": "with_skills",
        "models": ["gpt-a", "gpt-b"],
    }
    run_plan = compile_scenario(raw, base_dir=tmp_path).run_plan

    assert run_plan["run_count"] == 2
    assert run_plan["comparison_group_count"] == 1
    assert [entry["agent_model"] for entry in run_plan["entries"]] == ["gpt-a", "gpt-b"]
    assert [entry["model_source"] for entry in run_plan["entries"]] == ["comparison", "comparison"]
    assert "model=gpt_a" in run_plan["entries"][0]["record_dir"]
    assert "model=gpt_b" in run_plan["entries"][1]["record_dir"]


def test_model_comparison_dedupes_overlapping_top_level_models(tmp_path):
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    raw["agents"] = [{"name": "codex", "models": ["gpt-a", "gpt-b"]}]
    raw["comparison"] = {
        "type": "model_comparison",
        "agent": "codex",
        "mode": "with_skills",
        "models": ["gpt-a", "gpt-b"],
    }
    run_plan = compile_scenario(raw, base_dir=tmp_path).run_plan

    assert "model=gpt_a/" in run_plan["entries"][0]["record_dir"]
    assert "model=gpt_b/" in run_plan["entries"][1]["record_dir"]


def test_model_slug_fallback_avoids_unhandled_missing_key():
    from skills.harness.scenarios import model_slug_for

    assert model_slug_for({"models": {}}, "codex", "gpt-test") == "gpt_test"


def test_model_slug_key_is_visible_and_agent_scoped():
    from skills.harness.scenarios import model_slug_for, model_slug_key

    codex_key = model_slug_key("codex", "shared/model")
    claude_key = model_slug_key("claude", "shared/model")
    slugs = {"models": {codex_key: "shared_model_codex", claude_key: "shared_model_claude"}}

    assert "\0" not in codex_key
    assert model_slug_for(slugs, "codex", "shared/model") == "shared_model_codex"
    assert model_slug_for(slugs, "claude", "shared/model") == "shared_model_claude"


def test_slug_collisions_hash_original_values(tmp_path):
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    job_a = tmp_path / "my-job"
    job_b = tmp_path / "my job"
    job_a.mkdir()
    job_b.mkdir()
    raw["jobs"] = [
        {"name": "my-job", "path": job_a.name, "scale": "small"},
        {"name": "my job", "path": job_b.name, "scale": "small"},
    ]
    entries = compile_scenario(raw, base_dir=tmp_path).run_plan["entries"]
    job_slugs = {entry["job_slug"] for entry in entries}

    assert len(job_slugs) == 2
    assert all(job_slug.startswith("my_job_") for job_slug in job_slugs)
    assert len({entry["record_dir"] for entry in entries}) == len(entries)


def test_missing_job_scale_is_rejected(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["jobs"][0].pop("scale")

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "jobs[0].scale" in str(exc)
    else:
        raise AssertionError("scenario validation must require explicit job scale")


def test_fail_fast_requires_boolean(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["fail_fast"] = "false"

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "fail_fast must be a boolean" in str(exc)
    else:
        raise AssertionError("scenario validation must reject string fail_fast values")


def test_quality_gate_override_is_recorded_and_applied(tmp_path):
    from skills.harness.scenarios import compile_scenario, quality_gate_failures

    raw = base_scenario(tmp_path)
    raw["quality_gate"] = {"required_validation_metric_status": ["present"]}

    compilation = compile_scenario(raw, base_dir=tmp_path)

    assert compilation.scenario["quality_gate"]["required_validation_metric_status"] == ["present"]
    assert compilation.run_plan["quality_gate"]["required_validation_metric_status"] == ["present"]
    failures = quality_gate_failures(
        {
            "agent_process_passed": True,
            "final_container_exit_code": 0,
            "source_input_immutable_policy": {"status": "pass"},
            "required_validation_metric_status": "not_required",
        },
        {},
        0,
        compilation.run_plan["quality_gate"],
    )
    assert "required_validation_metric_status=not_required" in failures


def test_quality_gate_rejects_unknown_fields(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["quality_gate"] = {"unknown_check": True}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "quality_gate.unknown_check" in str(exc)
    else:
        raise AssertionError("unknown scenario quality-gate fields should fail validation")


def test_resource_policy_non_integer_values_are_validation_errors(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["resource_policy"] = {"small": {"agent_timeout_seconds": "fast"}}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "resource_policy.small.agent_timeout_seconds" in str(exc)
        assert "must be an integer greater than 0" in str(exc)
    else:
        raise AssertionError("non-integer scenario resource policy values should fail validation")

    job_policy_case = tmp_path / "job-policy"
    job_policy_case.mkdir()
    raw = base_scenario(job_policy_case)
    raw["jobs"][0]["resource_policy"] = {"agent_timeout_seconds": None}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "jobs[0].resource_policy.agent_timeout_seconds" in str(exc)
        assert "must be an integer greater than 0" in str(exc)
    else:
        raise AssertionError("non-integer job resource policy values should fail validation")


def test_resource_policy_rejects_bool_and_non_positive_values(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["resource_policy"] = {"small": {"agent_timeout_seconds": False}}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "resource_policy.small.agent_timeout_seconds" in str(exc)
        assert "must be an integer greater than 0" in str(exc)
    else:
        raise AssertionError("boolean resource policy values should fail validation")

    zero_case = tmp_path / "zero-policy"
    zero_case.mkdir()
    raw = base_scenario(zero_case)
    raw["jobs"][0]["resource_policy"] = {"container_timeout_seconds": 0}

    try:
        compile_scenario(raw, base_dir=zero_case)
    except ScenarioValidationError as exc:
        assert "jobs[0].resource_policy.container_timeout_seconds" in str(exc)
        assert "must be an integer greater than 0" in str(exc)
    else:
        raise AssertionError("zero resource policy values should fail validation")


def test_resource_policy_rejects_unknown_fields(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["resource_policy"] = {"small": {"agent_timout_seconds": 60}}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "resource_policy.small.agent_timout_seconds" in str(exc)
        assert "not a supported resource policy field" in str(exc)
    else:
        raise AssertionError("unknown scenario resource policy fields should fail validation")

    typo_case = tmp_path / "typo-policy"
    typo_case.mkdir()
    raw = base_scenario(typo_case)
    raw["jobs"][0]["resource_policy"] = {"container_timout_seconds": 60}

    try:
        compile_scenario(raw, base_dir=typo_case)
    except ScenarioValidationError as exc:
        assert "jobs[0].resource_policy.container_timout_seconds" in str(exc)
        assert "not a supported resource policy field" in str(exc)
    else:
        raise AssertionError("unknown job resource policy fields should fail validation")


def test_prompt_file_size_guard_rejects_large_prompt(tmp_path):
    from skills.harness.scenarios import MAX_PROMPT_BYTES, ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    prompt_path = tmp_path / raw["prompt"]
    with prompt_path.open("wb") as stream:
        stream.truncate(MAX_PROMPT_BYTES + 1)

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "Prompt file exceeds max size" in str(exc)
    else:
        raise AssertionError("scenario prompt files should be size-limited before read_bytes")


def test_prompt_file_size_guard_uses_read_bytes_length(tmp_path, monkeypatch):
    from skills.harness.scenarios import MAX_PROMPT_BYTES, ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    prompt_path = tmp_path / raw["prompt"]
    original_read_bytes = Path.read_bytes

    def fake_read_bytes(path):
        if path == prompt_path:
            return b"x" * (MAX_PROMPT_BYTES + 1)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "Prompt file exceeds max size" in str(exc)
    else:
        raise AssertionError("scenario prompt files should be capped by bytes read, not a stale stat")


def test_validate_path_budget_uses_longest_path_not_lexicographic_max():
    from skills.harness.scenarios import ScenarioValidationError, validate_path_budget

    short_late_path = "zz"
    long_early_path = "a" * 80
    budget = len(str(Path("results") / "path-budget" / short_late_path)) + 1

    try:
        validate_path_budget(
            "path budget",
            [{"artifact_paths": {"short": short_late_path, "long": long_early_path}}],
            budget,
        )
    except ScenarioValidationError as exc:
        assert long_early_path in str(exc)
        assert "path_budget" in str(exc)
    else:
        raise AssertionError("path budget validation must check the longest rendered artifact path")


def test_validate_path_budget_can_use_actual_result_root():
    from skills.harness.scenarios import ScenarioValidationError, validate_path_budget

    result_root = Path("r" * 40)
    artifact_path = "short.txt"
    synthetic_budget = len(str(Path("results") / "path-budget" / artifact_path)) + 1

    try:
        validate_path_budget(
            "path budget",
            [{"artifact_paths": {"short": artifact_path}}],
            synthetic_budget,
            result_root=result_root,
        )
    except ScenarioValidationError as exc:
        assert str(result_root) in str(exc)
    else:
        raise AssertionError("path budget validation should include the actual result root when provided")


def test_validate_path_budget_allows_empty_artifact_paths():
    from skills.harness.scenarios import validate_path_budget

    validate_path_budget(
        "path budget",
        [{"artifact_paths": {}}, {}],
        100,
    )


def test_quality_gate_failures_reports_missing_final_exit_as_not_recorded():
    from skills.harness.scenarios import quality_gate_failures

    failures = quality_gate_failures(
        {"agent_process_passed": True, "source_input_immutable_policy": {"status": "pass"}},
        {},
        0,
    )

    assert "final_container_exit_code=not_recorded" in failures
    assert "final_container_exit_code=None" not in failures


def test_quality_gate_failures_derives_missing_required_validation_metric():
    from skills.harness.scenarios import quality_gate_failures

    record = {
        "agent_process_passed": True,
        "final_container_exit_code": 0,
        "source_input_immutable_policy": {"status": "pass"},
        "quality_signals": {
            "job_guidance_primary_validation_metric": {
                "expected_primary_metric": "AUROC",
                "metric_value_available": False,
                "reported_validation_metric": {"name": None, "value": None, "reported_values": []},
            }
        },
    }

    failures = quality_gate_failures({}, record, 0)

    assert "required_validation_metric_status=missing" in failures


def test_quality_gate_failures_derives_critical_quality_check_failure():
    from skills.harness.scenarios import quality_gate_failures

    record = {
        "agent_process_passed": True,
        "final_container_exit_code": 0,
        "source_input_immutable_policy": {"status": "pass"},
        "quality_checks": [{"severity": "critical", "status": "fail"}],
    }

    failures = quality_gate_failures({}, record, 0)

    assert "critical_quality_checks_failed" in failures


def test_known_pending_agent_is_rejected(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["agents"] = [{"name": "hermes"}]

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "BENCHMARK_AGENT='hermes'" in str(exc)
        assert "known but not implemented" in str(exc)
    else:
        raise AssertionError("known-pending agents should fail preflight")


def test_claude_scenario_uses_adapter_default_model_when_unspecified(tmp_path):
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    raw["agents"] = [{"name": "claude"}]
    raw["comparison"] = {"type": "one", "mode": "with_skills"}

    compilation = compile_scenario(raw, base_dir=tmp_path)
    entry = compilation.run_plan["entries"][0]

    assert entry["agent_model"] == "unspecified_default"
    assert entry["model_source"] == "adapter_default"


def test_agent_comparison_requires_unambiguous_model_selection(tmp_path):
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["agents"] = [{"name": "codex", "models": ["gpt-a", "gpt-b"]}]
    raw["comparison"] = {"type": "agent_comparison", "mode": "with_skills", "agents": ["codex"]}

    try:
        compile_scenario(raw, base_dir=tmp_path)
    except ScenarioValidationError as exc:
        assert "ambiguous" in str(exc)
        assert "models_by_agent" in str(exc)
    else:
        raise AssertionError("agent comparison must reject ambiguous model selection")


def test_agent_comparison_models_by_agent_resolves_single_model(tmp_path):
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    raw["agents"] = [{"name": "codex", "models": ["gpt-a", "gpt-b"]}]
    raw["comparison"] = {
        "type": "agent_comparison",
        "mode": "with_skills",
        "agents": ["codex"],
        "models_by_agent": {"codex": "gpt-b"},
    }
    run_plan = compile_scenario(raw, base_dir=tmp_path).run_plan

    assert run_plan["run_count"] == 1
    assert run_plan["entries"][0]["agent_model"] == "gpt-b"
    assert run_plan["entries"][0]["model_source"] == "comparison.models_by_agent"


def test_execute_run_plan_writes_canonical_records_and_scenario_summary(tmp_path, monkeypatch):
    from skills.harness.common import write_json
    from skills.harness.host import runner
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    raw["path_budget"] = 400
    compilation = compile_scenario(raw, base_dir=tmp_path)
    result_root = tmp_path / "results"

    def fake_run_case(config, *, logs=(), prefix=None):
        records_dir = config.result_dir / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            config.result_dir / "run_summary.json",
            {
                "mode": config.mode,
                "agent": config.agent,
                "agent_model": config.agent_model,
                "skills_enabled": config.use_preinstalled_skills,
                "agent_elapsed_seconds": 2 if config.use_preinstalled_skills else 3,
                "token_count": 10,
                "agent_process_passed": True,
                "final_container_exit_code": 0,
                "source_input_immutable_policy": {"status": "pass"},
                "validation_metric_status": "not_required",
                "required_validation_metric_status": "not_required",
            },
        )
        write_json(config.result_dir / "container_exit_code.json", {"exit_code": 0})
        write_json(
            config.result_dir / "runtime_image.json",
            {
                "runtime_image": "runtime-image",
                "sdk_image_kind": "skills" if config.use_preinstalled_skills else "baseline",
            },
        )
        write_json(config.result_dir / "agent_activity.json", {"command_count": 2})
        write_json(records_dir / f"{config.mode}_agent_record.json", {"mode": config.mode})
        write_json(
            records_dir / f"{config.mode}_record.json",
            {
                "mode": config.mode,
                "agent": config.agent,
                "agent_model": config.agent_model,
                "agent_process_passed": True,
                "final_container_exit_code": 0,
                "source_input_immutable_policy": {"status": "pass"},
                "process_metrics": {
                    "agent_elapsed_seconds": 2 if config.use_preinstalled_skills else 3,
                    "elapsed_seconds": 2 if config.use_preinstalled_skills else 3,
                    "token_count": 10,
                    "agent_exit_code": 0,
                },
            },
        )
        return 0

    monkeypatch.setattr(runner, "run_case_safely", fake_run_case)
    monkeypatch.setattr(runner, "inspect_docker_image", lambda image: (True, ""))

    statuses, summary = runner.execute_run_plan(compilation, result_root=result_root)

    first_record_dir = result_root / compilation.run_plan["entries"][0]["record_dir"]
    assert statuses == {"run_00001": 0, "run_00002": 0}
    assert (result_root / "scenario.json").is_file()
    assert (result_root / "run_plan.json").is_file()
    assert json.loads((result_root / "docker_image_preflight.json").read_text(encoding="utf-8"))["status"] == "pass"
    assert (first_record_dir / "record_summary.json").is_file()
    assert (first_record_dir / "benchmark_record.json").is_file()
    assert not (result_root / "without_skills").exists()
    assert summary["status"] == "passed"
    assert "aggregate_metrics" in summary
    assert summary["aggregate_results"]["winner"]["label"] == "with_skills"
    first_record_summary = json.loads((first_record_dir / "record_summary.json").read_text(encoding="utf-8"))
    assert first_record_summary["scenario_name"] == raw["name"]
    assert first_record_summary["runtime_image"] == "runtime-image"
    assert first_record_summary["wheel_variant"] == "baseline"
    assert first_record_summary["command_count"] == 2
    assert "validation_metric_status" in first_record_summary
    assert (result_root / "reports" / "scenario_report.md").is_file()


def test_execute_run_plan_fails_preflight_before_missing_docker_image(tmp_path, monkeypatch):
    from skills.harness.host import runner
    from skills.harness.scenarios import ScenarioValidationError, compile_scenario

    raw = base_scenario(tmp_path)
    raw["comparison"] = {"type": "one", "mode": "with_skills"}
    raw["path_budget"] = 400
    compilation = compile_scenario(raw, base_dir=tmp_path)
    result_root = tmp_path / "results"
    called = {"run_case": False}

    monkeypatch.setattr(runner, "docker_context_name", lambda: "test-context")
    monkeypatch.setattr(
        runner,
        "docker_benchmark_image_list",
        lambda: ["agent-skills-benchmark:claude-skills abc123 1 minute ago 1GB"],
    )
    monkeypatch.setattr(runner, "inspect_docker_image", lambda image: (False, "image not found"))

    def fake_run_case(*args, **kwargs):
        called["run_case"] = True
        return 0

    monkeypatch.setattr(runner, "run_case_safely", fake_run_case)

    try:
        runner.execute_run_plan(compilation, result_root=result_root)
    except ScenarioValidationError as exc:
        message = str(exc)
        assert "Benchmark Docker image(s) are missing locally or could not be inspected" in message
        assert "Docker context: test-context" in message
        assert "Selected benchmark agent(s): codex" in message
        assert "image not found" in message
        assert "agent-skills-benchmark:claude-skills" in message
    else:
        raise AssertionError("missing Docker images should fail before any benchmark run starts")

    assert called["run_case"] is False
    preflight = json.loads((result_root / "docker_image_preflight.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "fail"
    assert preflight["docker_context"] == "test-context"
    assert preflight["local_benchmark_images"] == ["agent-skills-benchmark:claude-skills abc123 1 minute ago 1GB"]
    assert preflight["missing_images"] == ["agent-skills-benchmark:codex-skills"]
    assert preflight["images"]["agent-skills-benchmark:codex-skills"]["agents"] == ["codex"]
    summary = json.loads((result_root / "scenario_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["harness_failure"]["failure_category"] == "harness_preflight_failure"
    assert (result_root / "reports" / "scenario_report.json").is_file()


def test_scenario_summary_failed_when_all_runs_fail_quality_gate(tmp_path):
    from skills.harness.common import write_json
    from skills.harness.scenarios import compile_scenario, write_scenario_summaries

    raw = base_scenario(tmp_path)
    raw["path_budget"] = 400
    compilation = compile_scenario(raw, base_dir=tmp_path).write(tmp_path / "results")
    result_root = tmp_path / "results"
    statuses = {}
    for entry in compilation.run_plan["entries"]:
        record_dir = result_root / entry["record_dir"]
        record_dir.mkdir(parents=True)
        statuses[entry["run_id"]] = 1
        write_json(record_dir / "container_exit_code.json", {"exit_code": 1})
        write_json(record_dir / "record_summary.json", {"agent_process_passed": False, "final_container_exit_code": 1})

    summary = write_scenario_summaries(result_root, statuses)

    assert summary["status"] == "failed"
    assert summary["failed_run_count"] == 2


def test_scenario_summary_degraded_when_fail_fast_leaves_runs_unexecuted(tmp_path):
    from skills.harness.common import write_json
    from skills.harness.scenarios import compile_scenario, write_scenario_summaries

    raw = base_scenario(tmp_path)
    raw["fail_fast"] = True
    raw["path_budget"] = 400
    compilation = compile_scenario(raw, base_dir=tmp_path).write(tmp_path / "results")
    result_root = tmp_path / "results"
    first_entry = compilation.run_plan["entries"][0]
    record_dir = result_root / first_entry["record_dir"]
    record_dir.mkdir(parents=True)
    write_json(record_dir / "container_exit_code.json", {"exit_code": 1})
    write_json(record_dir / "record_summary.json", {"agent_process_passed": False, "final_container_exit_code": 1})

    summary = write_scenario_summaries(result_root, {first_entry["run_id"]: 1})

    assert summary["status"] == "degraded"
    assert summary["completed_run_count"] == 1
    assert summary["failed_run_count"] == 2


def test_scenario_summary_failed_when_report_generation_fails(tmp_path, monkeypatch):
    from skills.harness import scenarios
    from skills.harness.common import write_json

    raw = base_scenario(tmp_path)
    raw["comparison"] = {"type": "one", "mode": "with_skills"}
    compilation = scenarios.compile_scenario(raw, base_dir=tmp_path).write(tmp_path / "results")
    result_root = tmp_path / "results"
    entry = compilation.run_plan["entries"][0]
    record_dir = result_root / entry["record_dir"]
    record_dir.mkdir(parents=True)
    write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
    write_json(
        record_dir / "record_summary.json",
        {
            "agent_process_passed": True,
            "final_container_exit_code": 0,
            "source_input_immutable_policy": {"status": "pass"},
            "required_validation_metric_status": "not_required",
        },
    )
    monkeypatch.setattr(
        scenarios,
        "write_scenario_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    summary = scenarios.write_scenario_summaries(result_root, {entry["run_id"]: 0})

    assert summary["status"] == "failed"
    assert summary["report_generation_status"]["status"] == "failed"
    assert "render failed" in summary["report_generation_status"]["message"]


def test_scenario_summary_keeps_report_generation_pending_in_memory_only(tmp_path, monkeypatch):
    from skills.harness import scenarios
    from skills.harness.common import write_json

    raw = base_scenario(tmp_path)
    raw["comparison"] = {"type": "one", "mode": "with_skills"}
    compilation = scenarios.compile_scenario(raw, base_dir=tmp_path).write(tmp_path / "results")
    result_root = tmp_path / "results"
    entry = compilation.run_plan["entries"][0]
    record_dir = result_root / entry["record_dir"]
    record_dir.mkdir(parents=True)
    write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
    write_json(
        record_dir / "record_summary.json",
        {
            "agent_process_passed": True,
            "final_container_exit_code": 0,
            "source_input_immutable_policy": {"status": "pass"},
            "required_validation_metric_status": "not_required",
        },
    )

    def assert_pending_report_status(root, _summary):
        assert _summary["report_generation_status"]["status"] == "pending"
        assert not (root / "scenario_summary.json").exists()

    monkeypatch.setattr(scenarios, "write_scenario_report", assert_pending_report_status)

    summary = scenarios.write_scenario_summaries(result_root, {entry["run_id"]: 0})

    assert summary["report_generation_status"]["status"] == "ok"
    saved = json.loads((result_root / "scenario_summary.json").read_text(encoding="utf-8"))
    assert saved["report_generation_status"]["status"] == "ok"


def test_scenario_summary_replaces_stale_summary_atomically(tmp_path, monkeypatch):
    from skills.harness import scenarios
    from skills.harness.common import write_json

    raw = base_scenario(tmp_path)
    raw["comparison"] = {"type": "one", "mode": "with_skills"}
    compilation = scenarios.compile_scenario(raw, base_dir=tmp_path).write(tmp_path / "results")
    result_root = tmp_path / "results"
    (result_root / "scenario_summary.json").write_text('{"status": "stale"}', encoding="utf-8")
    entry = compilation.run_plan["entries"][0]
    record_dir = result_root / entry["record_dir"]
    record_dir.mkdir(parents=True)
    write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
    write_json(
        record_dir / "record_summary.json",
        {
            "agent_process_passed": True,
            "final_container_exit_code": 0,
            "source_input_immutable_policy": {"status": "pass"},
            "required_validation_metric_status": "not_required",
        },
    )

    summary = scenarios.write_scenario_summaries(result_root, {entry["run_id"]: 0})

    saved = json.loads((result_root / "scenario_summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == summary["status"]
    assert saved["status"] != "stale"
    assert saved["report_generation_status"]["status"] == "ok"
    assert not (result_root / ".scenario_summary.json.tmp").exists()


def test_scenario_summary_quality_gate_uses_container_exit_fallback(tmp_path):
    from skills.harness import scenario_summaries, scenarios
    from skills.harness.common import write_json

    raw = base_scenario(tmp_path)
    raw["comparison"] = {"type": "one", "mode": "with_skills"}
    compilation = scenarios.compile_scenario(raw, base_dir=tmp_path).write(tmp_path / "results")
    result_root = tmp_path / "results"
    entry = compilation.run_plan["entries"][0]
    record_dir = result_root / entry["record_dir"]
    record_dir.mkdir(parents=True)
    write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
    write_json(
        record_dir / "record_summary.json",
        {
            "agent_process_passed": True,
            "source_input_immutable_policy": {"status": "pass"},
            "required_validation_metric_status": "not_required",
        },
    )

    summary = scenario_summaries.write_scenario_summaries(
        result_root,
        {entry["run_id"]: 0},
        report_writer=lambda *_args, **_kwargs: None,
    )

    run = summary["runs"][0]
    assert run["final_container_exit_code"] == 0
    assert "final_container_exit_code=not_recorded" not in run["quality_gate_failures"]
    assert run["status"] == "passed"


def test_scenario_summary_records_per_entry_summary_exception(tmp_path, monkeypatch):
    from skills.harness import scenario_summaries, scenarios

    raw = base_scenario(tmp_path)
    raw["comparison"] = {"type": "one", "mode": "with_skills"}
    raw["quality_gate"] = {"required_validation_metric_status": ["present"]}
    compilation = scenarios.compile_scenario(raw, base_dir=tmp_path).write(tmp_path / "results")
    result_root = tmp_path / "results"
    entry = compilation.run_plan["entries"][0]

    def fail_run_summary(*_args, **_kwargs):
        raise RuntimeError("bad record payload")

    monkeypatch.setattr(scenario_summaries, "run_summary_for_entry", fail_run_summary)

    summary = scenario_summaries.write_scenario_summaries(
        result_root,
        {entry["run_id"]: 1},
        report_writer=lambda *_args, **_kwargs: None,
    )

    saved = json.loads((result_root / "scenario_summary.json").read_text(encoding="utf-8"))
    run = summary["runs"][0]
    assert saved["runs"][0]["status"] == "failed"
    assert run["quality_gate_failures"] == ["run_summary_generation_failed"]
    assert run["quality_gate"]["final_container_exit_code"] == 0
    assert run["quality_gate"]["required_validation_metric_status"] == ["present"]
    assert run["summary_generation_error"]["error_type"] == "RuntimeError"
    assert "bad record payload" in run["summary_generation_error"]["message"]


def test_scenario_summary_ignores_missing_run_id_when_indexing_runs(tmp_path, monkeypatch):
    from skills.harness import scenario_summaries
    from skills.harness.common import write_json

    result_root = tmp_path / "results"
    result_root.mkdir()
    run_plan = {
        "schema_version": "1",
        "scenario_name": "missing run id",
        "comparison_type": "mode_ablation",
        "quality_gate": {"final_container_exit_code": 0},
        "entries": [
            {"mode": "without_skills", "record_dir": "records/missing"},
            {"run_id": "run_00001", "mode": "with_skills", "record_dir": "records/valid"},
        ],
        "comparison_groups": [
            {
                "comparison_group_id": "group_001",
                "comparison_type": "mode_ablation",
                "compared_run_ids": ["None", "run_00001"],
            }
        ],
    }
    write_json(result_root / "run_plan.json", run_plan)
    write_json(result_root / "scenario.json", {"name": "missing run id"})

    def fake_run_summary(_root, entry, _statuses, _quality_gate):
        return {
            "run_id": entry.get("run_id"),
            "mode": entry["mode"],
            "record_dir": entry["record_dir"],
            "quality_gate_passed": entry.get("run_id") == "run_00001",
            "agent_elapsed_seconds": 1.0 if entry.get("run_id") == "run_00001" else None,
        }

    monkeypatch.setattr(scenario_summaries, "run_summary_for_entry", fake_run_summary)

    summary = scenario_summaries.write_scenario_summaries(
        result_root,
        {"run_00001": 0},
        report_writer=lambda *_args, **_kwargs: None,
    )

    compared = summary["comparison_groups"][0]["compared_runs"]
    assert [run["run_id"] for run in compared] == ["run_00001"]


def test_write_json_atomic_delegates_to_common_helper(tmp_path, monkeypatch):
    from skills.harness import scenario_summaries

    target = tmp_path / "scenario_summary.json"
    calls = []

    def fake_write_json_atomic(path, value):
        calls.append((path, value))

    monkeypatch.setattr(scenario_summaries, "common_write_json_atomic", fake_write_json_atomic)

    scenario_summaries.write_json_atomic(target, {"status": "ok"})

    assert calls == [(target, {"status": "ok"})]


def test_comparison_group_summary_ignores_non_numeric_token_count():
    from skills.harness.scenarios import comparison_group_summary

    group = {
        "comparison_group_id": "group_001",
        "comparison_type": "mode_ablation",
        "compared_run_ids": ["run_00001", "run_00002"],
    }
    runs_by_id = {
        "run_00001": {
            "run_id": "run_00001",
            "mode": "without_skills",
            "quality_gate_passed": True,
            "agent_elapsed_seconds": 2,
            "token_count": {"bad": "shape"},
        },
        "run_00002": {
            "run_id": "run_00002",
            "mode": "with_skills",
            "quality_gate_passed": True,
            "agent_elapsed_seconds": 3,
            "token_count": 1,
        },
    }

    summary = comparison_group_summary(group, runs_by_id)

    assert summary["winner"]["run_id"] == "run_00001"


def test_comparison_group_summary_sorts_missing_token_count_last():
    from skills.harness.scenarios import comparison_group_summary

    group = {
        "comparison_group_id": "group_001",
        "comparison_type": "mode_ablation",
        "compared_run_ids": ["run_00001", "run_00002"],
    }
    runs_by_id = {
        "run_00001": {
            "run_id": "run_00001",
            "mode": "without_skills",
            "quality_gate_passed": True,
            "agent_elapsed_seconds": 2,
            "token_count": None,
        },
        "run_00002": {
            "run_id": "run_00002",
            "mode": "with_skills",
            "quality_gate_passed": True,
            "agent_elapsed_seconds": 2,
            "token_count": 5,
        },
    }

    summary = comparison_group_summary(group, runs_by_id)

    assert summary["winner"]["run_id"] == "run_00002"


def test_comparison_group_summary_uses_stable_run_id_tiebreaker():
    from skills.harness.scenarios import comparison_group_summary

    group = {
        "comparison_group_id": "group_001",
        "comparison_type": "mode_ablation",
        "compared_run_ids": ["run_00002", "run_00001"],
    }
    runs_by_id = {
        "run_00001": {
            "run_id": "run_00001",
            "mode": "without_skills",
            "quality_gate_passed": True,
            "agent_elapsed_seconds": 2,
            "token_count": None,
        },
        "run_00002": {
            "run_id": "run_00002",
            "mode": "with_skills",
            "quality_gate_passed": True,
            "agent_elapsed_seconds": 2,
            "token_count": None,
        },
    }

    summary = comparison_group_summary(group, runs_by_id)

    assert summary["winner"]["run_id"] == "run_00001"


def test_aggregate_results_sorts_missing_token_median_last():
    from skills.harness.scenario_summaries import aggregate_results

    summary = aggregate_results(
        [
            {
                "mode": "without_skills",
                "quality_gate_passed": True,
                "agent_elapsed_seconds": 2,
                "token_count": None,
            },
            {
                "mode": "with_skills",
                "quality_gate_passed": True,
                "agent_elapsed_seconds": 2,
                "token_count": 5,
            },
        ]
    )

    assert summary["winner"]["label"] == "with_skills"


def test_replay_result_root_regenerates_agent_parser_artifacts(tmp_path):
    from skills.harness.agents.registry import load_agent_adapter
    from skills.harness.common import write_json
    from skills.harness.host.runner import replay_result_root
    from skills.harness.scenarios import compile_scenario

    raw = base_scenario(tmp_path)
    raw["agents"] = [{"name": "claude", "models": ["claude-test"]}]
    raw["comparison"] = {"type": "one", "mode": "with_skills"}
    compilation = compile_scenario(raw, base_dir=tmp_path)
    result_root = tmp_path / "captured"
    compilation.write(result_root)
    entry = compilation.run_plan["entries"][0]
    record_dir = result_root / entry["record_dir"]
    record_dir.mkdir(parents=True)
    adapter = load_agent_adapter("claude")
    raw_events = [
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "python job.py"}}],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "done",
            "usage": {"input_tokens": 4, "output_tokens": 5},
            "total_cost_usd": 0.01,
        },
    ]
    with (record_dir / "agent_events.jsonl").open("w", encoding="utf-8") as stream:
        for event in raw_events:
            stream.write(json.dumps(adapter.normalize_event(json.dumps(event))) + "\n")
    write_json(record_dir / "run_summary.json", {"agent_process_passed": True, "final_container_exit_code": 0})
    write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
    records_dir = record_dir / "records"
    records_dir.mkdir()
    write_json(records_dir / "with_skills_record.json", {"agent_process_passed": True, "final_container_exit_code": 0})

    summary = replay_result_root(result_root)

    usage = json.loads((record_dir / "agent_usage.json").read_text(encoding="utf-8"))
    activity = json.loads((record_dir / "agent_activity.json").read_text(encoding="utf-8"))
    replay_metadata = json.loads((result_root / "replay_metadata.json").read_text(encoding="utf-8"))
    assert usage["total_tokens"] == 9
    assert usage["cost"] == 0.01
    assert activity["commands"] == ["python job.py"]
    assert replay_metadata["replayed"] is True
    assert replay_metadata["agent_invocation"] == "replayed"
    assert summary["is_replay"] is True
    assert summary["agent_invocation"] == "replayed"
    assert summary["replay"]["source_result_root"] == str(result_root.resolve())
    assert (record_dir / "record_summary.json").is_file()
    assert summary["completed_run_count"] == 1
    report_json = json.loads((result_root / "reports" / "scenario_report.json").read_text(encoding="utf-8"))
    report_markdown = (result_root / "reports" / "scenario_report.md").read_text(encoding="utf-8")
    assert report_json["agent_invocation"] == "replayed"
    assert "Replay: `true`" in report_markdown
    assert "Agent invocation: `replayed`" in report_markdown
