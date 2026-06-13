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

"""Scenario validation and run-plan expansion."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .agents.registry import load_agent_adapter
from .common import write_json
from .modes import mode_spec
from .scenario_common import (
    COMPARISON_AGENT,
    COMPARISON_MODE_ABLATION,
    COMPARISON_MODEL,
    COMPARISON_ONE,
    COMPARISON_TYPES,
    DEFAULT_PATH_BUDGET,
    DEFAULT_QUALITY_GATE,
    DEFAULT_RESOURCE_POLICIES,
    DEFAULT_WINNER_POLICY,
    JOB_SCALES,
    QUALITY_GATE_KEYS,
    REQUIRED_VALIDATION_METRIC_STATUSES,
    SCHEMA_VERSION,
    ScenarioValidationError,
    as_list,
    load_yaml_file,
    model_list,
    require_mapping,
    require_non_empty_string,
    resolve_path,
    slugify,
    unique_slug_map,
    utc_timestamp,
)
from .scenario_prompt import materialize_prompt_for_output, resolve_prompt


@dataclass(frozen=True)
class ScenarioCompilation:
    scenario: dict[str, Any]
    run_plan: dict[str, Any]

    def materialized(self, output_dir: str | Path) -> "ScenarioCompilation":
        scenario = copy.deepcopy(self.scenario)
        run_plan = copy.deepcopy(self.run_plan)
        materialize_prompt_for_output(scenario, run_plan, Path(output_dir))
        return ScenarioCompilation(scenario=scenario, run_plan=run_plan)

    def write(self, output_dir: str | Path) -> "ScenarioCompilation":
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        materialized = self.materialized(path)
        write_json(path / "scenario.json", materialized.scenario)
        write_json(path / "run_plan.json", materialized.run_plan)
        return materialized


@dataclass(frozen=True)
class AgentSpec:
    name: str
    models: tuple[str, ...]
    default_model: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str


@dataclass(frozen=True)
class WorkflowSpec:
    name: str


@dataclass(frozen=True)
class JobSpec:
    path: Path
    name: str
    scale: str
    resource_policy: dict[str, int]


def resolve_agents(raw: Mapping[str, Any]) -> tuple[dict[str, AgentSpec], list[dict[str, Any]]]:
    agents_raw = as_list(raw.get("agents"), "agents")
    if not agents_raw:
        raise ScenarioValidationError("agents must contain at least one agent")
    agents: dict[str, AgentSpec] = {}
    resolved = []
    for index, item in enumerate(agents_raw):
        if isinstance(item, str):
            name = item
            models = ()
        else:
            data = require_mapping(item, f"agents[{index}]")
            name = require_non_empty_string(data.get("name"), f"agents[{index}].name")
            models = model_list(data.get("models", data.get("model")), f"agents[{index}].models")
        if name in agents:
            raise ScenarioValidationError(f"Duplicate agent entry: {name}")
        try:
            adapter = load_agent_adapter(name)
        except ValueError as exc:
            raise ScenarioValidationError(str(exc)) from exc
        try:
            default_model = adapter.default_model if models else adapter.model_from_env({})
        except ValueError as exc:
            raise ScenarioValidationError(str(exc)) from exc
        spec = AgentSpec(name=name, models=models, default_model=default_model)
        agents[name] = spec
        resolved.append(
            {
                "name": name,
                "models": list(models),
                "default_model": default_model,
                "model_source_when_unspecified": "adapter_default",
            }
        )
    return agents, resolved


def resolve_workflows(raw: Mapping[str, Any]) -> tuple[list[WorkflowSpec], list[dict[str, Any]]]:
    workflows_raw = as_list(raw.get("workflows"), "workflows")
    if not workflows_raw:
        raise ScenarioValidationError("workflows must contain at least one workflow")
    workflows = []
    for index, item in enumerate(workflows_raw):
        if isinstance(item, str):
            name = item
        else:
            data = require_mapping(item, f"workflows[{index}]")
            name = data.get("name")
        workflows.append(WorkflowSpec(require_non_empty_string(name, f"workflows[{index}].name")))
    if len({item.name for item in workflows}) != len(workflows):
        raise ScenarioValidationError("workflows contains duplicate names")
    return workflows, [{"name": item.name} for item in workflows]


def _integer_policy_overrides(raw: Mapping[str, Any], field_path: str) -> dict[str, int]:
    overrides = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ScenarioValidationError(f"{field_path}.{key} must be an integer greater than 0; got {value!r}")
        overrides[str(key)] = value
    return overrides


def resource_policy_for(
    scale: str, scenario_raw: Mapping[str, Any], job_raw: Mapping[str, Any], job_policy_path: str
) -> dict[str, int]:
    policy = dict(DEFAULT_RESOURCE_POLICIES[scale])
    scenario_policy = scenario_raw.get("resource_policy") or {}
    if isinstance(scenario_policy, dict):
        global_overrides = scenario_policy.get(scale) or {}
        if isinstance(global_overrides, dict):
            policy.update(_integer_policy_overrides(global_overrides, f"resource_policy.{scale}"))
    job_policy = job_raw.get("resource_policy") or {}
    if isinstance(job_policy, dict):
        policy.update(_integer_policy_overrides(job_policy, job_policy_path))
    return policy


def resolve_jobs(raw: Mapping[str, Any], base_dir: Path) -> tuple[list[JobSpec], list[dict[str, Any]]]:
    jobs_raw = as_list(raw.get("jobs"), "jobs")
    if not jobs_raw:
        raise ScenarioValidationError("jobs must contain at least one job")
    jobs = []
    resolved = []
    for index, item in enumerate(jobs_raw):
        data = require_mapping(item, f"jobs[{index}]")
        path_value = require_non_empty_string(data.get("path"), f"jobs[{index}].path")
        path = resolve_path(path_value, base_dir).resolve()
        if not path.is_dir():
            raise ScenarioValidationError(f"Job path must be an existing directory: {path}")
        scale = data.get("scale", data.get("job_scale"))
        scale = require_non_empty_string(scale, f"jobs[{index}].scale")
        if scale not in JOB_SCALES:
            raise ScenarioValidationError(f"jobs[{index}].scale must be one of: {', '.join(sorted(JOB_SCALES))}")
        name = require_non_empty_string(data.get("name") or path.name, f"jobs[{index}].name")
        policy = resource_policy_for(scale, raw, data, f"jobs[{index}].resource_policy")
        jobs.append(JobSpec(path=path, name=name, scale=scale, resource_policy=policy))
        resolved.append({"name": name, "path": str(path), "scale": scale, "resource_policy": policy})
    return jobs, resolved


def resolve_fail_fast(raw: Mapping[str, Any]) -> bool:
    value = raw.get("fail_fast", False)
    if not isinstance(value, bool):
        raise ScenarioValidationError(f"fail_fast must be a boolean; got {value!r}")
    return value


def resolve_quality_gate(raw: Mapping[str, Any]) -> dict[str, Any]:
    override = raw.get("quality_gate")
    gate = {
        "agent_process_passed": DEFAULT_QUALITY_GATE["agent_process_passed"],
        "final_container_exit_code": DEFAULT_QUALITY_GATE["final_container_exit_code"],
        "source_input_modified": DEFAULT_QUALITY_GATE["source_input_modified"],
        "required_validation_metric_status": list(DEFAULT_QUALITY_GATE["required_validation_metric_status"]),
        "critical_quality_checks_failed": DEFAULT_QUALITY_GATE["critical_quality_checks_failed"],
    }
    if override is None:
        return gate
    if not isinstance(override, dict):
        raise ScenarioValidationError("quality_gate must be a mapping")
    for key, value in override.items():
        if key not in QUALITY_GATE_KEYS:
            raise ScenarioValidationError(
                f"quality_gate.{key} is not supported; expected one of: {', '.join(sorted(QUALITY_GATE_KEYS))}"
            )
        if key in {"agent_process_passed", "source_input_modified", "critical_quality_checks_failed"}:
            if not isinstance(value, bool):
                raise ScenarioValidationError(f"quality_gate.{key} must be a boolean")
            gate[key] = value
        elif key == "final_container_exit_code":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ScenarioValidationError("quality_gate.final_container_exit_code must be an integer")
            gate[key] = value
        elif key == "required_validation_metric_status":
            statuses = model_list(value, "quality_gate.required_validation_metric_status")
            if not statuses:
                raise ScenarioValidationError("quality_gate.required_validation_metric_status must not be empty")
            unknown = [status for status in statuses if status not in REQUIRED_VALIDATION_METRIC_STATUSES]
            if unknown:
                raise ScenarioValidationError(
                    "quality_gate.required_validation_metric_status contains unsupported value(s): "
                    + ", ".join(unknown)
                )
            gate[key] = list(statuses)
    return gate


def resolve_path_budget(raw: Mapping[str, Any]) -> int:
    value = raw.get("path_budget", DEFAULT_PATH_BUDGET)
    if isinstance(value, bool) or not isinstance(value, int) or value < 80:
        raise ScenarioValidationError("path_budget must be an integer >= 80")
    return value


def validate_mode(mode: Any, label: str) -> str:
    mode_name = require_non_empty_string(mode, label)
    try:
        mode_spec(mode_name)
    except ValueError as exc:
        raise ScenarioValidationError(str(exc)) from exc
    return mode_name


def validate_comparison(raw: Mapping[str, Any], agents: Mapping[str, AgentSpec]) -> dict[str, Any]:
    comparison = require_mapping(raw.get("comparison"), "comparison")
    comparison_type = require_non_empty_string(comparison.get("type"), "comparison.type")
    if comparison_type not in COMPARISON_TYPES:
        raise ScenarioValidationError(
            f"comparison.type must be one of: {', '.join(sorted(COMPARISON_TYPES))}; got {comparison_type}"
        )
    resolved = dict(comparison)
    if comparison_type == COMPARISON_MODE_ABLATION:
        modes = [
            validate_mode(item, "comparison.modes[]") for item in as_list(comparison.get("modes"), "comparison.modes")
        ]
        if not modes:
            raise ScenarioValidationError("comparison.modes must contain at least one mode")
        if len(set(modes)) != len(modes):
            raise ScenarioValidationError("comparison.modes contains duplicate modes")
        resolved["modes"] = modes
    elif comparison_type == COMPARISON_ONE:
        resolved["mode"] = validate_mode(comparison.get("mode"), "comparison.mode")
    elif comparison_type == COMPARISON_AGENT:
        resolved["mode"] = validate_mode(comparison.get("mode"), "comparison.mode")
        compared_agents = [
            item if isinstance(item, str) else require_mapping(item, "comparison.agents[]").get("name")
            for item in as_list(comparison.get("agents"), "comparison.agents")
        ]
        compared_agents = [require_non_empty_string(item, "comparison.agents[]") for item in compared_agents]
        if not compared_agents:
            raise ScenarioValidationError("comparison.agents must contain at least one agent")
        if len(set(compared_agents)) != len(compared_agents):
            raise ScenarioValidationError("comparison.agents contains duplicate agents")
        for agent in compared_agents:
            if agent not in agents:
                raise ScenarioValidationError(f"comparison.agents includes {agent!r}, but agents does not define it")
        resolved["agents"] = compared_agents
    elif comparison_type == COMPARISON_MODEL:
        agent = require_non_empty_string(comparison.get("agent"), "comparison.agent")
        if agent not in agents:
            raise ScenarioValidationError(f"comparison.agent {agent!r} is not listed in agents")
        resolved["agent"] = agent
        resolved["mode"] = validate_mode(comparison.get("mode"), "comparison.mode")
        models = model_list(comparison.get("models"), "comparison.models")
        if not models:
            raise ScenarioValidationError("comparison.models must contain at least one model")
        resolved["models"] = list(models)
    return resolved


def agent_model_options(agent: AgentSpec) -> list[ModelSpec]:
    if agent.models:
        return [ModelSpec(name=model, source="scenario") for model in agent.models]
    return [ModelSpec(name=agent.default_model, source="adapter_default")]


def resolve_agent_comparison_models(
    comparison: Mapping[str, Any], agents: Mapping[str, AgentSpec]
) -> dict[str, ModelSpec]:
    explicit = comparison.get("models_by_agent") or {}
    if explicit and not isinstance(explicit, dict):
        raise ScenarioValidationError("comparison.models_by_agent must be a mapping")
    resolved = {}
    for agent_name in comparison["agents"]:
        if agent_name in explicit:
            models = model_list(explicit[agent_name], f"comparison.models_by_agent.{agent_name}")
            if len(models) != 1:
                raise ScenarioValidationError(
                    f"comparison.models_by_agent.{agent_name} must resolve to exactly one model"
                )
            resolved[agent_name] = ModelSpec(models[0], "comparison.models_by_agent")
            continue
        agent = agents[agent_name]
        if len(agent.models) > 1:
            raise ScenarioValidationError(
                f"agent_comparison model selection is ambiguous for {agent_name}; "
                "use comparison.models_by_agent or configure exactly one top-level model"
            )
        if len(agent.models) == 1:
            resolved[agent_name] = ModelSpec(agent.models[0], "scenario")
        else:
            resolved[agent_name] = ModelSpec(agent.default_model, "adapter_default")
    return resolved


def artifact_paths(record_dir: str) -> dict[str, str]:
    return {
        "record_dir": record_dir,
        "record_summary": f"{record_dir}/record_summary.json",
        "agent_events": f"{record_dir}/agent_events.jsonl",
        "agent_usage": f"{record_dir}/agent_usage.json",
        "agent_activity": f"{record_dir}/agent_activity.json",
        "agent_last_message": f"{record_dir}/agent_last_message.txt",
        "agent_stderr": f"{record_dir}/agent_stderr.txt",
        "agent_record": f"{record_dir}/agent_record.json",
        "benchmark_record": f"{record_dir}/benchmark_record.json",
        "input_delta_manifest": f"{record_dir}/input_delta_manifest.json",
        "workspace_delta_manifest": f"{record_dir}/workspace_delta_manifest.json",
    }


def record_dir_for(
    *,
    agent_slug: str,
    model_slug: str,
    workflow_slug: str,
    job_slug: str,
    mode: str,
) -> str:
    return f"records/agent={agent_slug}/model={model_slug}/workflow={workflow_slug}/job={job_slug}/mode={mode}"


def model_slug_for(slugs: Mapping[str, Mapping[str, str]], agent_name: str, model_name: str) -> str:
    return slugs["models"].get(model_slug_key(agent_name, model_name)) or slugify(model_name)


def model_slug_key(agent_name: str, model_name: str) -> str:
    return json.dumps([agent_name, model_name], ensure_ascii=False, separators=(",", ":"))


def build_run_entry(
    *,
    scenario_name: str,
    comparison_type: str,
    comparison_group_id: str,
    sequence: int,
    agent: AgentSpec,
    model: ModelSpec,
    workflow: WorkflowSpec,
    job: JobSpec,
    mode: str,
    prompt: Mapping[str, Any],
    slugs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    spec = mode_spec(mode)
    agent_slug = slugs["agents"][agent.name]
    model_slug = model_slug_for(slugs, agent.name, model.name)
    workflow_slug = slugs["workflows"][workflow.name]
    job_slug = slugs["jobs"][job.name]
    record_dir = record_dir_for(
        agent_slug=agent_slug,
        model_slug=model_slug,
        workflow_slug=workflow_slug,
        job_slug=job_slug,
        mode=mode,
    )
    return {
        "run_id": f"run_{sequence:05d}",
        "sequence": sequence,
        "scenario_name": scenario_name,
        "comparison_type": comparison_type,
        "comparison_group_id": comparison_group_id,
        "agent": agent.name,
        "agent_slug": agent_slug,
        "agent_model": model.name,
        "agent_model_slug": model_slug,
        "model_source": model.source,
        "workflow": workflow.name,
        "workflow_slug": workflow_slug,
        "job_name": job.name,
        "job_slug": job_slug,
        "job_path": str(job.path),
        "job_scale": job.scale,
        "resource_policy": job.resource_policy,
        "mode": mode,
        "mode_label": spec.label,
        "skills_enabled": spec.skills_enabled,
        "prompt_source": prompt["path"],
        "prompt_hash": prompt["sha256"],
        "prompt_bytes": prompt["bytes"],
        "record_dir": record_dir,
        "artifact_paths": artifact_paths(record_dir),
    }


def slug_context(
    agents: Mapping[str, AgentSpec],
    workflows: list[WorkflowSpec],
    jobs: list[JobSpec],
    comparison: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    model_values_by_agent: dict[str, list[str]] = {}

    def append_model(agent_name: str, model_name: str) -> None:
        values = model_values_by_agent.setdefault(agent_name, [])
        if model_name not in values:
            values.append(model_name)

    for agent in agents.values():
        for model in agent.models or (agent.default_model,):
            append_model(agent.name, model)
    if comparison.get("type") == COMPARISON_MODEL:
        agent_name = str(comparison["agent"])
        for model in comparison["models"]:
            append_model(agent_name, str(model))
    if comparison.get("type") == COMPARISON_AGENT and isinstance(comparison.get("models_by_agent"), dict):
        for agent_name, value in comparison["models_by_agent"].items():
            for model in model_list(value, f"comparison.models_by_agent.{agent_name}"):
                append_model(str(agent_name), model)
    model_slugs = {}
    for agent_name, values in model_values_by_agent.items():
        per_agent = unique_slug_map(values)
        for value, slug in per_agent.items():
            model_slugs[model_slug_key(agent_name, value)] = slug
    return {
        "agents": unique_slug_map(agent.name for agent in agents.values()),
        "models": model_slugs,
        "workflows": unique_slug_map(workflow.name for workflow in workflows),
        "jobs": unique_slug_map(job.name for job in jobs),
    }


def append_group(
    *,
    groups: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    group_index: int,
    group_axes: dict[str, Any],
    compared_entries: list[dict[str, Any]],
    comparison_type: str,
) -> None:
    group_id = f"group_{group_index:05d}"
    for entry in compared_entries:
        entry["comparison_group_id"] = group_id
        entries.append(entry)
    groups.append(
        {
            "comparison_group_id": group_id,
            "comparison_type": comparison_type,
            "group_axes": group_axes,
            "compared_run_ids": [entry["run_id"] for entry in compared_entries],
        }
    )


def expand_run_plan(
    *,
    scenario_name: str,
    comparison: Mapping[str, Any],
    agents: Mapping[str, AgentSpec],
    workflows: list[WorkflowSpec],
    jobs: list[JobSpec],
    prompt: Mapping[str, Any],
    path_budget: int,
    quality_gate: Mapping[str, Any],
    winner_policy: str = DEFAULT_WINNER_POLICY,
) -> dict[str, Any]:
    comparison_type = str(comparison["type"])
    slugs = slug_context(agents, workflows, jobs, comparison)
    entries: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    sequence = 0
    group_index = 0

    def next_entry(**kwargs: Any) -> dict[str, Any]:
        nonlocal sequence
        sequence += 1
        return build_run_entry(
            scenario_name=scenario_name,
            comparison_type=comparison_type,
            sequence=sequence,
            prompt=prompt,
            slugs=slugs,
            **kwargs,
        )

    if comparison_type in {COMPARISON_MODE_ABLATION, COMPARISON_ONE}:
        modes = comparison.get("modes") if comparison_type == COMPARISON_MODE_ABLATION else [comparison["mode"]]
        for agent in agents.values():
            for model in agent_model_options(agent):
                for workflow in workflows:
                    for job in jobs:
                        group_index += 1
                        compared = [
                            next_entry(
                                comparison_group_id="",
                                agent=agent,
                                model=model,
                                workflow=workflow,
                                job=job,
                                mode=mode,
                            )
                            for mode in modes
                        ]
                        append_group(
                            groups=groups,
                            entries=entries,
                            group_index=group_index,
                            group_axes={
                                "agent": agent.name,
                                "agent_model": model.name,
                                "workflow": workflow.name,
                                "job_slug": slugs["jobs"][job.name],
                            },
                            compared_entries=compared,
                            comparison_type=comparison_type,
                        )
    elif comparison_type == COMPARISON_AGENT:
        model_by_agent = resolve_agent_comparison_models(comparison, agents)
        mode = str(comparison["mode"])
        for workflow in workflows:
            for job in jobs:
                group_index += 1
                compared = [
                    next_entry(
                        comparison_group_id="",
                        agent=agents[agent_name],
                        model=model_by_agent[agent_name],
                        workflow=workflow,
                        job=job,
                        mode=mode,
                    )
                    for agent_name in comparison["agents"]
                ]
                append_group(
                    groups=groups,
                    entries=entries,
                    group_index=group_index,
                    group_axes={
                        "mode": mode,
                        "workflow": workflow.name,
                        "job_slug": slugs["jobs"][job.name],
                    },
                    compared_entries=compared,
                    comparison_type=comparison_type,
                )
    elif comparison_type == COMPARISON_MODEL:
        agent = agents[str(comparison["agent"])]
        mode = str(comparison["mode"])
        models = [ModelSpec(str(model), "comparison") for model in comparison["models"]]
        for workflow in workflows:
            for job in jobs:
                group_index += 1
                compared = [
                    next_entry(
                        comparison_group_id="",
                        agent=agent,
                        model=model,
                        workflow=workflow,
                        job=job,
                        mode=mode,
                    )
                    for model in models
                ]
                append_group(
                    groups=groups,
                    entries=entries,
                    group_index=group_index,
                    group_axes={
                        "agent": agent.name,
                        "mode": mode,
                        "workflow": workflow.name,
                        "job_slug": slugs["jobs"][job.name],
                    },
                    compared_entries=compared,
                    comparison_type=comparison_type,
                )

    # Compile-time check uses a root-agnostic proxy; execution revalidates with the actual result root.
    validate_path_budget(scenario_name, entries, path_budget)
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_name": scenario_name,
        "generated_at": utc_timestamp(),
        "comparison_type": comparison_type,
        "run_count": len(entries),
        "comparison_group_count": len(groups),
        "execution": {"parallelism": 1},
        "quality_gate": dict(quality_gate),
        "winner_policy": winner_policy,
        "entries": entries,
        "comparison_groups": groups,
    }


def scenario_reproducibility_metadata(
    *,
    agents: Mapping[str, AgentSpec],
    prompt: Mapping[str, Any],
    comparison: Mapping[str, Any],
    jobs: list[JobSpec],
) -> dict[str, Any]:
    agent_metadata = {}
    image_targets = {}
    build_args = {}
    agent_versions = {}
    for agent_name in sorted(agents):
        adapter = load_agent_adapter(agent_name)
        targets = adapter.image_targets()
        args = adapter.build_args()
        agent_metadata[agent_name] = adapter.metadata()
        image_targets[agent_name] = {
            "skills": targets.skills,
            "baseline": targets.baseline,
            "report": targets.report,
        }
        build_args[agent_name] = args
        agent_versions[agent_name] = {
            key: value
            for key, value in sorted(args.items())
            if key in {"AGENT_CLI_NAME", "AGENT_INSTALL_COMMAND", "AGENT_VERSION_COMMAND"}
        }
    modes = []
    comparison_type = comparison.get("type")
    if comparison_type == COMPARISON_MODE_ABLATION:
        modes = list(comparison.get("modes") or [])
    elif comparison_type in {COMPARISON_AGENT, COMPARISON_MODEL, COMPARISON_ONE}:
        modes = [str(comparison.get("mode"))]
    wheel_variants = sorted(
        {
            (
                "local_wheel_with_preinstalled_skills"
                if mode_spec(str(mode)).skills_enabled
                else "local_wheel_without_packaged_skills"
            )
            for mode in modes
        }
    )
    return {
        "compiled_at": utc_timestamp(),
        "prompt_hash": prompt.get("sha256"),
        "prompt_bytes": prompt.get("bytes"),
        "agent_adapters": agent_metadata,
        "agent_versions": agent_versions,
        "image_targets": image_targets,
        "build_args": build_args,
        "wheel_variants": wheel_variants,
        "job_paths": {job.name: str(job.path) for job in jobs},
    }


def validate_path_budget(
    scenario_name: str, entries: list[dict[str, Any]], path_budget: int, result_root: str | Path | None = None
) -> None:
    prefix = Path(result_root) if result_root is not None else Path("results") / slugify(scenario_name)
    for entry in entries:
        artifact_values = list((entry.get("artifact_paths") or {}).values())
        longest = max((str(Path(path)) for path in artifact_values), key=len) if artifact_values else ""
        candidate = prefix / longest
        if len(str(candidate)) > path_budget:
            raise ScenarioValidationError(f"Expanded artifact path exceeds path_budget={path_budget}: {candidate}")


def compile_scenario(
    raw: Mapping[str, Any],
    *,
    base_dir: str | Path,
    source_path: str | Path | None = None,
    allow_external_prompt: bool = False,
) -> ScenarioCompilation:
    base_path = Path(base_dir)
    name = require_non_empty_string(raw.get("name"), "name")
    prompt = resolve_prompt(raw, base_path, allow_external_prompt=allow_external_prompt)
    agents, resolved_agents = resolve_agents(raw)
    workflows, resolved_workflows = resolve_workflows(raw)
    jobs, resolved_jobs = resolve_jobs(raw, base_path)
    path_budget = resolve_path_budget(raw)
    comparison = validate_comparison(raw, agents)
    fail_fast = resolve_fail_fast(raw)
    quality_gate = resolve_quality_gate(raw)

    scenario = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "scenario_slug": slugify(name),
        "source_path": str(Path(source_path).resolve()) if source_path else None,
        "prompt": prompt,
        "agents": resolved_agents,
        "workflows": resolved_workflows,
        "jobs": resolved_jobs,
        "comparison": comparison,
        "fail_fast": fail_fast,
        "quality_gate": quality_gate,
        "winner_policy": DEFAULT_WINNER_POLICY,
        "path_budget": path_budget,
        "resource_policy_defaults": DEFAULT_RESOURCE_POLICIES,
    }
    scenario["reproducibility"] = scenario_reproducibility_metadata(
        agents=agents,
        prompt=prompt,
        comparison=comparison,
        jobs=jobs,
    )
    run_plan = expand_run_plan(
        scenario_name=name,
        comparison=comparison,
        agents=agents,
        workflows=workflows,
        jobs=jobs,
        prompt=prompt,
        path_budget=path_budget,
        quality_gate=quality_gate,
        winner_policy=DEFAULT_WINNER_POLICY,
    )
    run_plan["source_path"] = scenario["source_path"]
    run_plan["fail_fast"] = fail_fast
    return ScenarioCompilation(scenario=scenario, run_plan=run_plan)


def compile_scenario_file(path: str | Path) -> ScenarioCompilation:
    scenario_path = Path(path)
    raw = load_yaml_file(scenario_path)
    return compile_scenario(raw, base_dir=scenario_path.parent, source_path=scenario_path)
