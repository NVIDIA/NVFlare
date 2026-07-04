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
"""Scenario public API for benchmark scenario compilation and summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .reports.scenario_report import write_scenario_report
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
    SLUG_VISIBLE_LENGTH,
    SUMMARY_RUN_FIELDS,
    UNAVAILABLE_STRUCTURE_QUALITY_SIGNAL,
    ScenarioValidationError,
    as_list,
    load_yaml_file,
    model_list,
    require_mapping,
    require_non_empty_string,
    resolve_path,
    slug_base,
    slugify,
    stable_hash,
    unique_slug_map,
    utc_timestamp,
)
from .scenario_prompt import (
    GENERATED_PROMPT_DIR,
    MAX_PROMPT_BYTES,
    PROMPT_TEMPLATE_LITERAL_BRACE_HINT,
    PROMPT_TEMPLATE_SCALAR_TYPES,
    PROMPT_TEMPLATE_VARIABLE_PATTERN,
    materialize_prompt_for_output,
    prompt_template_fields,
    prompt_template_variables,
    read_prompt_bytes,
    render_prompt_template,
    rendered_prompt_filename,
    resolve_prompt,
    resolve_prompt_path,
)
from .scenario_run_plan import (
    AgentSpec,
    JobSpec,
    ModelSpec,
    ScenarioCompilation,
    WorkflowSpec,
    agent_model_options,
    append_group,
    artifact_paths,
    build_run_entry,
    compile_scenario,
    compile_scenario_file,
    expand_run_plan,
    model_slug_for,
    model_slug_key,
    record_dir_for,
    resolve_agent_comparison_models,
    resolve_agents,
    resolve_fail_fast,
    resolve_jobs,
    resolve_path_budget,
    resolve_quality_gate,
    resolve_workflows,
    scenario_reproducibility_metadata,
    slug_context,
    validate_comparison,
    validate_mode,
    validate_path_budget,
)
from .scenario_summaries import (
    agent_record_path,
    aggregate_results,
    benchmark_record_path,
    comparison_group_summary,
    comparison_label,
    is_number,
    number_or_none,
    quality_gate_failures,
    read_entry_artifacts,
    record_dir_path,
    required_validation_metric_status_from_artifacts,
    run_summary_for_entry,
    scenario_status,
    stats_for_values,
    write_json_atomic,
)
from .scenario_summaries import write_scenario_summaries as _write_scenario_summaries

__all__ = (
    "AgentSpec",
    "COMPARISON_AGENT",
    "COMPARISON_MODE_ABLATION",
    "COMPARISON_MODEL",
    "COMPARISON_ONE",
    "COMPARISON_TYPES",
    "DEFAULT_PATH_BUDGET",
    "DEFAULT_QUALITY_GATE",
    "DEFAULT_RESOURCE_POLICIES",
    "DEFAULT_WINNER_POLICY",
    "GENERATED_PROMPT_DIR",
    "JOB_SCALES",
    "JobSpec",
    "MAX_PROMPT_BYTES",
    "ModelSpec",
    "PROMPT_TEMPLATE_LITERAL_BRACE_HINT",
    "PROMPT_TEMPLATE_SCALAR_TYPES",
    "PROMPT_TEMPLATE_VARIABLE_PATTERN",
    "QUALITY_GATE_KEYS",
    "REQUIRED_VALIDATION_METRIC_STATUSES",
    "SCHEMA_VERSION",
    "SLUG_VISIBLE_LENGTH",
    "SUMMARY_RUN_FIELDS",
    "ScenarioCompilation",
    "ScenarioValidationError",
    "UNAVAILABLE_STRUCTURE_QUALITY_SIGNAL",
    "WorkflowSpec",
    "agent_model_options",
    "agent_record_path",
    "aggregate_results",
    "append_group",
    "artifact_paths",
    "as_list",
    "benchmark_record_path",
    "build_run_entry",
    "compile_scenario",
    "compile_scenario_file",
    "comparison_group_summary",
    "comparison_label",
    "expand_run_plan",
    "is_number",
    "load_yaml_file",
    "materialize_prompt_for_output",
    "model_list",
    "model_slug_for",
    "model_slug_key",
    "number_or_none",
    "prompt_template_fields",
    "prompt_template_variables",
    "quality_gate_failures",
    "read_entry_artifacts",
    "read_prompt_bytes",
    "record_dir_for",
    "record_dir_path",
    "render_prompt_template",
    "rendered_prompt_filename",
    "require_mapping",
    "require_non_empty_string",
    "required_validation_metric_status_from_artifacts",
    "resolve_agent_comparison_models",
    "resolve_agents",
    "resolve_fail_fast",
    "resolve_jobs",
    "resolve_path",
    "resolve_path_budget",
    "resolve_prompt",
    "resolve_prompt_path",
    "resolve_quality_gate",
    "resolve_workflows",
    "run_summary_for_entry",
    "scenario_reproducibility_metadata",
    "scenario_status",
    "slug_base",
    "slug_context",
    "slugify",
    "stable_hash",
    "stats_for_values",
    "unique_slug_map",
    "utc_timestamp",
    "validate_comparison",
    "validate_mode",
    "validate_path_budget",
    "write_json_atomic",
    "write_scenario_report",
    "write_scenario_summaries",
)


def write_scenario_summaries(
    result_root: str | Path,
    statuses: Mapping[str, int] | None = None,
    *,
    harness_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _write_scenario_summaries(
        result_root,
        statuses,
        harness_failure=harness_failure,
        report_writer=write_scenario_report,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compile an agent benchmark scenario into scenario.json/run_plan.json."
    )
    parser.add_argument("scenario", help="Scenario YAML file")
    parser.add_argument(
        "--output-dir", required=True, help="Directory where scenario.json and run_plan.json are written"
    )
    args = parser.parse_args(argv)
    compilation = compile_scenario_file(args.scenario)
    compilation.write(args.output_dir)


if __name__ == "__main__":
    main()
