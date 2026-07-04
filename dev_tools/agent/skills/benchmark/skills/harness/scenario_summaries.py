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

"""Scenario summary aggregation and report status helpers."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping

from .common import load_json
from .common import write_json_atomic as common_write_json_atomic
from .quality_signals import critical_quality_checks_failed, required_validation_metric_status
from .reports.scenario_report import write_scenario_report
from .safe_paths import path_beneath
from .scenario_common import (
    COMPARISON_AGENT,
    COMPARISON_MODE_ABLATION,
    COMPARISON_MODEL,
    DEFAULT_QUALITY_GATE,
    DEFAULT_WINNER_POLICY,
    SCHEMA_VERSION,
    SUMMARY_RUN_FIELDS,
    UNAVAILABLE_STRUCTURE_QUALITY_SIGNAL,
)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def number_or_none(value: Any) -> float | None:
    return float(value) if is_number(value) else None


def optional_number_sort_value(value: Any) -> tuple[int, float]:
    return (0, float(value)) if is_number(value) else (1, 0.0)


def stats_for_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "mean": None, "min": None, "max": None, "stddev": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.stdev(values) if len(values) >= 2 else None,
    }


def record_dir_path(result_root: Path, entry: Mapping[str, Any]) -> Path:
    return path_beneath(
        result_root,
        entry.get("record_dir"),
        label="run_plan entry record_dir",
        required_prefix="records",
    )


def benchmark_record_path(record_dir: Path, mode: str) -> Path:
    direct = record_dir / "benchmark_record.json"
    if direct.is_file():
        return direct
    return record_dir / "records" / f"{mode}_record.json"


def agent_record_path(record_dir: Path, mode: str) -> Path:
    direct = record_dir / "agent_record.json"
    if direct.is_file():
        return direct
    return record_dir / "records" / f"{mode}_agent_record.json"


def read_entry_artifacts(result_root: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    record_dir = record_dir_path(result_root, entry)
    mode = str(entry["mode"])
    summary = load_json(record_dir / "record_summary.json", None)
    if not isinstance(summary, dict):
        summary = load_json(record_dir / "run_summary.json", {}) or {}
    record = load_json(benchmark_record_path(record_dir, mode), {}) or {}
    container_exit = load_json(record_dir / "container_exit_code.json", {}) or {}
    return {
        "record_dir": record_dir,
        "summary": summary if isinstance(summary, dict) else {},
        "record": record if isinstance(record, dict) else {},
        "container_exit": container_exit if isinstance(container_exit, dict) else {},
    }


def quality_gate_failures(
    summary: Mapping[str, Any],
    record: Mapping[str, Any],
    status: int | None,
    quality_gate: Mapping[str, Any] | None = None,
    *,
    final_container_exit_code: Any = None,
) -> list[str]:
    gate = quality_gate or DEFAULT_QUALITY_GATE
    failures = []
    if status not in (0, None):
        failures.append(f"host_status={status}")
    agent_passed = summary.get("agent_process_passed", record.get("agent_process_passed"))
    expected_agent_passed = gate.get("agent_process_passed", True)
    if agent_passed != expected_agent_passed:
        failures.append(
            "agent_process_passed" if expected_agent_passed is True else f"agent_process_passed={agent_passed}"
        )
    final_exit = final_container_exit_code
    if final_exit is None:
        final_exit = summary.get("final_container_exit_code", record.get("final_container_exit_code"))
    expected_final_exit = gate.get("final_container_exit_code", 0)
    if final_exit is None:
        failures.append("final_container_exit_code=not_recorded")
    elif final_exit != expected_final_exit:
        failures.append(f"final_container_exit_code={final_exit}")
    policy = summary.get("source_input_immutable_policy")
    if not isinstance(policy, dict):
        policy = (
            record.get("source_input_immutable_policy")
            if isinstance(record.get("source_input_immutable_policy"), dict)
            else {}
        )
    source_input_modified = policy.get("status") == "fail"
    expected_source_input_modified = gate.get("source_input_modified", False)
    if source_input_modified != expected_source_input_modified:
        failures.append(
            "source_input_modified"
            if expected_source_input_modified is False
            else f"source_input_modified={source_input_modified}"
        )
    metric_status = (
        record.get("required_validation_metric_status")
        or summary.get("required_validation_metric_status")
        or required_validation_metric_status_from_artifacts(summary, record)
    )
    allowed_metric_statuses = set(gate.get("required_validation_metric_status") or ())
    if metric_status and metric_status not in allowed_metric_statuses:
        failures.append(f"required_validation_metric_status={metric_status}")
    critical_checks_failed = bool(
        record.get("critical_quality_checks_failed")
        or summary.get("critical_quality_checks_failed")
        or critical_quality_checks_failed(summary, record)
    )
    expected_critical_failed = gate.get("critical_quality_checks_failed", False)
    if critical_checks_failed != expected_critical_failed:
        failures.append(
            "critical_quality_checks_failed"
            if expected_critical_failed is False
            else f"critical_quality_checks_failed={critical_checks_failed}"
        )
    for field in ("eval_contract_status", "expected_skill_status", "required_behavior_status"):
        value = record.get(field) or summary.get(field) or "not_scored"
        allowed = set(gate.get(field) or ())
        if value not in allowed:
            failures.append(f"{field}={value}")
    if not summary and not record:
        failures.append("missing_record_summary")
    return failures


def required_validation_metric_status_from_artifacts(summary: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    for artifact in (record, summary):
        quality = artifact.get("quality_signals") if isinstance(artifact, Mapping) else None
        if isinstance(quality, Mapping):
            signal = quality.get("job_guidance_primary_validation_metric") or quality.get(
                "readme_primary_validation_metric"
            )
            status = required_validation_metric_status(signal if isinstance(signal, Mapping) else None)
            if status != "not_required":
                return status
    return "not_required"


def run_summary_for_entry(
    result_root: Path,
    entry: Mapping[str, Any],
    statuses: Mapping[str, int],
    quality_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = read_entry_artifacts(result_root, entry)
    summary = artifacts["summary"]
    record = artifacts["record"]
    container_exit = artifacts["container_exit"]
    status = statuses.get(str(entry["run_id"]))
    if status is None:
        status = statuses.get(str(entry["mode"]))
    final_exit = summary.get("final_container_exit_code", record.get("final_container_exit_code"))
    if final_exit is None:
        final_exit = container_exit.get("exit_code")
    effective_quality_gate = quality_gate or DEFAULT_QUALITY_GATE
    failures = quality_gate_failures(
        summary,
        record,
        status,
        effective_quality_gate,
        final_container_exit_code=final_exit,
    )
    required_metric_status = (
        record.get("required_validation_metric_status")
        or summary.get("required_validation_metric_status")
        or required_validation_metric_status_from_artifacts(summary, record)
    )
    critical_checks_failed = bool(
        record.get("critical_quality_checks_failed")
        or summary.get("critical_quality_checks_failed")
        or critical_quality_checks_failed(summary, record)
    )
    eval_contract_status = record.get("eval_contract_status") or summary.get("eval_contract_status") or "not_scored"
    expected_skill_status = record.get("expected_skill_status") or summary.get("expected_skill_status") or "not_scored"
    required_behavior_status = (
        record.get("required_behavior_status") or summary.get("required_behavior_status") or "not_scored"
    )
    process_metrics = record.get("process_metrics") if isinstance(record.get("process_metrics"), dict) else {}
    agent_elapsed = summary.get("agent_elapsed_seconds")
    if agent_elapsed is None:
        agent_elapsed = process_metrics.get("agent_elapsed_seconds")
    token_count = summary.get("token_count")
    if token_count is None:
        token_count = process_metrics.get("token_count")
    payload = {key: entry.get(key) for key in SUMMARY_RUN_FIELDS}
    payload.update(
        {
            "status": "passed" if not failures else "failed",
            "host_status": status,
            "quality_gate_passed": not failures,
            "quality_gate_failures": failures,
            "quality_gate": dict(effective_quality_gate),
            "required_validation_metric_status": required_metric_status,
            "validation_metric_status": required_metric_status,
            "critical_quality_checks_failed": critical_checks_failed,
            "eval_contract_status": eval_contract_status,
            "expected_skill_status": expected_skill_status,
            "required_behavior_status": required_behavior_status,
            "instruction_compliance": record.get("instruction_compliance")
            or summary.get("instruction_compliance")
            or {},
            "mandatory_behavior": record.get("mandatory_behavior") or summary.get("mandatory_behavior") or {},
            "prohibited_behavior": record.get("prohibited_behavior") or summary.get("prohibited_behavior") or {},
            "failure_analysis": record.get("failure_analysis") or summary.get("failure_analysis") or {},
            "agent_elapsed_seconds": agent_elapsed,
            "elapsed_seconds": (
                agent_elapsed
                if agent_elapsed is not None
                else summary.get("elapsed_seconds", process_metrics.get("elapsed_seconds"))
            ),
            "phase_seconds": summary.get("phase_seconds", process_metrics.get("phase_seconds")),
            "token_count": token_count,
            "command_count": summary.get("command_count", process_metrics.get("command_count")),
            "cost": summary.get("cost", process_metrics.get("cost")),
            "agent_process_passed": summary.get("agent_process_passed", record.get("agent_process_passed")),
            "agent_exit_code": summary.get("agent_exit_code", process_metrics.get("agent_exit_code")),
            "final_container_exit_code": final_exit,
            "failure_root_cause": record.get("failure_root_cause") or record.get("failure_category"),
            "observed_skill_name": summary.get("observed_skill_name")
            or record.get("skill")
            or record.get("skill_name"),
            "skill_name_source": summary.get("skill_name_source") or record.get("agent_record_source"),
            "validation_metric": summary.get("validation_metric")
            or record.get("validation_metric")
            or record.get("reported_validation_metric"),
            "structure_quality_signal": summary.get("structure_quality_signal")
            or record.get("structure_quality_signal")
            or UNAVAILABLE_STRUCTURE_QUALITY_SIGNAL,
            "artifact_paths": entry.get("artifact_paths") or {},
        }
    )
    return payload


def failed_run_summary_for_entry(
    entry: Mapping[str, Any],
    statuses: Mapping[str, int],
    exc: Exception,
    *,
    quality_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    effective_quality_gate = quality_gate or DEFAULT_QUALITY_GATE
    status = statuses.get(str(entry.get("run_id")))
    if status is None:
        status = statuses.get(str(entry.get("mode")))
    payload = {key: entry.get(key) for key in SUMMARY_RUN_FIELDS}
    payload.update(
        {
            "status": "failed",
            "host_status": status,
            "quality_gate_passed": False,
            "quality_gate_failures": ["run_summary_generation_failed"],
            "quality_gate": dict(effective_quality_gate),
            "required_validation_metric_status": "unknown",
            "validation_metric_status": "unknown",
            "critical_quality_checks_failed": True,
            "agent_elapsed_seconds": None,
            "elapsed_seconds": None,
            "phase_seconds": None,
            "token_count": None,
            "command_count": None,
            "cost": None,
            "agent_process_passed": False,
            "agent_exit_code": None,
            "final_container_exit_code": None,
            "failure_root_cause": "run_summary_generation_failed",
            "observed_skill_name": None,
            "skill_name_source": None,
            "validation_metric": None,
            "structure_quality_signal": UNAVAILABLE_STRUCTURE_QUALITY_SIGNAL,
            "artifact_paths": entry.get("artifact_paths") or {},
            "summary_generation_error": {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        }
    )
    return payload


def comparison_label(entry: Mapping[str, Any]) -> str:
    comparison_type = str(entry.get("comparison_type") or "")
    if comparison_type == COMPARISON_MODE_ABLATION:
        return str(entry.get("mode"))
    if comparison_type == COMPARISON_AGENT:
        return str(entry.get("agent"))
    if comparison_type == COMPARISON_MODEL:
        return str(entry.get("agent_model"))
    return str(entry.get("mode") or entry.get("run_id"))


def comparison_group_summary(
    group: Mapping[str, Any],
    runs_by_id: Mapping[str, dict[str, Any]],
    quality_gate: Mapping[str, Any] | None = None,
    winner_policy: str = DEFAULT_WINNER_POLICY,
) -> dict[str, Any]:
    effective_quality_gate = quality_gate or DEFAULT_QUALITY_GATE
    compared = [runs_by_id[run_id] for run_id in group.get("compared_run_ids", []) if run_id in runs_by_id]
    winner = None
    candidates = [
        run for run in compared if run.get("quality_gate_passed") and is_number(run.get("agent_elapsed_seconds"))
    ]
    if candidates:
        winner_run = min(
            candidates,
            key=lambda item: (
                float(item["agent_elapsed_seconds"]),
                optional_number_sort_value(item.get("token_count")),
                str(item.get("run_id") or ""),
            ),
        )
        winner = {
            "run_id": winner_run["run_id"],
            "label": comparison_label(winner_run),
            "agent_elapsed_seconds": winner_run.get("agent_elapsed_seconds"),
            "token_count": winner_run.get("token_count"),
        }
    return {
        "comparison_group_id": group.get("comparison_group_id"),
        "comparison_type": group.get("comparison_type"),
        "group_axes": group.get("group_axes") or {},
        "compared_runs": compared,
        "compared_records": compared,
        "aggregate_results": aggregate_results(compared, winner_policy=winner_policy),
        "winner_policy": winner_policy,
        "quality_gate": dict(effective_quality_gate),
        "winner": winner,
        "status": "passed" if compared and all(run.get("quality_gate_passed") for run in compared) else "degraded",
    }


def aggregate_results(runs: list[dict[str, Any]], winner_policy: str = DEFAULT_WINNER_POLICY) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_label.setdefault(comparison_label(run), []).append(run)
    aggregate = {}
    for label, items in sorted(by_label.items()):
        elapsed_values = [
            float(item["agent_elapsed_seconds"]) for item in items if is_number(item.get("agent_elapsed_seconds"))
        ]
        token_values = [float(item["token_count"]) for item in items if is_number(item.get("token_count"))]
        aggregate[label] = {
            "run_count": len(items),
            "quality_pass_count": sum(1 for item in items if item.get("quality_gate_passed")),
            "quality_fail_count": sum(1 for item in items if not item.get("quality_gate_passed")),
            "agent_elapsed_seconds": stats_for_values(elapsed_values),
            "token_count": stats_for_values(token_values),
        }
    candidates = [
        (label, data)
        for label, data in aggregate.items()
        if data["quality_pass_count"] > 0 and data["agent_elapsed_seconds"]["median"] is not None
    ]
    winner = None
    if candidates:
        label, data = min(
            candidates,
            key=lambda item: (
                float(item[1]["agent_elapsed_seconds"]["median"]),
                optional_number_sort_value(item[1]["token_count"]["median"]),
            ),
        )
        winner = {
            "label": label,
            "median_agent_elapsed_seconds": data["agent_elapsed_seconds"]["median"],
            "median_token_count": data["token_count"]["median"],
        }
    effective_policy = winner_policy if winner else "no_quality_qualified_winner"
    return {"by_label": aggregate, "winner": winner, "winner_policy": effective_policy}


def write_json_atomic(path: Path, value: object) -> None:
    common_write_json_atomic(path, value)


def scenario_status(
    *,
    runs: list[dict[str, Any]],
    completed: int,
    failed: int,
    harness_failure: Mapping[str, Any] | None = None,
    report_generation_failed: bool = False,
) -> str:
    if harness_failure or report_generation_failed:
        return "failed"
    if not runs:
        return "not_run"
    if completed == 0:
        return "not_run"
    if failed == 0:
        return "passed"
    if completed == len(runs) and failed == len(runs):
        return "failed"
    return "degraded"


def write_scenario_summaries(
    result_root: str | Path,
    statuses: Mapping[str, int] | None = None,
    *,
    harness_failure: Mapping[str, Any] | None = None,
    report_writer: Any = None,
) -> dict[str, Any]:
    root = Path(result_root)
    if report_writer is None:
        report_writer = write_scenario_report
    statuses = statuses or {}
    run_plan = load_json(root / "run_plan.json", {}) or {}
    scenario = load_json(root / "scenario.json", {}) or {}
    replay_metadata = load_json(root / "replay_metadata.json", {}) or {}
    if not isinstance(replay_metadata, dict):
        replay_metadata = {}
    entries = run_plan.get("entries") if isinstance(run_plan.get("entries"), list) else []
    quality_gate = (
        run_plan.get("quality_gate") if isinstance(run_plan.get("quality_gate"), dict) else DEFAULT_QUALITY_GATE
    )
    winner_policy = str(run_plan.get("winner_policy") or DEFAULT_WINNER_POLICY)
    runs = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            runs.append(run_summary_for_entry(root, entry, statuses, quality_gate))
        except Exception as exc:
            runs.append(failed_run_summary_for_entry(entry, statuses, exc, quality_gate=quality_gate))
    runs_by_id = {str(run["run_id"]): run for run in runs if run.get("run_id") is not None}
    comparison_groups = [
        comparison_group_summary(group, runs_by_id, quality_gate, winner_policy)
        for group in run_plan.get("comparison_groups", [])
        if isinstance(group, dict)
    ]
    aggregate = aggregate_results(runs, winner_policy=winner_policy)
    completed = sum(
        1 for run in runs if run.get("host_status") is not None or run.get("final_container_exit_code") is not None
    )
    failed = sum(1 for run in runs if not run.get("quality_gate_passed"))
    status = scenario_status(runs=runs, completed=completed, failed=failed, harness_failure=harness_failure)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scenario_name": run_plan.get("scenario_name") or scenario.get("name"),
        "comparison_type": run_plan.get("comparison_type"),
        "expanded_case_count": len(entries),
        "completed_run_count": completed,
        "failed_run_count": failed,
        "status": status,
        "quality_gate": quality_gate,
        "winner_policy": winner_policy,
        "reproducibility": scenario.get("reproducibility") if isinstance(scenario, dict) else {},
        "replay": replay_metadata,
        "is_replay": bool(replay_metadata.get("replayed")),
        "agent_invocation": "replayed" if replay_metadata.get("replayed") else "live",
        "runs": runs,
        "comparison_groups": comparison_groups,
        "aggregate_results": aggregate,
        "aggregate_metrics": aggregate,
    }
    if harness_failure:
        summary["harness_failure"] = dict(harness_failure)
        summary["failure_root_cause"] = harness_failure.get("failure_category") or harness_failure.get("message")
    summary["report_generation_status"] = {"status": "pending"}
    try:
        report_writer(root, summary)
    except Exception as exc:
        summary["status"] = scenario_status(
            runs=runs,
            completed=completed,
            failed=failed,
            harness_failure=harness_failure,
            report_generation_failed=True,
        )
        summary["report_generation_status"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        write_json_atomic(root / "scenario_summary.json", summary)
    else:
        summary["report_generation_status"] = {"status": "ok"}
        write_json_atomic(root / "scenario_summary.json", summary)
    return summary
