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

"""Host-side benchmark orchestration CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from ..agents.registry import DEFAULT_BENCHMARK_AGENT, load_agent_adapter
from ..common import load_json, write_json
from ..modes import PAIR_RUNS, mode_spec
from ..scenarios import (
    ScenarioCompilation,
    ScenarioValidationError,
    compile_scenario,
    compile_scenario_file,
    slugify,
    validate_path_budget,
    write_scenario_summaries,
)
from .common import (
    CONTAINER_PROMPT_PATH,
    CaseConfig,
    ImageConfig,
    absolute_path,
    add_agent_auth_mounts,
    add_agent_passthrough_env,
    default_results_root,
    docker_args_for_case,
    docker_env,
    emit,
    parse_host_cli_options,
    stream_command,
    timestamp_slug,
    write_runtime_image,
)

STALE_RESULT_FILES = (
    "comprehensive_report.json",
    "comprehensive_report.md",
    "metrics_plots.png",
    "metrics_plots.svg",
    "metrics_summary.json",
    "metrics_report.html",
    "metrics_report.json",
    "metrics_report.md",
    "benchmark_insights.md",
    "pair_summary.json",
    "process_eval_ablation_summary.json",
    "report_generator_status.json",
    "skill_benchmark.json",
    "skill_benchmark.md",
    "skill_performance.json",
    "skill_performance.txt",
    "skill_report_status.json",
    "scenario.json",
    "run_plan.json",
    "scenario_summary.json",
)
STALE_RESULT_DIRS = (
    "process_eval_runs",
    "records",
    "reports",
    "with_skills_eval_off",
    "with_skills_eval_on",
)


@dataclass(frozen=True)
class InteractiveRuntimeConfig:
    agent: str
    agent_model: str
    model_was_explicit: bool


@dataclass(frozen=True)
class ScenarioCliOptions:
    scenario_path: Path
    results_root: Path | None = None
    result_root: Path | None = None
    agent_home: Path | None = None
    mount_agent_auth: bool | None = None


@dataclass(frozen=True)
class ReplayCliOptions:
    result_root: Path


@dataclass(frozen=True)
class RuntimeAuthOptions:
    agent_home: Path | None = None
    mount_agent_auth: bool | None = None

    @property
    def has_overrides(self) -> bool:
        return self.agent_home is not None or self.mount_agent_auth is not None


def run_one_case(config: CaseConfig, *, logs: Iterable[Path] = (), prefix: str | None = None) -> int:
    config.result_dir.mkdir(parents=True, exist_ok=True)
    emit(f"Running mode={config.mode} with runtime image: {config.run_image}", logs=logs, prefix=prefix)
    emit(f"Report image: {config.images.report_image_name}", logs=logs, prefix=prefix)
    emit(f"Job folder: {config.job_input_dir} -> /workspace/input", logs=logs, prefix=prefix)
    emit(f"Prompt file: {config.prompt_path} -> {CONTAINER_PROMPT_PATH}", logs=logs, prefix=prefix)
    write_runtime_image(config)
    status = stream_command(
        docker_args_for_case(config, logs=logs, prefix=prefix),
        logs=logs,
        prefix=prefix,
        timeout_seconds=config.container_timeout_seconds,
    )
    write_json(config.result_dir / "container_exit_code.json", {"exit_code": status})
    if enforce_result_size_budget(config, logs=logs, prefix=prefix):
        return 1
    return combined_exit_status({config.mode: status})


def directory_size_bytes(path: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if not (Path(dirpath) / name).is_symlink()]
        for name in filenames:
            item = Path(dirpath) / name
            try:
                if item.is_symlink() or not item.is_file():
                    continue
                total += item.stat().st_size
            except OSError:
                continue
    return total


def enforce_result_size_budget(config: CaseConfig, *, logs: Iterable[Path] = (), prefix: str | None = None) -> bool:
    budget = config.result_size_budget_bytes
    if budget is None:
        return False
    used = directory_size_bytes(config.result_dir)
    failed = used > budget
    write_json(
        config.result_dir / "result_size_budget.json",
        {
            "status": "fail" if failed else "pass",
            "result_size_bytes": used,
            "budget_bytes": budget,
        },
    )
    if failed:
        emit(
            f"Result size budget exceeded: {used} bytes > {budget} bytes.",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
    return failed


def write_host_error(path: Path, exc: BaseException) -> None:
    write_json(
        path,
        {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    )


def run_case_safely(config: CaseConfig, *, logs: Iterable[Path] = (), prefix: str | None = None) -> int:
    try:
        return run_one_case(config, logs=logs, prefix=prefix)
    except Exception as exc:
        config.result_dir.mkdir(parents=True, exist_ok=True)
        write_host_error(config.result_dir / "host_case_error.json", exc)
        emit(
            f"Case failed before completion: {type(exc).__name__}: {exc}",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
        return 1


def run_one(argv: list[str]) -> int:
    options = parse_host_cli_options(argv, "run-one")
    try:
        adapter = agent_adapter_from_options(options)
        images = image_config_from_adapter(adapter)
        compilation = one_compilation_from_options(options)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc)
    mode = str(compilation.run_plan["entries"][0]["mode"])
    result_root = single_result_dir(options, mode)
    result_root.mkdir(parents=True, exist_ok=True)
    clean_pair_result_root(result_root)
    console_log = result_root / "console_output.log"
    console_log.write_text("", encoding="utf-8")
    logs = (console_log,)

    emit(f"Result root: {result_root}", logs=logs)
    emit(f"Console log: {console_log}", logs=logs)
    emit(f"Runtime image: {images.image_name}", logs=logs)
    emit(f"Report image: {images.report_image_name}", logs=logs)
    emit(f"Job folder: {options.job_input}", logs=logs)
    emit(f"Prompt file: {options.prompt_path} -> {CONTAINER_PROMPT_PATH}", logs=logs)

    try:
        run_statuses, scenario_summary = execute_run_plan(
            compilation,
            result_root=result_root,
            logs=logs,
            **execute_auth_kwargs(options),
        )
    except ScenarioValidationError as exc:
        status = emit_scenario_validation_error(exc, logs=logs)
        write_host_report_status(result_root)
        return status

    emit(f"Scenario summary: {result_root / 'scenario_summary.json'}", logs=logs)
    emit(f"Scenario report: {result_root / 'reports' / 'scenario_report.md'}", logs=logs)
    write_host_report_status(result_root)
    return (
        1
        if any(status != 0 for status in run_statuses.values())
        or scenario_summary.get("status") in {"degraded", "failed"}
        else 0
    )


def single_result_dir(options, mode: str) -> Path:
    if options.result_dir is not None:
        return options.result_dir
    if options.results_root is not None:
        return options.results_root / "single" / f"{timestamp_slug()}_{mode}"
    return absolute_path(
        os.environ.get("RESULT_DIR", str(default_results_root() / "single" / f"{timestamp_slug()}_{mode}"))
    )


def comparison_result_root(options, *, default_prefix: str | None = None) -> Path:
    if options.result_root is not None:
        return options.result_root
    timestamp = timestamp_slug()
    default_name = f"{default_prefix}_{timestamp}" if default_prefix else timestamp
    if options.results_root is not None:
        return options.results_root / default_name
    return absolute_path(os.environ.get("RESULT_ROOT", str(default_results_root() / default_name)))


def scenario_result_root(options: ScenarioCliOptions, compilation: ScenarioCompilation) -> Path:
    if options.result_root is not None:
        return options.result_root
    slug = compilation.scenario.get("scenario_slug") or slugify(str(compilation.scenario.get("name") or "scenario"))
    default_name = f"{slug}_{timestamp_slug()}"
    if options.results_root is not None:
        return options.results_root / default_name
    return absolute_path(os.environ.get("RESULT_ROOT", str(default_results_root() / default_name)))


def parse_scenario_cli_options(argv: list[str]) -> ScenarioCliOptions:
    scenario_path: Path | None = None
    results_root: Path | None = None
    result_root: Path | None = None
    agent_home: Path | None = None
    mount_agent_auth: bool | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--results-root" or arg.startswith("--results-root="):
            value, index = _scenario_option_value(argv, index, "--results-root")
            if results_root is not None:
                raise SystemExit("Expected only one --results-root")
            results_root = absolute_path(value)
        elif (
            arg in {"--output-dir", "--result-root"}
            or arg.startswith("--output-dir=")
            or arg.startswith("--result-root=")
        ):
            option = "--output-dir" if arg.startswith("--output-dir") else "--result-root"
            value, index = _scenario_option_value(argv, index, option)
            if result_root is not None:
                raise SystemExit("Expected only one exact output directory")
            result_root = absolute_path(value)
        elif arg == "--agent-home" or arg.startswith("--agent-home="):
            value, index = _scenario_option_value(argv, index, "--agent-home")
            if agent_home is not None:
                raise SystemExit("Expected only one --agent-home")
            agent_home = absolute_path(value)
        elif arg == "--no-agent-auth-mount":
            if mount_agent_auth is False:
                raise SystemExit("Expected only one --no-agent-auth-mount")
            mount_agent_auth = False
            index += 1
        elif arg in {"-h", "--help"}:
            print(
                "Usage: run.sh scenario SCENARIO.yaml "
                "[--results-root PATH|--output-dir PATH] [--agent-home PATH] [--no-agent-auth-mount]"
            )
            raise SystemExit(0)
        elif arg.startswith("-"):
            raise SystemExit(f"Unknown scenario option: {arg}")
        else:
            if scenario_path is not None:
                raise SystemExit("Expected only one scenario YAML file")
            scenario_path = absolute_path(arg)
            index += 1
    if scenario_path is None:
        raise SystemExit("Scenario YAML file is required.")
    if not scenario_path.is_file():
        raise SystemExit(f"Scenario YAML file must exist: {scenario_path}")
    if results_root is not None and result_root is not None:
        raise SystemExit("Use --results-root or --output-dir, not both.")
    return ScenarioCliOptions(
        scenario_path=scenario_path,
        results_root=results_root,
        result_root=result_root,
        agent_home=agent_home,
        mount_agent_auth=mount_agent_auth,
    )


def parse_replay_cli_options(argv: list[str]) -> ReplayCliOptions:
    result_root: Path | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if (
            arg in {"--result-root", "--output-dir"}
            or arg.startswith("--result-root=")
            or arg.startswith("--output-dir=")
        ):
            option = "--result-root" if arg.startswith("--result-root") else "--output-dir"
            value, index = _scenario_option_value(argv, index, option)
            if result_root is not None:
                raise SystemExit("Expected only one replay result root")
            result_root = absolute_path(value)
        elif arg in {"-h", "--help"}:
            print("Usage: run.sh replay RESULT_ROOT")
            raise SystemExit(0)
        elif arg.startswith("-"):
            raise SystemExit(f"Unknown replay option: {arg}")
        else:
            if result_root is not None:
                raise SystemExit("Expected only one replay result root")
            result_root = absolute_path(arg)
            index += 1
    if result_root is None:
        raise SystemExit("Replay result root is required.")
    if not (result_root / "run_plan.json").is_file():
        raise SystemExit(f"Replay result root must contain run_plan.json: {result_root}")
    return ReplayCliOptions(result_root=result_root)


def _scenario_option_value(argv: list[str], index: int, option: str) -> tuple[str, int]:
    arg = argv[index]
    if arg.startswith(f"{option}="):
        return arg.split("=", 1)[1], index + 1
    if index + 1 >= len(argv):
        raise SystemExit(f"{option} requires a path")
    return argv[index + 1], index + 2


def checked_bool_override(name: str, expected: bool, mode: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return expected
    if value not in {"true", "false"}:
        raise SystemExit(f"{name} must be true or false; got {value}")
    actual = value == "true"
    if actual != expected:
        expected_text = "true" if expected else "false"
        raise SystemExit(f"{name}={value} conflicts with MODE={mode}; expected {expected_text}.")
    return actual


def reject_parallel_comparison_runs(command: str) -> None:
    parallel = os.environ.get("PARALLEL_CASES", "false").strip().lower()
    if parallel not in {"", "0", "false", "no", "off"}:
        raise SystemExit(
            f"PARALLEL_CASES is no longer supported for benchmark comparisons; {command} runs sequentially."
        )


def clean_pair_result_root(result_root: Path) -> None:
    """Remove generated artifacts from older harness layouts before a fresh pair run."""

    for spec in PAIR_RUNS:
        path = result_root / spec.mode
        if path.exists():
            shutil.rmtree(path)
    for name in STALE_RESULT_DIRS:
        path = result_root / name
        if path.exists():
            shutil.rmtree(path)
    for name in STALE_RESULT_FILES:
        path = result_root / name
        if path.exists() and path.is_file():
            path.unlink()


def pair_compilation_from_options(options) -> ScenarioCompilation:
    adapter = agent_adapter_from_options(options)
    agent_model, model_was_explicit = agent_model_from_options(adapter, options)
    agent_entry: dict[str, object] = {"name": adapter.name}
    if model_was_explicit:
        agent_entry["models"] = [agent_model]
    raw = {
        "name": f"pair {adapter.name} {options.job_input.name}",
        "prompt": str(options.prompt_path),
        "agents": [agent_entry],
        "comparison": {"type": "mode_ablation", "modes": [spec.mode for spec in PAIR_RUNS]},
        "workflows": [{"name": options.workflow or os.environ.get("BENCHMARK_WORKFLOW", "default")}],
        "jobs": [
            {
                "name": options.job_input.name,
                "path": str(options.job_input),
                "scale": options.job_scale
                or os.environ.get("BENCHMARK_JOB_SCALE", os.environ.get("JOB_SCALE", "small")),
            }
        ],
    }
    return compile_scenario(raw, base_dir=Path.cwd(), allow_external_prompt=True)


def one_compilation_from_options(options) -> ScenarioCompilation:
    adapter = agent_adapter_from_options(options)
    mode = options.mode or os.environ.get("MODE", "with_skills")
    try:
        spec = mode_spec(mode)
    except ValueError as exc:
        raise ScenarioValidationError(str(exc).replace("Unknown mode", "Unknown MODE")) from exc
    checked_bool_override(
        "USE_PREINSTALLED_SKILLS",
        spec.skills_enabled,
        mode,
    )
    agent_model, model_was_explicit = agent_model_from_options(adapter, options)
    agent_entry: dict[str, object] = {"name": adapter.name}
    if model_was_explicit:
        agent_entry["models"] = [agent_model]
    raw = {
        "name": f"one {adapter.name} {mode} {options.job_input.name}",
        "prompt": str(options.prompt_path),
        "agents": [agent_entry],
        "comparison": {"type": "one", "mode": mode},
        "workflows": [{"name": options.workflow or os.environ.get("BENCHMARK_WORKFLOW", "default")}],
        "jobs": [
            {
                "name": options.job_input.name,
                "path": str(options.job_input),
                "scale": options.job_scale
                or os.environ.get("BENCHMARK_JOB_SCALE", os.environ.get("JOB_SCALE", "small")),
            }
        ],
    }
    return compile_scenario(raw, base_dir=Path.cwd(), allow_external_prompt=True)


def image_config_for_agent(agent: str) -> ImageConfig:
    return image_config_from_adapter(load_agent_adapter(agent))


def image_config_from_adapter(adapter) -> ImageConfig:
    try:
        return ImageConfig.for_adapter(adapter)
    except ValueError as exc:
        raise ScenarioValidationError(str(exc)) from exc


def model_was_explicit_for_entry(entry: dict[str, object]) -> bool:
    return str(entry.get("model_source") or "") != "adapter_default"


def positive_int_resource_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def case_config_for_entry(
    entry: dict[str, object],
    result_root: Path,
    runtime_auth_options: RuntimeAuthOptions | None = None,
) -> CaseConfig:
    adapter = load_agent_adapter(str(entry["agent"]))
    resource_policy = entry.get("resource_policy") if isinstance(entry.get("resource_policy"), dict) else {}
    runtime_auth_options = runtime_auth_options or RuntimeAuthOptions()
    host_agent_home = runtime_auth_options.agent_home or absolute_path(str(adapter.host_home_from_env(os.environ)))
    mount_host_agent_auth = (
        runtime_auth_options.mount_agent_auth
        if runtime_auth_options.mount_agent_auth is not None
        else adapter.mount_auth_from_env(os.environ)
    )
    return CaseConfig(
        mode=str(entry["mode"]),
        use_preinstalled_skills=bool(entry["skills_enabled"]),
        job_input_dir=Path(str(entry["job_path"])),
        result_dir=result_root / str(entry["record_dir"]),
        prompt_path=Path(str(entry["prompt_source"])),
        images=image_config_for_agent(adapter.name),
        progress_interval_seconds=os.environ.get("PROGRESS_INTERVAL_SECONDS", "60"),
        agent=adapter.name,
        agent_model=str(entry["agent_model"]),
        model_was_explicit=model_was_explicit_for_entry(entry),
        adapter=adapter,
        host_agent_home=host_agent_home,
        mount_host_agent_auth=mount_host_agent_auth,
        agent_timeout_seconds=positive_int_resource_value(resource_policy.get("agent_timeout_seconds")),
        container_timeout_seconds=positive_int_resource_value(resource_policy.get("container_timeout_seconds")),
        result_size_budget_bytes=positive_int_resource_value(resource_policy.get("result_size_budget_bytes")),
    )


def inspect_docker_image(image: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return result.returncode == 0, result.stderr.strip()


def preflight_docker_images(
    entries: Iterable[dict[str, object]],
    *,
    result_root: Path,
    logs: Iterable[Path] = (),
    runtime_auth_options: RuntimeAuthOptions | None = None,
) -> None:
    images_by_run: dict[str, list[str]] = {}
    inspect_results: dict[str, dict[str, object]] = {}
    for entry in entries:
        config = case_config_for_entry(entry, result_root, runtime_auth_options)
        images_by_run.setdefault(config.run_image, []).append(str(entry.get("run_id")))
    for image in sorted(images_by_run):
        available, detail = inspect_docker_image(image)
        inspect_results[image] = {
            "available": available,
            "detail": detail,
            "run_ids": images_by_run[image],
        }
    missing = [image for image, result in inspect_results.items() if not result["available"]]
    payload = {
        "status": "fail" if missing else "pass",
        "images": inspect_results,
        "missing_images": missing,
    }
    write_json(result_root / "docker_image_preflight.json", payload)
    if not missing:
        return
    message = (
        "Benchmark Docker image(s) are missing locally: "
        + ", ".join(missing)
        + ". Run ./bin/build.sh from assist_tools/skills_benchmark before running the benchmark."
    )
    emit(message, logs=logs, stderr=True)
    raise ScenarioValidationError(message)


def copy_file_if_present(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return True


def canonicalize_entry_artifacts(result_root: Path, entry: dict[str, object], status: int | None) -> None:
    record_dir = result_root / str(entry["record_dir"])
    mode = str(entry["mode"])
    record_dir.mkdir(parents=True, exist_ok=True)
    write_json(record_dir / "run_plan_entry.json", entry)
    copy_file_if_present(record_dir / "records" / f"{mode}_agent_record.json", record_dir / "agent_record.json")
    copy_file_if_present(record_dir / "records" / f"{mode}_record.json", record_dir / "benchmark_record.json")
    copy_file_if_present(record_dir / "run_summary.json", record_dir / "record_summary.json")
    summary = load_json(record_dir / "record_summary.json", {}) or {}
    if not isinstance(summary, dict):
        summary = {}
    runtime_image = load_json(record_dir / "runtime_image.json", {}) or {}
    if not isinstance(runtime_image, dict):
        runtime_image = {}
    activity = load_json(record_dir / "agent_activity.json", {}) or {}
    if not isinstance(activity, dict):
        activity = {}
    summary.update(
        {
            key: entry.get(key)
            for key in (
                "run_id",
                "scenario_name",
                "comparison_type",
                "comparison_group_id",
                "agent",
                "agent_model",
                "workflow",
                "job_name",
                "job_slug",
                "job_path",
                "job_scale",
                "mode",
                "skills_enabled",
                "prompt_hash",
                "prompt_source",
            )
        }
    )
    summary["host_status"] = status
    summary["artifact_paths"] = entry.get("artifact_paths") or {}
    summary.setdefault("runtime_image", runtime_image.get("runtime_image"))
    summary.setdefault("wheel_variant", runtime_image.get("sdk_image_kind"))
    summary.setdefault("command_count", activity.get("command_count"))
    write_json(record_dir / "record_summary.json", summary)


def execute_run_plan(
    compilation: ScenarioCompilation,
    *,
    result_root: Path,
    logs: Iterable[Path] = (),
    runtime_auth_options: RuntimeAuthOptions | None = None,
) -> tuple[dict[str, int], dict[str, object]]:
    path_budget = compilation.scenario.get("path_budget")
    if isinstance(path_budget, int):
        validate_path_budget(
            str(compilation.scenario.get("name") or compilation.run_plan.get("scenario_name")),
            compilation.run_plan.get("entries") if isinstance(compilation.run_plan.get("entries"), list) else [],
            path_budget,
            result_root,
        )
    execution = compilation.write(result_root)
    run_plan = execution.run_plan
    entries = run_plan.get("entries") if isinstance(run_plan.get("entries"), list) else []
    try:
        preflight_docker_images(
            entries,
            result_root=result_root,
            logs=logs,
            runtime_auth_options=runtime_auth_options,
        )
    except ScenarioValidationError as exc:
        write_scenario_summaries(
            result_root,
            {},
            harness_failure={
                "status": "failed",
                "failure_category": "harness_preflight_failure",
                "message": str(exc),
            },
        )
        raise
    statuses: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        prefix = str(entry["run_id"])
        emit(
            "Starting run_id={} agent={} model={} mode={} record_dir={}".format(
                entry["run_id"],
                entry["agent"],
                entry["agent_model"],
                entry["mode"],
                entry["record_dir"],
            ),
            logs=logs,
            prefix=prefix,
        )
        config = case_config_for_entry(entry, result_root, runtime_auth_options)
        status = run_case_safely(config, logs=logs, prefix=prefix)
        statuses[str(entry["run_id"])] = status
        canonicalize_entry_artifacts(result_root, entry, status)
        emit(f"Finished run_id={entry['run_id']} status={status}", logs=logs, prefix=prefix)
        if status != 0 and run_plan.get("fail_fast"):
            emit("Stopping scenario early because fail_fast=true.", logs=logs, stderr=True)
            break
    summary = write_scenario_summaries(result_root, statuses)
    return statuses, summary


def replay_result_root(result_root: Path, *, logs: Iterable[Path] = ()) -> dict[str, object]:
    replay_metadata = {
        "schema_version": "1",
        "replayed": True,
        "agent_invocation": "replayed",
        "replayed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_result_root": str(result_root.resolve()),
        "note": "Replay regenerates parser/report artifacts from captured records and does not execute Docker.",
    }
    write_json(result_root / "replay_metadata.json", replay_metadata)
    run_plan = load_json(result_root / "run_plan.json", {}) or {}
    entries = run_plan.get("entries") if isinstance(run_plan.get("entries"), list) else []
    statuses: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        record_dir = result_root / str(entry["record_dir"])
        events_path = record_dir / "agent_events.jsonl"
        if events_path.is_file():
            adapter = load_agent_adapter(str(entry["agent"]))
            write_json(record_dir / "agent_usage.json", adapter.parse_usage(events_path))
            write_json(record_dir / "agent_activity.json", adapter.parse_activity(events_path))
            emit(f"Replayed parsers for {entry['run_id']}: {events_path}", logs=logs)
        container_exit = load_json(record_dir / "container_exit_code.json", {}) or {}
        if isinstance(container_exit, dict) and isinstance(container_exit.get("exit_code"), int):
            statuses[str(entry["run_id"])] = int(container_exit["exit_code"])
        canonicalize_entry_artifacts(result_root, entry, statuses.get(str(entry["run_id"])))
    return write_scenario_summaries(result_root, statuses)


def write_host_report_status(result_root: Path) -> None:
    scenario_report = result_root / "reports" / "scenario_report.md"
    payload = {
        "status": "ok" if scenario_report.is_file() else "missing_report",
        "scenario_report": str(scenario_report),
    }
    write_json(
        result_root / "host_report_status.json",
        payload,
    )


def combined_exit_status(case_statuses: dict[str, int], report_statuses: dict[str, int] | None = None) -> int:
    report_statuses = report_statuses or {}
    return 1 if any(status != 0 for status in [*case_statuses.values(), *report_statuses.values()]) else 0


def emit_scenario_validation_error(exc: ScenarioValidationError, *, logs: Iterable[Path] = ()) -> int:
    emit(f"Scenario validation failed: {exc}", logs=logs, stderr=True)
    return 1


def run_pair(argv: list[str]) -> int:
    reject_parallel_comparison_runs("pair")
    options = parse_host_cli_options(argv, "pair")
    result_root = comparison_result_root(options)
    result_root.mkdir(parents=True, exist_ok=True)
    clean_pair_result_root(result_root)
    console_log = result_root / "console_output.log"
    console_log.write_text("", encoding="utf-8")
    logs = (console_log,)
    try:
        adapter = agent_adapter_from_options(options)
        compilation = pair_compilation_from_options(options)
        images = image_config_from_adapter(adapter)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc, logs=logs)

    emit(f"Result root: {result_root}", logs=logs)
    emit(f"Console log: {console_log}", logs=logs)
    emit(f"Skills image: {images.image_name}", logs=logs)
    emit(f"Baseline image: {images.baseline_image_name}", logs=logs)
    emit(f"Report image: {images.report_image_name}", logs=logs)
    emit(f"Job folder: {options.job_input}", logs=logs)
    emit(f"Prompt file: {options.prompt_path} -> {CONTAINER_PROMPT_PATH}", logs=logs)

    try:
        run_statuses, scenario_summary = execute_run_plan(
            compilation,
            result_root=result_root,
            logs=logs,
            **execute_auth_kwargs(options),
        )
    except ScenarioValidationError as exc:
        status = emit_scenario_validation_error(exc, logs=logs)
        write_host_report_status(result_root)
        return status

    emit(f"Scenario summary: {result_root / 'scenario_summary.json'}", logs=logs)
    emit(f"Scenario report: {result_root / 'reports' / 'scenario_report.md'}", logs=logs)
    write_host_report_status(result_root)
    return (
        1
        if any(status != 0 for status in run_statuses.values())
        or scenario_summary.get("status") in {"degraded", "failed"}
        else 0
    )


def run_scenario(argv: list[str]) -> int:
    reject_parallel_comparison_runs("scenario")
    options = parse_scenario_cli_options(argv)
    try:
        compilation = compile_scenario_file(options.scenario_path)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc)
    result_root = scenario_result_root(options, compilation)
    result_root.mkdir(parents=True, exist_ok=True)
    console_log = result_root / "console_output.log"
    console_log.write_text("", encoding="utf-8")
    logs = (console_log,)

    emit(f"Result root: {result_root}", logs=logs)
    emit(f"Console log: {console_log}", logs=logs)
    emit(f"Scenario file: {options.scenario_path}", logs=logs)
    emit(f"Run count: {compilation.run_plan.get('run_count')}", logs=logs)

    try:
        statuses, summary = execute_run_plan(
            compilation,
            result_root=result_root,
            logs=logs,
            **execute_auth_kwargs(options),
        )
    except ScenarioValidationError as exc:
        status = emit_scenario_validation_error(exc, logs=logs)
        write_host_report_status(result_root)
        return status
    emit(f"Scenario summary: {result_root / 'scenario_summary.json'}", logs=logs)
    emit(f"Scenario report: {result_root / 'reports' / 'scenario_report.md'}", logs=logs)
    write_host_report_status(result_root)
    return (
        1 if any(status != 0 for status in statuses.values()) or summary.get("status") in {"degraded", "failed"} else 0
    )


def run_replay(argv: list[str]) -> int:
    options = parse_replay_cli_options(argv)
    console_log = options.result_root / "replay_console_output.log"
    console_log.write_text("", encoding="utf-8")
    logs = (console_log,)
    emit(f"Replay result root: {options.result_root}", logs=logs)
    summary = replay_result_root(options.result_root, logs=logs)
    emit(f"Scenario summary: {options.result_root / 'scenario_summary.json'}", logs=logs)
    emit(f"Scenario report: {options.result_root / 'reports' / 'scenario_report.md'}", logs=logs)
    write_host_report_status(options.result_root)
    # Replay regenerates artifacts from an existing result tree; it preserves
    # degraded benchmark status instead of re-asserting pass/fail.
    return 0 if summary.get("status") in {"passed", "degraded"} else 1


def run_interactive(argv: list[str]) -> int:
    options = parse_host_cli_options(argv, "interactive")
    try:
        adapter = agent_adapter_from_options(options)
        agent_model, model_was_explicit = agent_model_from_options(adapter, options)
        images = image_config_from_adapter(adapter)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc)
    runtime_auth_options = runtime_auth_options_from_host_cli(options)
    host_agent_home = runtime_auth_options.agent_home or absolute_path(str(adapter.host_home_from_env(os.environ)))
    mount_agent_auth = (
        runtime_auth_options.mount_agent_auth
        if runtime_auth_options.mount_agent_auth is not None
        else adapter.mount_auth_from_env(os.environ)
    )
    container_records = os.environ.get("CONTAINER_RECORDS", "/tmp/agent_benchmark/records")
    args = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-v",
        f"{options.job_input}:/workspace/input",
        "-v",
        f"{options.prompt_path}:{CONTAINER_PROMPT_PATH}:ro",
        *docker_env("JOB_INPUT_DIR", "/workspace/input"),
        *docker_env("TRAINING_CODE", "/workspace/input"),
        *docker_env("PROMPT_SOURCE", CONTAINER_PROMPT_PATH),
        *docker_env("RECORDS_DIR", container_records),
    ]
    for name, value in sorted(
        adapter.runtime_env(
            InteractiveRuntimeConfig(
                agent=adapter.name,
                agent_model=agent_model,
                model_was_explicit=model_was_explicit,
            )
        ).items()
    ):
        args.extend(docker_env(name, value))
    add_agent_passthrough_env(args, adapter)
    if mount_agent_auth:
        interactive_config = SimpleNamespace(host_agent_home=host_agent_home)
        add_agent_auth_mounts(args, mounts=adapter.auth_mounts(interactive_config))
    emit(f"Mounting job folder: {options.job_input} -> /workspace/input")
    emit(f"Using prompt file: {options.prompt_path} -> {CONTAINER_PROMPT_PATH}")
    try:
        return subprocess.call([*args, images.image_name, "bash"])
    except OSError as exc:
        emit(f"Failed to start interactive container: {type(exc).__name__}: {exc}", stderr=True)
        return 127


def agent_adapter_from_options(options):
    agent_name = getattr(options, "agent", None) or os.environ.get("BENCHMARK_AGENT", DEFAULT_BENCHMARK_AGENT)
    try:
        return load_agent_adapter(agent_name)
    except ValueError as exc:
        raise ScenarioValidationError(str(exc)) from exc


def runtime_auth_options_from_host_cli(options) -> RuntimeAuthOptions:
    return RuntimeAuthOptions(
        agent_home=getattr(options, "agent_home", None),
        mount_agent_auth=getattr(options, "mount_agent_auth", None),
    )


def execute_auth_kwargs(options) -> dict[str, RuntimeAuthOptions]:
    runtime_auth_options = runtime_auth_options_from_host_cli(options)
    return {"runtime_auth_options": runtime_auth_options} if runtime_auth_options.has_overrides else {}


def agent_model_from_options(adapter, options) -> tuple[str, bool]:
    env = os.environ if not getattr(options, "model", None) else {**os.environ, "BENCHMARK_AGENT_MODEL": options.model}
    try:
        return adapter.model_from_env(env), adapter.model_was_explicit(env)
    except ValueError as exc:
        raise ScenarioValidationError(str(exc)) from exc


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage: python -m assist_tools.skills_benchmark.skills.harness.host.runner {run-one,pair,scenario,replay,interactive} "
            "--prompt PATH [--training-code PATH] [--results-root PATH] [PATH]"
        )
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "run-one":
        status = run_one(argv)
    elif command == "pair":
        status = run_pair(argv)
    elif command == "scenario":
        status = run_scenario(argv)
    elif command == "replay":
        status = run_replay(argv)
    elif command == "interactive":
        status = run_interactive(argv)
    else:
        raise SystemExit(f"Unknown command: {command}")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
