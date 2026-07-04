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

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from ..agents.parsers import parse_cached_usage_and_activity
from ..agents.registry import DEFAULT_BENCHMARK_AGENT, load_agent_adapter
from ..behavior_checks import apply_deterministic_behavior_checks
from ..common import load_json, write_json, write_text_atomic
from ..eval_contracts import apply_behavior_contract
from ..modes import PAIR_RUNS, mode_spec
from ..records import (
    apply_expected_skill_compliance,
    apply_instruction_compliance,
    update_failure_analysis,
    write_run_summary,
)
from ..reports import benchmark_insights, generic_reports, metrics_report
from ..safe_paths import (
    UnsafeArtifactPath,
    atomic_replace_bytes,
    path_beneath,
    read_regular_file_beneath,
    reject_symlink_components,
)
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
    prepare_result_mount,
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
BENCHMARK_METRICS_TITLE = "Agent Skills Benchmark Metrics"
FAILURE_STDERR_LINE_LIMIT = 4
FAILURE_STDERR_CHAR_LIMIT = 1200
MAX_CANONICAL_ARTIFACT_BYTES = 5 * 1024 * 1024
MAX_RUN_PLAN_BYTES = 10 * 1024 * 1024
MAX_RUN_PLAN_ENTRIES = 10000
MAX_REPLAY_EVENT_BYTES = 20 * 1024 * 1024
RESULT_ROOT_MARKER = ".nvflare-benchmark-root.json"


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
    require_trusted_input_for_credentials(config)
    prepare_result_mount(config.result_dir, logs=logs, prefix=prefix)
    emit(f"Running mode={config.mode} with runtime image: {config.run_image}", logs=logs, prefix=prefix)
    emit(f"Report image: {config.images.report_image_name}", logs=logs, prefix=prefix)
    emit(f"Job folder: {config.job_input_dir} -> /workspace/input", logs=logs, prefix=prefix)
    emit(f"Prompt file: {config.prompt_path} -> {CONTAINER_PROMPT_PATH}", logs=logs, prefix=prefix)
    write_runtime_image(config)
    with tempfile.TemporaryDirectory(prefix="agent-benchmark-auth-") as auth_staging:
        status = stream_command(
            docker_args_for_case(config, logs=logs, prefix=prefix, auth_staging_dir=Path(auth_staging)),
            logs=logs,
            prefix=prefix,
            timeout_seconds=config.container_timeout_seconds,
        )
    write_json(config.result_dir / "container_exit_code.json", {"exit_code": status})
    if status != 0:
        emit_case_failure_summary(config, status, logs=logs, prefix=prefix)
    if enforce_result_size_budget(config, logs=logs, prefix=prefix):
        return 1
    return combined_exit_status({config.mode: status})


def require_trusted_input_for_credentials(config: CaseConfig) -> None:
    """Require an explicit trust decision before exposing reusable credentials."""

    exposed_env = [name for name in config.adapter.passthrough_env_names() if os.environ.get(name)]
    mounted_auth = []
    if config.mount_host_agent_auth:
        mounted_auth = [mount for mount in config.adapter.auth_mounts(config) if mount.host_path.expanduser().exists()]
    if not exposed_env and not mounted_auth:
        return
    if os.environ.get("BENCHMARK_TRUSTED_INPUT", "").strip().lower() == "true":
        return
    credential_sources = [*exposed_env, *(str(mount.host_path) for mount in mounted_auth)]
    raise ScenarioValidationError(
        "Benchmark inputs can instruct a sandbox-bypassed agent with network access. Refusing to expose reusable "
        "credentials until BENCHMARK_TRUSTED_INPUT=true is set after reviewing every prompt and input file. "
        f"Credential sources: {', '.join(credential_sources)}"
    )


def truncate_text(text: object, limit: int = 240) -> str:
    rendered = str(text or "").strip().replace("\n", " ")
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3].rstrip() + "..."


def bounded_stderr_excerpt(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    excerpt = "\n".join(lines[:FAILURE_STDERR_LINE_LIMIT])
    if len(excerpt) > FAILURE_STDERR_CHAR_LIMIT:
        excerpt = excerpt[: FAILURE_STDERR_CHAR_LIMIT - 3].rstrip() + "..."
    return excerpt


def read_agent_stderr_excerpt(result_dir: Path, exit_summary: dict[str, object]) -> str:
    excerpt = str(exit_summary.get("stderr_excerpt") or "")
    if not excerpt:
        stderr_path = result_dir / "agent_stderr.txt"
        try:
            excerpt = stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            excerpt = ""
    return bounded_stderr_excerpt(excerpt)


def emit_case_failure_summary(
    config: CaseConfig,
    status: int,
    *,
    logs: Iterable[Path] = (),
    prefix: str | None = None,
) -> None:
    result_dir = config.result_dir
    run_summary = load_json(result_dir / "run_summary.json", {}) or {}
    exit_summary = load_json(result_dir / "agent_exit_summary.json", {}) or {}
    early_failure = load_json(result_dir / "early_failure.json", {}) or {}
    host_error = load_json(result_dir / "host_case_error.json", {}) or {}
    if not isinstance(run_summary, dict):
        run_summary = {}
    if not isinstance(exit_summary, dict):
        exit_summary = {}
    if not isinstance(early_failure, dict):
        early_failure = {}
    if not isinstance(host_error, dict):
        host_error = {}

    emit(
        f"Run failed: mode={config.mode}; final_status={status}; result_dir={result_dir}",
        logs=logs,
        prefix=prefix,
        stderr=True,
    )
    agent_exit = run_summary.get("agent_process_exit_code")
    final_exit = run_summary.get("final_container_exit_code")
    if agent_exit is not None or final_exit is not None:
        emit(
            f"Failure exit codes: agent_process_exit={agent_exit}; final_container_exit={final_exit}",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
    failure_category = (
        run_summary.get("failure_root_cause")
        or run_summary.get("failure_category")
        or exit_summary.get("failure_category")
    )
    if failure_category:
        emit(
            f"Failure category: {truncate_text(failure_category)}",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
    harness_message = early_failure.get("message") or host_error.get("message")
    if harness_message:
        phase = early_failure.get("phase") or host_error.get("error_type") or "host"
        emit(
            f"Harness failure detail: {phase}: {truncate_text(harness_message)}",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
    stderr_excerpt = read_agent_stderr_excerpt(result_dir, exit_summary)
    if stderr_excerpt:
        emit("Agent stderr excerpt:", logs=logs, prefix=prefix, stderr=True)
        for line in stderr_excerpt.splitlines():
            emit(f"  {line}", logs=logs, prefix=prefix, stderr=True)
    emit(
        f"Failure artifacts: {result_dir / 'agent_stderr.txt'}, {result_dir / 'agent_exit_summary.json'}, "
        f"{result_dir / 'run_summary.json'}",
        logs=logs,
        prefix=prefix,
        stderr=True,
    )


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
    try:
        initialize_result_root(result_root)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc)
    console_log = result_root / "console_output.log"
    write_text_atomic(console_log, "")
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
    report_statuses = write_benchmark_reports(result_root, logs=logs)
    emit_benchmark_report_paths(result_root, logs=logs)
    write_host_report_status(result_root, report_statuses)
    return (
        1
        if any(status != 0 for status in run_statuses.values())
        or scenario_summary.get("status") in {"degraded", "failed"}
        or any(status != 0 for status in report_statuses.values())
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
    try:
        load_validated_run_plan(result_root)
    except (UnsafeArtifactPath, ValueError) as exc:
        raise SystemExit(f"Replay result root is unsafe or invalid: {exc}") from exc
    return ReplayCliOptions(result_root=result_root)


def validate_run_plan_entries(result_root: Path, run_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    reject_symlink_components(result_root)
    entries = run_plan.get("entries")
    if not isinstance(entries, list) or len(entries) > MAX_RUN_PLAN_ENTRIES:
        raise UnsafeArtifactPath(f"run_plan.entries must be a list capped at {MAX_RUN_PLAN_ENTRIES}")
    validated: list[dict[str, Any]] = []
    run_ids = set()
    record_dirs = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise UnsafeArtifactPath(f"run_plan.entries[{index}] must be an object")
        run_id = entry.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise UnsafeArtifactPath(f"run_plan.entries[{index}].run_id must be unique and non-empty")
        record_value = entry.get("record_dir")
        record_dir = path_beneath(
            result_root,
            record_value,
            label=f"run_plan.entries[{index}].record_dir",
            required_prefix="records",
        )
        normalized = record_dir.relative_to(result_root).as_posix()
        if normalized in record_dirs:
            raise UnsafeArtifactPath(f"duplicate run_plan record_dir: {normalized}")
        run_ids.add(run_id)
        record_dirs.add(normalized)
        validated.append(entry)
    reports_dir = result_root / "reports"
    if reports_dir.exists() or reports_dir.is_symlink():
        reject_symlink_components(reports_dir)
    return validated


def load_validated_run_plan(result_root: Path) -> dict[str, Any]:
    raw = read_regular_file_beneath(
        result_root,
        "run_plan.json",
        label="run_plan.json",
        max_bytes=MAX_RUN_PLAN_BYTES,
    )
    if raw is None:
        raise UnsafeArtifactPath(f"missing bounded regular run_plan.json beneath {result_root}")
    try:
        run_plan = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"run_plan.json is invalid JSON: {exc}") from exc
    if not isinstance(run_plan, dict):
        raise ValueError("run_plan.json must contain an object")
    validate_run_plan_entries(result_root, run_plan)
    return run_plan


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

    marker = result_root / RESULT_ROOT_MARKER
    if not marker.is_file() or marker.is_symlink():
        raise ScenarioValidationError(
            f"Refusing to clean unowned output directory without {RESULT_ROOT_MARKER}: {result_root}"
        )

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


def initialize_result_root(result_root: Path) -> None:
    """Create an exclusive host-owned output root without deleting existing data."""

    reject_symlink_components(result_root)
    if result_root.exists():
        if result_root.is_symlink() or not result_root.is_dir():
            raise ScenarioValidationError(f"Result root must be a non-symlink directory: {result_root}")
        try:
            existing = list(result_root.iterdir())
        except OSError as exc:
            raise ScenarioValidationError(f"Could not inspect result root {result_root}: {exc}") from exc
        if existing:
            raise ScenarioValidationError(
                f"Result root must be new or empty; refusing to delete or mix existing data: {result_root}"
            )
    else:
        result_root.mkdir(parents=True, mode=0o700)
    try:
        result_root.chmod(0o700)
    except OSError:
        pass
    write_json(
        result_root / RESULT_ROOT_MARKER,
        {
            "schema_version": "1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "owner": "nvflare_agent_skills_benchmark",
        },
    )


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
        result_dir=path_beneath(
            result_root,
            entry.get("record_dir"),
            label="run_plan entry record_dir",
            required_prefix="records",
        ),
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
        memory_limit_bytes=positive_int_resource_value(resource_policy.get("memory_limit_bytes")),
        cpu_limit_millis=positive_int_resource_value(resource_policy.get("cpu_limit_millis")),
        pids_limit=positive_int_resource_value(resource_policy.get("pids_limit")),
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


def docker_context_name() -> str:
    try:
        result = subprocess.run(
            ["docker", "context", "show"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"
    if result.returncode == 0:
        return result.stdout.strip() or "unknown"
    detail = result.stderr.strip() or result.stdout.strip()
    return f"unavailable ({detail})" if detail else "unavailable"


def docker_benchmark_image_list() -> list[str]:
    try:
        result = subprocess.run(
            [
                "docker",
                "image",
                "ls",
                "agent-skills-benchmark",
                "--format",
                "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.CreatedSince}}\t{{.Size}}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return [f"unavailable ({type(exc).__name__}: {exc})"]
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return [f"unavailable ({detail})"] if detail else ["unavailable"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def preflight_docker_images(
    entries: Iterable[dict[str, object]],
    *,
    result_root: Path,
    logs: Iterable[Path] = (),
    runtime_auth_options: RuntimeAuthOptions | None = None,
) -> None:
    images_by_run: dict[str, list[str]] = {}
    agents_by_image: dict[str, set[str]] = {}
    inspect_results: dict[str, dict[str, object]] = {}
    docker_context = docker_context_name()
    local_benchmark_images = docker_benchmark_image_list()
    for entry in entries:
        config = case_config_for_entry(entry, result_root, runtime_auth_options)
        images_by_run.setdefault(config.run_image, []).append(str(entry.get("run_id")))
        agents_by_image.setdefault(config.run_image, set()).add(config.agent)
    for image in sorted(images_by_run):
        available, detail = inspect_docker_image(image)
        inspect_results[image] = {
            "available": available,
            "agents": sorted(agents_by_image.get(image, set())),
            "detail": detail,
            "run_ids": images_by_run[image],
        }
    missing = [image for image, result in inspect_results.items() if not result["available"]]
    payload = {
        "status": "fail" if missing else "pass",
        "docker_context": docker_context,
        "images": inspect_results,
        "local_benchmark_images": local_benchmark_images,
        "missing_images": missing,
    }
    write_json(result_root / "docker_image_preflight.json", payload)
    if not missing:
        return
    message = (
        "Benchmark Docker image(s) are missing locally or could not be inspected: "
        + ", ".join(missing)
        + f". Docker context: {docker_context}. "
        + "Selected benchmark agent(s): "
        + ", ".join(sorted({agent for image in missing for agent in agents_by_image.get(image, set())}) or ["unknown"])
        + ". "
        + "Run ./bin/build.sh from dev_tools/agent/skills/benchmark before running the benchmark, "
        + "and verify the same Docker context is used for build and run."
    )
    details = [f"{image}: {str(inspect_results[image].get('detail') or 'not found')}" for image in missing]
    if details:
        message += " Inspect details: " + "; ".join(details)
    if local_benchmark_images:
        message += " Local agent-skills-benchmark tags: " + "; ".join(local_benchmark_images)
    emit(message, logs=logs, stderr=True)
    raise ScenarioValidationError(message)


def copy_file_if_present(source: Path, target: Path, *, root: Path | None = None) -> bool:
    trusted_root = root or source.parent
    try:
        relative = source.relative_to(trusted_root).as_posix()
    except ValueError:
        return False
    data = read_regular_file_beneath(
        trusted_root,
        relative,
        label="canonical artifact source",
        max_bytes=MAX_CANONICAL_ARTIFACT_BYTES,
    )
    if data is None:
        return False
    atomic_replace_bytes(target, data)
    return True


def load_json_beneath(root: Path, relative: str, *, max_bytes: int = MAX_CANONICAL_ARTIFACT_BYTES) -> Any:
    raw = read_regular_file_beneath(root, relative, label=relative, max_bytes=max_bytes)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def remove_stale_file(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
        raise UnsafeArtifactPath(f"refusing to replace non-file canonical artifact: {path}")
    path.unlink()


def restrict_result_permissions(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (directory / name).is_symlink()]
        try:
            directory.chmod(0o700)
        except OSError:
            pass
        for filename in filenames:
            path = directory / filename
            try:
                if not path.is_symlink():
                    path.chmod(0o600)
            except OSError:
                pass


def eval_fixture_matches_contract(contract: Mapping[str, Any], job_path: Path) -> bool:
    files = contract.get("files")
    if not isinstance(files, list) or not job_path.is_dir() or job_path.is_symlink():
        return False
    if not files:
        try:
            return not any(job_path.iterdir())
        except OSError:
            return False
    for entry in files:
        if not isinstance(entry, Mapping):
            return False
        relative_value = entry.get("fixture_path")
        if not isinstance(relative_value, str) or not relative_value or "\\" in relative_value:
            return False
        relative = Path(*relative_value.split("/"))
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = job_path / relative
        try:
            if path.is_symlink() or not path.is_file():
                return False
            raw = path.read_bytes()
        except OSError:
            return False
        if len(raw) != entry.get("size_bytes") or hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            return False
    return True


def evaluate_entry_behavior(record_dir: Path, entry: Mapping[str, Any]) -> None:
    contract = entry.get("behavior_contract")
    if not isinstance(contract, Mapping):
        return
    record_path = record_dir / "benchmark_record.json"
    record = load_json(record_path, {}) or {}
    if not isinstance(record, dict):
        record = {}
    apply_behavior_contract(record, contract)
    job_path = Path(str(entry.get("job_path") or ""))
    fixture_matches = eval_fixture_matches_contract(contract, job_path)
    record["eval_contract_status"] = "passed" if fixture_matches else "failed"
    workspace_delta_manifest_path = record_dir / "workspace_delta_manifest.json"
    workspace_delta = load_json_beneath(record_dir, "workspace_delta_manifest.json") or {}
    if fixture_matches and isinstance(workspace_delta, dict):
        apply_deterministic_behavior_checks(
            record,
            input_dir=job_path,
            workspace_delta=workspace_delta,
            workspace_delta_manifest_path=workspace_delta_manifest_path,
        )
    else:
        for category in ("mandatory_behavior", "prohibited_behavior"):
            behavior_map = record.get(category)
            if not isinstance(behavior_map, dict):
                continue
            for behavior in behavior_map.values():
                if isinstance(behavior, dict):
                    behavior["status"] = "missing"
                    behavior["evidence"] = "Captured input did not match the immutable eval fixture manifest."
                    behavior["source"] = "eval_contract_validation"
    apply_instruction_compliance(record)
    apply_expected_skill_compliance(record)
    update_failure_analysis(record)
    write_json(record_path, record)
    write_run_summary(record_path, record_dir / "record_summary.json", print_summary=False)


def canonicalize_entry_artifacts(
    result_root: Path,
    entry: dict[str, object],
    status: int | None,
    *,
    preserve_existing: bool = False,
    evaluate_behavior: bool = True,
) -> None:
    record_dir = path_beneath(
        result_root,
        entry.get("record_dir"),
        label="run_plan entry record_dir",
        required_prefix="records",
    )
    mode = str(entry["mode"])
    record_dir.mkdir(parents=True, exist_ok=True)
    write_json(record_dir / "run_plan_entry.json", entry)
    copies = (
        (record_dir / "records" / f"{mode}_agent_record.json", record_dir / "agent_record.json"),
        (record_dir / "records" / f"{mode}_record.json", record_dir / "benchmark_record.json"),
        (record_dir / "run_summary.json", record_dir / "record_summary.json"),
    )
    for source, target in copies:
        if preserve_existing:
            existing = read_regular_file_beneath(
                record_dir,
                target.relative_to(record_dir).as_posix(),
                label="existing canonical artifact",
                max_bytes=MAX_CANONICAL_ARTIFACT_BYTES,
            )
            if existing is not None:
                continue
        if not copy_file_if_present(source, target, root=record_dir):
            remove_stale_file(target)
    if evaluate_behavior:
        evaluate_entry_behavior(record_dir, entry)
    summary = load_json_beneath(record_dir, "record_summary.json") or {}
    if not isinstance(summary, dict):
        summary = {}
    runtime_image = load_json_beneath(record_dir, "runtime_image.json") or {}
    if not isinstance(runtime_image, dict):
        runtime_image = {}
    activity = load_json_beneath(record_dir, "agent_activity.json") or {}
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
    restrict_result_permissions(record_dir)


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
    entries = validate_run_plan_entries(result_root, run_plan)
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
    run_plan = load_validated_run_plan(result_root)
    replay_metadata = {
        "schema_version": "1",
        "replayed": True,
        "agent_invocation": "replayed",
        "replayed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_result_root": str(result_root.resolve()),
        "note": "Replay regenerates parser/report artifacts from captured records and does not execute Docker.",
    }
    write_json(result_root / "replay_metadata.json", replay_metadata)
    entries = validate_run_plan_entries(result_root, run_plan)
    statuses: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        record_dir = path_beneath(
            result_root,
            entry.get("record_dir"),
            label="run_plan entry record_dir",
            required_prefix="records",
        )
        events_data = read_regular_file_beneath(
            record_dir,
            "agent_events.jsonl",
            label="agent_events.jsonl",
            max_bytes=MAX_REPLAY_EVENT_BYTES,
        )
        if events_data is not None:
            adapter = load_agent_adapter(str(entry["agent"]))
            with tempfile.TemporaryDirectory(prefix="agent-benchmark-replay-events-") as staging:
                staged_events = Path(staging) / "agent_events.jsonl"
                staged_events.write_bytes(events_data)
                write_json(record_dir / "agent_usage.json", adapter.parse_usage(staged_events))
                write_json(record_dir / "agent_activity.json", adapter.parse_activity(staged_events))
            emit(f"Replayed parsers for {entry['run_id']}: bounded captured events", logs=logs)
        container_exit = load_json_beneath(record_dir, "container_exit_code.json") or {}
        if isinstance(container_exit, dict) and isinstance(container_exit.get("exit_code"), int):
            statuses[str(entry["run_id"])] = int(container_exit["exit_code"])
        canonicalize_entry_artifacts(
            result_root,
            entry,
            statuses.get(str(entry["run_id"])),
            preserve_existing=True,
            evaluate_behavior=False,
        )
    return write_scenario_summaries(result_root, statuses)


def write_report_generator_status(result_root: Path, statuses: dict[str, int]) -> None:
    write_json(
        result_root / "report_generator_status.json",
        {
            "status": "ok" if all(status == 0 for status in statuses.values()) else "failed",
            "exit_codes": statuses,
        },
    )


def needs_run_id_reports(result_root: Path) -> bool:
    run_plan = load_json(result_root / "run_plan.json", {}) or {}
    entries = run_plan.get("entries") if isinstance(run_plan, dict) else None
    if not isinstance(entries, list) or not entries:
        return False
    mode_counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            return True
        mode = str(entry.get("mode") or "")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    known_modes = {spec.mode for spec in PAIR_RUNS}
    return (
        len(entries) > len(PAIR_RUNS)
        or any(count > 1 for count in mode_counts.values())
        or not set(mode_counts).issubset(known_modes)
    )


def write_benchmark_reports(result_root: Path, *, logs: Iterable[Path] = ()) -> dict[str, int]:
    statuses: dict[str, int] = {}
    try:
        if needs_run_id_reports(result_root):
            try:
                generic_reports.write_reports(result_root, BENCHMARK_METRICS_TITLE)
            except Exception as exc:
                write_host_error(result_root / "generic_reports_error.json", exc)
                emit(f"Run-id report generation failed: {type(exc).__name__}: {exc}", logs=logs, stderr=True)
                statuses.update({"metrics_report": 1, "benchmark_insights": 1})
            else:
                statuses.update({"metrics_report": 0, "benchmark_insights": 0})
            write_report_generator_status(result_root, statuses)
            return statuses
        try:
            metrics_report.write_reports(result_root, BENCHMARK_METRICS_TITLE)
        except Exception as exc:
            write_host_error(result_root / "metrics_report_error.json", exc)
            emit(f"Metrics report failed: {type(exc).__name__}: {exc}", logs=logs, stderr=True)
            statuses["metrics_report"] = 1
        else:
            statuses["metrics_report"] = 0

        try:
            runs = benchmark_insights.collect_benchmark_runs(result_root)
            write_text_atomic(
                result_root / "benchmark_insights.md",
                benchmark_insights.benchmark_report(result_root, runs),
            )
        except Exception as exc:
            write_host_error(result_root / "benchmark_insights_error.json", exc)
            emit(f"Benchmark insights report failed: {type(exc).__name__}: {exc}", logs=logs, stderr=True)
            statuses["benchmark_insights"] = 1
        else:
            statuses["benchmark_insights"] = 0

        write_report_generator_status(result_root, statuses)
        return statuses
    finally:
        parse_cached_usage_and_activity.cache_clear()


def emit_benchmark_report_paths(result_root: Path, *, logs: Iterable[Path] = ()) -> None:
    emit(f"Benchmark insights: {result_root / 'benchmark_insights.md'}", logs=logs)
    emit(f"Metrics report: {result_root / 'metrics_report.md'}", logs=logs)
    emit(f"Metrics HTML: {result_root / 'metrics_report.html'}", logs=logs)


def write_host_report_status(result_root: Path, report_generator_statuses: dict[str, int] | None = None) -> None:
    report_generator_statuses = report_generator_statuses or {}
    scenario_report = result_root / "reports" / "scenario_report.md"
    benchmark_report = result_root / "benchmark_insights.md"
    metrics_report_path = result_root / "metrics_report.md"
    benchmark_reports_expected = bool(report_generator_statuses)
    all_reports_ok = (
        scenario_report.is_file()
        and all(status == 0 for status in report_generator_statuses.values())
        and (not benchmark_reports_expected or (benchmark_report.is_file() and metrics_report_path.is_file()))
    )
    payload = {
        "status": "ok" if all_reports_ok else "missing_report",
        "scenario_report": str(scenario_report),
        "benchmark_insights": str(benchmark_report),
        "metrics_report": str(metrics_report_path),
        "report_generators": report_generator_statuses,
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
    try:
        initialize_result_root(result_root)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc)
    console_log = result_root / "console_output.log"
    write_text_atomic(console_log, "")
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
    report_statuses = write_benchmark_reports(result_root, logs=logs)
    emit_benchmark_report_paths(result_root, logs=logs)
    write_host_report_status(result_root, report_statuses)
    return (
        1
        if any(status != 0 for status in run_statuses.values())
        or scenario_summary.get("status") in {"degraded", "failed"}
        or any(status != 0 for status in report_statuses.values())
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
    try:
        initialize_result_root(result_root)
    except ScenarioValidationError as exc:
        return emit_scenario_validation_error(exc)
    console_log = result_root / "console_output.log"
    write_text_atomic(console_log, "")
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
    report_statuses = write_benchmark_reports(result_root, logs=logs)
    emit_benchmark_report_paths(result_root, logs=logs)
    write_host_report_status(result_root, report_statuses)
    return (
        1
        if any(status != 0 for status in statuses.values())
        or summary.get("status") in {"degraded", "failed"}
        or any(status != 0 for status in report_statuses.values())
        else 0
    )


def run_replay(argv: list[str]) -> int:
    options = parse_replay_cli_options(argv)
    console_log = options.result_root / "replay_console_output.log"
    write_text_atomic(console_log, "")
    logs = (console_log,)
    emit(f"Replay result root: {options.result_root}", logs=logs)
    summary = replay_result_root(options.result_root, logs=logs)
    emit(f"Scenario summary: {options.result_root / 'scenario_summary.json'}", logs=logs)
    emit(f"Scenario report: {options.result_root / 'reports' / 'scenario_report.md'}", logs=logs)
    report_statuses = write_benchmark_reports(options.result_root, logs=logs)
    emit_benchmark_report_paths(options.result_root, logs=logs)
    write_host_report_status(options.result_root, report_statuses)
    # Replay regenerates artifacts from an existing result tree; it preserves
    # degraded benchmark status instead of re-asserting pass/fail.
    return (
        0
        if summary.get("status") in {"passed", "degraded"} and all(status == 0 for status in report_statuses.values())
        else 1
    )


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
            "Usage: python -m skills.harness.host.runner {run-one,pair,scenario,replay,interactive} "
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
