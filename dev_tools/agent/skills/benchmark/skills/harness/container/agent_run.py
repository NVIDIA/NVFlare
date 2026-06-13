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

"""In-container agent benchmark lifecycle runner."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext, suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agents.base import (
    AgentAdapter,
    AgentLaunchContext,
    AgentLaunchSpec,
    FinalMessageSource,
    SkillExposureContext,
    SkillExposureResult,
)
from ..agents.registry import load_agent_adapter
from ..artifacts import capture_workspace_delta, write_workspace_baseline
from ..common import bool_from_text, load_json, make_tree_readable, write_json
from ..modes import mode_spec
from ..records import AgentRecordSynthesisInputs, merge_record, synthesize_agent_record, write_run_summary
from ..timing import LifecycleEpochs, finalize_timing
from .progress import ProgressWriter
from .skills import apply_skill_exposure
from .skills import copy_optional_metadata_files as _copy_optional_metadata_files

DEFAULT_CONTAINER_VENV_DIR = "/workspace/venv"
RUNTIME_ARTIFACT_ROOT = Path("/tmp/agent_benchmark")
NVFLARE_RUNTIME_WORKSPACES_ROOT = Path("/tmp/nvflare/workspaces")
AGENT_TIMEOUT_EXIT_CODE = 124
AGENT_TERMINATE_GRACE_SECONDS = 10
MAX_STDOUT_TAIL_LINES = 1000
MAX_STDOUT_TAIL_LINE_BYTES = 4096
STDOUT_TAIL_TRUNCATED_MARKER = "...[truncated]"


def epoch_seconds() -> int:
    return int(time.time())


def epoch_nanoseconds() -> int:
    return time.time_ns()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def required_bool(name: str, value: str) -> bool:
    if value not in {"true", "false"}:
        raise SystemExit(f"{name} must be true or false; got {value}")
    return bool_from_text(value)


@dataclass(frozen=True)
class AgentRunConfig:
    mode: str
    use_preinstalled_skills: bool
    job_input_dir: Path
    result_dir: Path
    records_dir: Path
    run_root: Path
    prompt_source: Path
    progress_interval_seconds: int
    sdk_image_kind: str
    agent: str
    agent_model: str
    agent_home: Path
    agent_model_was_explicit: bool
    agent_timeout_seconds: int | None = None

    @property
    def skill_run_mode(self) -> str:
        return self.mode

    @property
    def run_input_dir(self) -> Path:
        return self.run_root / "input"

    @property
    def run_workspace_dir(self) -> Path:
        return self.run_root / "workspace"

    @property
    def agent_record_path(self) -> Path:
        return self.records_dir / f"{self.mode}_agent_record.json"

    @property
    def final_record_path(self) -> Path:
        return self.records_dir / f"{self.mode}_record.json"

    @property
    def agent_events_path(self) -> Path:
        return self.result_dir / "agent_events.jsonl"

    @property
    def agent_usage_path(self) -> Path:
        return self.result_dir / "agent_usage.json"

    @property
    def agent_activity_path(self) -> Path:
        return self.result_dir / "agent_activity.json"

    @property
    def agent_last_message_path(self) -> Path:
        return self.result_dir / "agent_last_message.txt"

    @property
    def agent_stderr_path(self) -> Path:
        return self.result_dir / "agent_stderr.txt"

    @property
    def prompt_file_path(self) -> Path:
        return self.result_dir / "prompt.txt"

    @property
    def progress_log_path(self) -> Path:
        return self.result_dir / "progress.jsonl"

    @classmethod
    def from_env(cls) -> "AgentRunConfig":
        env = os.environ
        mode = env.get("MODE", "with_skills")
        try:
            spec = mode_spec(mode)
        except ValueError as exc:
            raise SystemExit(str(exc).replace("Unknown mode", "Unknown MODE")) from exc
        use_preinstalled = env.get("USE_PREINSTALLED_SKILLS")
        if use_preinstalled is None:
            use_preinstalled = "true" if spec.skills_enabled else "false"
        use_preinstalled_skills = required_bool("USE_PREINSTALLED_SKILLS", use_preinstalled)
        if use_preinstalled_skills != spec.skills_enabled:
            expected = "true" if spec.skills_enabled else "false"
            raise SystemExit(
                f"USE_PREINSTALLED_SKILLS={use_preinstalled} conflicts with MODE={mode}; expected {expected}."
            )
        job_input = env.get("JOB_INPUT_DIR") or env.get("TRAINING_CODE") or "/workspace/input"
        result_dir = env.get("RESULT_DIR", "/workspace/results")
        agent_name = env.get("BENCHMARK_AGENT")
        if not agent_name:
            raise SystemExit("BENCHMARK_AGENT is required inside the benchmark container.")
        adapter = load_agent_adapter(agent_name)
        agent_model = adapter.model_from_env(env)
        agent_home = Path(env.get("BENCHMARK_AGENT_HOME") or env.get(adapter.agent_home_env, adapter.container_home))
        agent_timeout_seconds = optional_positive_int_env("AGENT_TIMEOUT_SECONDS", env.get("AGENT_TIMEOUT_SECONDS"))
        progress_interval_seconds = (
            optional_positive_int_env("PROGRESS_INTERVAL_SECONDS", env.get("PROGRESS_INTERVAL_SECONDS")) or 60
        )
        return cls(
            mode=mode,
            use_preinstalled_skills=use_preinstalled_skills,
            job_input_dir=Path(job_input),
            result_dir=Path(result_dir),
            records_dir=Path(env.get("RECORDS_DIR", str(Path(result_dir) / "records"))),
            run_root=Path(env.get("RUN_ROOT", f"/workspace/run/{mode}")),
            prompt_source=Path(env.get("PROMPT_SOURCE", "/workspace/prompts/benchmark_prompt.txt")),
            progress_interval_seconds=progress_interval_seconds,
            sdk_image_kind=env.get("SDK_IMAGE_KIND", "unknown"),
            agent=adapter.name,
            agent_model=agent_model,
            agent_home=agent_home,
            agent_model_was_explicit=adapter.model_was_explicit(env),
            agent_timeout_seconds=agent_timeout_seconds,
        )


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def optional_positive_int_env(name: str, value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer; got {value}") from exc
    if parsed <= 0:
        raise SystemExit(f"{name} must be positive; got {value}")
    return parsed


def login_shell_runtime_probe() -> dict[str, Any]:
    sdk_import_name = os.environ.get("SDK_IMPORT_NAME", "")
    sdk_version_command = os.environ.get("SDK_VERSION_COMMAND", "")
    sdk_import_probe = (
        "python -c 'import importlib, os; "
        'name=os.environ.get("SDK_IMPORT_NAME", ""); '
        "module=importlib.import_module(name) if name else None; "
        'print("sdk_import_name=" + name); '
        'print("sdk_import_version=" + (getattr(module, "__version__", "unknown") if module else ""))'
        "'"
    )
    script = "\n".join(
        [
            "printf 'PATH=%s\\n' \"$PATH\"",
            "printf 'python=%s\\n' \"$(command -v python)\"",
            sdk_import_probe,
        ]
    )
    command = ["/bin/bash", "-lc", script]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return {
            "command": command,
            "exit_code": 127,
            "output": f"{type(exc).__name__}: {exc}",
            "ok": False,
            "reason": "failed_to_start_login_shell_probe",
        }

    sdk_version_output = None
    sdk_version_exit_code = None
    sdk_version_error = None
    if sdk_version_command:
        try:
            sdk_version_argv = shlex.split(sdk_version_command)
        except ValueError as exc:
            sdk_version_error = f"failed_to_parse_sdk_version_command: {exc}"
        else:
            if not sdk_version_argv:
                sdk_version_error = "empty_sdk_version_command"
            else:
                try:
                    sdk_version_result = subprocess.run(
                        sdk_version_argv,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                except OSError as exc:
                    sdk_version_exit_code = 127
                    sdk_version_error = f"{type(exc).__name__}: {exc}"
                else:
                    sdk_version_exit_code = sdk_version_result.returncode
                    sdk_version_output = sdk_version_result.stdout.strip()

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    expected_venv = os.environ.get("BENCHMARK_CONTAINER_VENV_DIR") or os.environ.get(
        "VIRTUAL_ENV", DEFAULT_CONTAINER_VENV_DIR
    )
    expected_python = str(Path(expected_venv) / "bin" / "python")
    ok = result.returncode == 0 and values.get("python") == expected_python
    reason = "ok"
    if result.returncode != 0:
        reason = f"probe_exit_code_{result.returncode}"
    elif values.get("python") != expected_python:
        reason = f"python_resolved_to_{values.get('python') or 'missing'}"
    return {
        "command": command,
        "exit_code": result.returncode,
        "output": result.stdout,
        "path": values.get("PATH"),
        "python": values.get("python"),
        "sdk_import_name": sdk_import_name,
        "sdk_import_version": values.get("sdk_import_version"),
        "sdk_version_command": sdk_version_command,
        "sdk_version_exit_code": sdk_version_exit_code,
        "sdk_version_output": sdk_version_output,
        "sdk_version_error": sdk_version_error,
        "expected_python": expected_python,
        "ok": ok,
        "reason": reason,
    }


def agent_launch_context(config: AgentRunConfig) -> AgentLaunchContext:
    return AgentLaunchContext(
        workspace_dir=config.run_workspace_dir,
        prompt_file=config.prompt_file_path,
        result_dir=config.result_dir,
        events_dest=config.agent_events_path,
        stderr_dest=config.agent_stderr_path,
        final_message_dest=config.agent_last_message_path,
        model=config.agent_model,
        model_was_explicit=config.agent_model_was_explicit,
        timeout_seconds=config.agent_timeout_seconds,
    )


def build_agent_launch(config: AgentRunConfig) -> AgentLaunchSpec:
    adapter = load_agent_adapter(config.agent)
    return adapter.launch_spec(agent_launch_context(config))


def launch_subprocess_argv(argv: list[str], *, login_shell: bool) -> list[str]:
    if not login_shell:
        return list(argv)
    return ["/bin/bash", "--login", "-c", "exec " + shlex.join(argv)]


def persist_container_runtime_metadata(config: AgentRunConfig, launch: AgentLaunchSpec | None = None) -> None:
    build_metadata = load_json(config.agent_home / "build_metadata.json", {}) or {}
    sdk_wheel_metadata = load_json(config.agent_home / "sdk_wheel_metadata.json", {}) or {}
    if build_metadata:
        write_json(config.result_dir / "image_build_metadata.json", build_metadata)
    if sdk_wheel_metadata:
        write_json(config.result_dir / "sdk_wheel_metadata.json", sdk_wheel_metadata)

    runtime_path = config.result_dir / "runtime_image.json"
    runtime_metadata = load_json(runtime_path, {}) or {}
    launch = launch or build_agent_launch(config)
    probe = (
        login_shell_runtime_probe()
        if launch.login_shell
        else {
            "ok": True,
            "reason": "skipped_adapter_does_not_use_login_shell",
            "required": False,
        }
    )
    runtime_metadata.update(
        {
            "container_python_executable": sys.executable,
            "container_python_version": sys.version.split()[0],
            "container_virtual_env": os.environ.get("VIRTUAL_ENV"),
            "container_path_prefix": os.environ.get("PATH", "").split(":")[:3],
            "container_pip_version": command_output([sys.executable, "-m", "pip", "--version"]),
            "container_uv_version": command_output(["uv", "--version"]),
            "login_shell_required": launch.login_shell,
            "login_shell_runtime_probe": probe,
            "image_build_metadata": build_metadata,
            "sdk_wheel_metadata": sdk_wheel_metadata,
        }
    )
    write_json(runtime_path, runtime_metadata)
    if launch.login_shell and not probe.get("ok"):
        raise RuntimeError(f"Login-shell runtime probe failed: {probe.get('reason')}")


def run_agent_availability_probe(config: AgentRunConfig) -> None:
    adapter = load_agent_adapter(config.agent)
    command = adapter.availability_probe()
    if not command:
        write_json(
            config.result_dir / "agent_availability_probe.json",
            {"status": "skipped", "reason": "adapter did not define an availability probe"},
        )
        return
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=agent_subprocess_env(adapter.runtime_env(config), adapter),
        )
    except OSError as exc:
        write_json(
            config.result_dir / "agent_availability_probe.json",
            {
                "status": "failed",
                "command": command,
                "exit_code": 127,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise RuntimeError(f"Agent availability probe failed to start: {type(exc).__name__}: {exc}") from exc
    write_json(
        config.result_dir / "agent_availability_probe.json",
        {
            "status": "passed" if result.returncode == 0 else "failed",
            "command": command,
            "exit_code": result.returncode,
            "output": result.stdout,
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"Agent availability probe failed with exit code {result.returncode}: {command}")


def setup_skill_availability(config: AgentRunConfig) -> tuple[int, int, SkillExposureResult]:
    start = epoch_seconds()
    adapter = load_agent_adapter(config.agent)
    spec = adapter.skill_exposure(
        SkillExposureContext(
            result_dir=config.result_dir,
            container_home=config.agent_home,
            mode=config.mode,
            skills_enabled=config.use_preinstalled_skills,
            sdk_image_kind=config.sdk_image_kind,
        )
    )
    result = apply_skill_exposure(
        spec=spec,
        skills_enabled=config.use_preinstalled_skills,
        result_dir=config.result_dir,
        sdk_image_kind=config.sdk_image_kind,
    )
    write_json(config.result_dir / "skills_exposure_result.json", asdict(result))
    return start, epoch_seconds(), result


def copy_optional_metadata_files(source_dir: Path, result_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    return _copy_optional_metadata_files(source_dir, result_dir, names)


def validate_input_symlinks(input_dir: Path) -> None:
    root = input_dir.resolve()
    for dirpath, dirnames, filenames in os.walk(input_dir, followlinks=False):
        current = Path(dirpath)
        for name in [*dirnames, *filenames]:
            path = current / name
            if not path.is_symlink():
                continue
            target = path.readlink()
            resolved_target = (target if target.is_absolute() else path.parent / target).resolve(strict=False)
            if not resolved_target.is_relative_to(root):
                rel = path.relative_to(input_dir)
                raise RuntimeError(f"Job input symlink escapes input directory: {rel} -> {target}")


def prepare_input_workspace(config: AgentRunConfig) -> tuple[int, int]:
    start = epoch_seconds()
    config.run_root.mkdir(parents=True, exist_ok=True)
    for name in ("input", "generated", "job_config", "workspace"):
        shutil.rmtree(config.run_root / name, ignore_errors=True)
    validate_input_symlinks(config.job_input_dir)
    shutil.copytree(config.job_input_dir, config.run_input_dir, symlinks=False)
    shutil.copytree(config.run_input_dir, config.run_workspace_dir, symlinks=False)
    for name in ("generated", "job_config"):
        (config.run_root / name).mkdir(parents=True, exist_ok=True)
    return start, epoch_seconds()


def prepare_prompt(config: AgentRunConfig) -> tuple[int, int]:
    start = epoch_seconds()
    shutil.copy2(config.prompt_source, config.prompt_file_path)
    prompt_bytes = config.prompt_file_path.read_bytes()
    write_json(
        config.result_dir / "prompt_metadata.json",
        {
            "template_path": str(config.prompt_source),
            "prompt_path": str(config.prompt_file_path),
            "template_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "template_bytes": len(prompt_bytes),
            "prompt_bytes": len(prompt_bytes),
            "verbatim_copy": True,
            "harness_prompt_injection": False,
            "note": "The harness copies the mounted prompt file verbatim and does not append mode, path, or skill instructions.",
        },
    )
    return start, epoch_seconds()


def write_agent_compatibility_copies(config: AgentRunConfig) -> None:
    # Use file copies rather than symlinks because benchmark results may live on
    # Docker volume mounts where symlink behavior varies by host platform.
    adapter = load_agent_adapter(config.agent)
    suffixes = {
        "events": config.agent_events_path,
        "usage": config.agent_usage_path,
        "activity": config.agent_activity_path,
        "last_message": config.agent_last_message_path,
        "stderr": config.agent_stderr_path,
    }
    for prefix in adapter.artifact_alias_prefixes():
        for suffix, source in suffixes.items():
            target = (
                config.result_dir
                / f"{prefix}_{suffix}.{'jsonl' if suffix == 'events' else 'json' if suffix in {'usage', 'activity'} else 'txt'}"
            )
            if source.exists() and source != target:
                shutil.copy2(source, target)


AGENT_ENV_DENYLIST = {
    "MODE",
    "USE_PREINSTALLED_SKILLS",
    "SDK_IMAGE_KIND",
    "RESULT_DIR",
    "RECORDS_DIR",
    "SDK_AGENT_RECORD",
    "RUN_ROOT",
    "PROMPT_SOURCE",
    "JOB_INPUT_DIR",
    "TRAINING_CODE",
    "PROGRESS_INTERVAL_SECONDS",
    "BENCHMARK_AGENT",
    "BENCHMARK_AGENT_MODEL",
    "BENCHMARK_AGENT_HOME",
    "AGENT_TIMEOUT_SECONDS",
}


def agent_subprocess_env(launch_env: dict[str, str], adapter=None) -> dict[str, str]:
    denied = set(AGENT_ENV_DENYLIST)
    if adapter is not None:
        denied.update(str(item) for item in adapter.model_env_names())
    env = {key: value for key, value in os.environ.items() if key not in denied}
    env.update(launch_env)
    return env


def raw_tail_text(lines: deque[str], tail_bytes: int | None) -> str:
    text = "".join(lines)
    if not tail_bytes or tail_bytes <= 0:
        return text
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= tail_bytes:
        return text
    return encoded[-tail_bytes:].decode("utf-8", errors="replace")


def truncate_stdout_tail_line(line: str) -> str:
    encoded = line.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_STDOUT_TAIL_LINE_BYTES:
        return line
    marker = f"{STDOUT_TAIL_TRUNCATED_MARKER}\n" if line.endswith("\n") else STDOUT_TAIL_TRUNCATED_MARKER
    marker_bytes = marker.encode("utf-8")
    head = encoded[: max(0, MAX_STDOUT_TAIL_LINE_BYTES - len(marker_bytes))]
    return head.decode("utf-8", errors="replace") + marker


def event_matches_selector(event: dict[str, Any], selector: dict[str, Any] | None) -> bool:
    if not selector:
        return True
    return all(event.get(key) == value for key, value in selector.items())


def event_message_text(event: dict[str, Any]) -> str:
    for key in ("final_message", "message", "content", "text", "output", "value"):
        if key not in event:
            continue
        value = event[key]
        return value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return json.dumps(event, sort_keys=True)


def final_message_metadata(
    source: FinalMessageSource, status: str, *, message: str = "", stdout_tail_truncated: bool = False
) -> dict[str, Any]:
    return {
        "status": status,
        "source_type": source.source_type,
        "source_path": str(source.path) if source.path else None,
        "event_selector": source.event_selector,
        "tail_bytes": source.tail_bytes,
        "parser": source.parser,
        "parser_warnings": source.parser_warnings,
        "stdout_tail_truncated": stdout_tail_truncated,
        "message": message,
    }


def materialize_structured_event_message(config: AgentRunConfig, source: FinalMessageSource) -> str | None:
    selected: dict[str, Any] | None = None
    try:
        with config.agent_events_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event_matches_selector(event, source.event_selector):
                    selected = event
    except OSError:
        return None
    return event_message_text(selected) if selected else None


def materialize_final_message(
    config: AgentRunConfig,
    adapter: AgentAdapter,
    stdout_tail_lines: deque[str],
    *,
    stdout_tail_truncated: bool = False,
) -> None:
    source = adapter.final_message_source(config.result_dir)
    destination = config.agent_last_message_path
    if source.source_type == "file":
        if source.path and source.path.is_file():
            if source.path.resolve(strict=False) != destination.resolve(strict=False):
                shutil.copy2(source.path, destination)
            write_json(
                config.result_dir / "final_message_source.json",
                final_message_metadata(source, "materialized", stdout_tail_truncated=stdout_tail_truncated),
            )
            return
        destination.write_text("", encoding="utf-8")
        write_json(
            config.result_dir / "final_message_source.json",
            final_message_metadata(
                source,
                "missing",
                message="configured final message file was not written",
                stdout_tail_truncated=stdout_tail_truncated,
            ),
        )
        return
    if source.source_type == "stdout_tail":
        destination.write_text(raw_tail_text(stdout_tail_lines, source.tail_bytes), encoding="utf-8")
        write_json(
            config.result_dir / "final_message_source.json",
            final_message_metadata(source, "materialized", stdout_tail_truncated=stdout_tail_truncated),
        )
        return
    if source.source_type == "structured_event":
        if stdout_tail_truncated:
            destination.write_text("", encoding="utf-8")
            write_json(
                config.result_dir / "final_message_source.json",
                final_message_metadata(
                    source,
                    "missing",
                    message="stdout reader was still active; structured event final message was not read",
                    stdout_tail_truncated=stdout_tail_truncated,
                ),
            )
            return
        text = materialize_structured_event_message(config, source)
        destination.write_text(text or "", encoding="utf-8")
        write_json(
            config.result_dir / "final_message_source.json",
            final_message_metadata(
                source,
                "materialized" if text is not None else "missing",
                message="" if text is not None else "no matching structured event found",
                stdout_tail_truncated=stdout_tail_truncated,
            ),
        )
        return
    if source.source_type == "not_available":
        destination.write_text("", encoding="utf-8")
        write_json(
            config.result_dir / "final_message_source.json",
            final_message_metadata(source, "skipped", stdout_tail_truncated=stdout_tail_truncated),
        )
        return
    raise ValueError(f"Unsupported final message source_type: {source.source_type}")


def write_launch_spec_metadata(
    config: AgentRunConfig,
    launch_argv: list[str],
    launch_env: dict[str, str],
    launch: AgentLaunchSpec,
    skill_exposure: SkillExposureResult | None,
) -> None:
    write_json(
        config.result_dir / "launch_spec_metadata.json",
        {
            "agent": config.agent,
            "agent_model": config.agent_model,
            "agent_model_explicit": config.agent_model_was_explicit,
            "argv": launch_argv,
            "cwd": str(launch.cwd),
            "prompt_input_mode": launch.prompt_input_mode,
            "stdout_events_dest": str(launch.stdout_events_dest),
            "stderr_dest": str(launch.stderr_dest),
            "final_message_dest": str(launch.final_message_dest) if launch.final_message_dest else None,
            "login_shell": launch.login_shell,
            "approval_flags": launch.approval_flags,
            "sandbox_flags": launch.sandbox_flags,
            "bypass_reason": launch.bypass_reason,
            "launch_timeout": launch.launch_timeout,
            "environment_keys": sorted(launch_env),
            "skill_launch_args": skill_exposure.launch_args if skill_exposure else [],
            "skill_environment_keys": sorted(skill_exposure.environment) if skill_exposure else [],
        },
    )


def terminate_timed_out_process(process: subprocess.Popen, stderr, timeout: int | None) -> None:
    message = f"Agent command timed out after {timeout} seconds; terminating process.\n"
    stderr.write(message.encode("utf-8", errors="replace"))
    stderr.flush()
    process.terminate()
    try:
        process.wait(timeout=AGENT_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        stderr.write(b"Agent process did not terminate; killing process.\n")
        stderr.flush()
        process.kill()
        process.wait()


def _write_agent_home_diagnostic(config: AgentRunConfig, launch_env: dict) -> None:
    """Dump CLAUDE_CONFIG_DIR contents to a diagnostic file before launching the agent."""
    import os

    diag: dict = {}
    agent_home = launch_env.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR") or ""
    diag["claude_config_dir"] = agent_home
    if agent_home:
        home_path = Path(agent_home)
        try:
            diag["files"] = sorted(str(p.relative_to(home_path)) for p in home_path.rglob("*") if p.is_file())
        except Exception as e:
            diag["files_error"] = str(e)
        settings_path = home_path / "settings.json"
        try:
            diag["settings_json"] = json.loads(settings_path.read_text()) if settings_path.exists() else None
        except Exception as e:
            diag["settings_json_error"] = str(e)
        etc_managed = Path("/etc/claude-code/managed-settings.json")
        try:
            diag["etc_managed_settings"] = json.loads(etc_managed.read_text()) if etc_managed.exists() else None
        except Exception as e:
            diag["etc_managed_settings_error"] = str(e)
    config.result_dir.joinpath("agent_home_diagnostic.json").write_text(
        json.dumps(diag, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_agent(
    config: AgentRunConfig,
    progress: ProgressWriter,
    skill_exposure: SkillExposureResult | None = None,
    launch: AgentLaunchSpec | None = None,
) -> tuple[int, int, int]:
    start = epoch_seconds()
    config.progress_log_path.write_text("", encoding="utf-8")
    progress.write("agent_exec", "start", start)
    progress.start_heartbeat("agent_exec", config.progress_interval_seconds)
    agent_exit = 127
    launch_error: OSError | None = None
    stdout_tail_lines: deque[str] = deque(maxlen=MAX_STDOUT_TAIL_LINES)
    stdout_tail_lock = threading.Lock()
    adapter = load_agent_adapter(config.agent)
    launch = launch or build_agent_launch(config)
    launch_argv = list(launch.argv)
    launch_env = dict(launch.environment)
    if config.use_preinstalled_skills and skill_exposure is not None:
        launch_argv.extend(skill_exposure.launch_args)
        launch_env.update(skill_exposure.environment)
    process_argv = launch_subprocess_argv(launch_argv, login_shell=launch.login_shell)
    write_launch_spec_metadata(config, process_argv, launch_env, launch, skill_exposure)
    _write_agent_home_diagnostic(config, launch_env)

    try:
        try:
            prompt_stdin_context = (
                launch.prompt_file.open("rb") if launch.prompt_input_mode == "stdin" else nullcontext(None)
            )
            with (
                prompt_stdin_context as prompt_stdin,
                launch.stdout_events_dest.open("w", encoding="utf-8") as events_out,
                launch.stderr_dest.open("wb") as stderr,
            ):
                stdin = prompt_stdin if launch.prompt_input_mode == "stdin" else subprocess.DEVNULL
                try:
                    process = subprocess.Popen(
                        process_argv,
                        cwd=launch.cwd,
                        stdin=stdin,
                        stdout=subprocess.PIPE,
                        stderr=stderr,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=agent_subprocess_env(launch_env, adapter),
                    )
                except OSError as exc:
                    launch_error = exc
                    message = f"Failed to start agent command: {type(exc).__name__}: {exc}\n"
                    stderr.write(message.encode("utf-8", errors="replace"))
                    print(message.rstrip(), file=sys.stderr)
                else:
                    if process.stdout is None:
                        raise RuntimeError("Agent stdout pipe was not created")
                    reader_error: list[BaseException] = []
                    reader_stop = threading.Event()
                    events_write_lock = threading.Lock()

                    def stream_stdout() -> None:
                        try:
                            for line in process.stdout:
                                if reader_stop.is_set():
                                    break
                                with stdout_tail_lock:
                                    stdout_tail_lines.append(truncate_stdout_tail_line(line))
                                event = adapter.normalize_event(line)
                                normalized = (
                                    json.dumps(event, sort_keys=True, separators=(",", ":"))
                                    if event is not None
                                    else None
                                )
                                if normalized:
                                    with events_write_lock:
                                        if reader_stop.is_set():
                                            break
                                        events_out.write(normalized + "\n")
                                        events_out.flush()
                        except BaseException as exc:
                            if not reader_stop.is_set():
                                reader_error.append(exc)

                    reader = threading.Thread(target=stream_stdout, daemon=True)
                    reader.start()
                    try:
                        agent_exit = process.wait(timeout=launch.launch_timeout)
                    except subprocess.TimeoutExpired:
                        terminate_timed_out_process(process, stderr, launch.launch_timeout)
                        agent_exit = AGENT_TIMEOUT_EXIT_CODE
                    reader.join(timeout=AGENT_TERMINATE_GRACE_SECONDS)
                    stdout_tail_truncated = reader.is_alive()
                    if stdout_tail_truncated:
                        reader_stop.set()
                        with suppress(OSError, ValueError):
                            process.stdout.close()
                        reader.join(timeout=AGENT_TERMINATE_GRACE_SECONDS)
                        # Closing stdout can make the reader observe ValueError; reader_stop makes that non-fatal.
                        with events_write_lock:
                            pass
                        stderr.write(
                            "Agent stdout reader did not finish within "
                            f"{AGENT_TERMINATE_GRACE_SECONDS} seconds after process exit.\n".encode("utf-8")
                        )
                    with stdout_tail_lock:
                        stdout_tail_snapshot = deque(stdout_tail_lines, maxlen=stdout_tail_lines.maxlen)
                    materialize_final_message(
                        config,
                        adapter,
                        stdout_tail_snapshot,
                        stdout_tail_truncated=stdout_tail_truncated,
                    )
                    if reader_error:
                        raise RuntimeError(f"Failed to read agent stdout: {reader_error[0]}")
        except OSError as exc:
            launch_error = exc
            print(f"Failed to prepare agent command streams: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        progress.stop_heartbeat()

    end = epoch_seconds()
    progress.write("agent_exec", "failed_to_start" if launch_error is not None else "finished", end)
    if launch_error is not None and not config.agent_last_message_path.exists():
        config.agent_last_message_path.write_text("", encoding="utf-8")
    return start, end, agent_exit


def post_process(
    config: AgentRunConfig, elapsed_seconds: int, agent_exit: int, run_start_time_ns: int
) -> tuple[int, int]:
    start = epoch_seconds()
    input_delta_manifest = config.result_dir / "input_delta_manifest.json"
    capture_workspace_delta(
        config.run_input_dir,
        config.result_dir / "input_baseline_manifest.json",
        config.result_dir / "input_delta",
        input_delta_manifest,
        RUNTIME_ARTIFACT_ROOT,
        delta_scope="input_snapshot",
        include_runtime_artifacts=False,
    )
    workspace_delta_manifest = config.result_dir / "workspace_delta_manifest.json"
    capture_workspace_delta(
        config.run_workspace_dir,
        config.result_dir / "workspace_baseline_manifest.json",
        config.result_dir / "workspace_delta",
        workspace_delta_manifest,
        RUNTIME_ARTIFACT_ROOT,
        delta_scope="agent_workspace",
        extra_runtime_artifact_sources=[("runtime_workspaces", NVFLARE_RUNTIME_WORKSPACES_ROOT)],
    )
    adapter = load_agent_adapter(config.agent)
    exit_summary = adapter.exit_summary(
        agent_exit,
        config.agent_stderr_path,
        evidence_paths=(config.agent_last_message_path, config.agent_events_path),
    )
    write_json(config.agent_usage_path, adapter.parse_usage(config.agent_events_path))
    write_json(config.agent_activity_path, adapter.parse_activity(config.agent_events_path))
    write_json(config.result_dir / "agent_exit_summary.json", exit_summary)
    write_agent_compatibility_copies(config)
    synthesize_agent_record(
        AgentRecordSynthesisInputs(
            agent_record_path=config.agent_record_path,
            records_dir=config.records_dir,
            events_path=config.agent_events_path,
            usage_path=config.agent_usage_path,
            activity_path=config.agent_activity_path,
            last_message_path=config.agent_last_message_path,
            input_dir=config.run_input_dir,
            mode=config.mode,
            elapsed_seconds=elapsed_seconds,
            agent_exit=agent_exit,
            skills_enabled=config.use_preinstalled_skills,
            skill_run_mode=config.skill_run_mode,
            agent=config.agent,
            agent_model=config.agent_model,
            run_start_time_ns=run_start_time_ns,
            workspace_delta_manifest_path=workspace_delta_manifest,
            input_delta_manifest_path=input_delta_manifest,
            prompt_path=config.prompt_file_path,
        )
    )
    merge_record(
        agent_record_path=config.agent_record_path,
        final_record_path=config.final_record_path,
        usage_path=config.agent_usage_path,
        mode=config.mode,
        elapsed_seconds=elapsed_seconds,
        agent_exit=agent_exit,
        skills_enabled=config.use_preinstalled_skills,
        skill_run_mode=config.skill_run_mode,
        agent=config.agent,
        agent_model=config.agent_model,
    )
    record = load_json(config.final_record_path, {}) or {}
    if isinstance(record, dict):
        record["agent_exit_summary"] = exit_summary
        if exit_summary.get("failure_category"):
            record["failure_category"] = exit_summary.get("failure_category")
            record["failure_root_cause"] = exit_summary.get("failure_category")
        metrics = record.setdefault("process_metrics", {})
        if isinstance(metrics, dict):
            metrics["agent_exit_classifier"] = exit_summary.get("classifier")
        write_json(config.final_record_path, record)
    write_run_summary(config.final_record_path, config.result_dir / "run_summary.json", print_summary=False)
    return start, epoch_seconds()


def report_exit_status(command_statuses: dict[str, int]) -> int:
    return 1 if any(status != 0 for status in command_statuses.values()) else 0


def record_has_policy_failure(record: dict[str, Any]) -> bool:
    metrics = record.get("process_metrics")
    if isinstance(metrics, dict) and metrics.get("source_input_immutable_violation"):
        return True
    violation = record.get("source_input_immutable_violation")
    return isinstance(violation, dict) and violation.get("status") == "fail"


def final_container_exit_status(agent_exit: int, report_statuses: dict[str, int], *, policy_failed: bool) -> int:
    if agent_exit != 0:
        return agent_exit
    if policy_failed:
        return 1
    return report_exit_status(report_statuses)


def write_report_outcome(config: AgentRunConfig, agent_exit: int, report_statuses: dict[str, int]) -> int:
    report_exit = report_exit_status(report_statuses)
    record = load_json(config.final_record_path, {}) or {}
    if not isinstance(record, dict):
        record = {}
    policy_failed = record_has_policy_failure(record)
    final_exit = final_container_exit_status(agent_exit, report_statuses, policy_failed=policy_failed)
    record["agent_report_exit_codes"] = report_statuses
    record["agent_report_exit_code"] = report_exit
    record["agent_report_failed"] = report_exit != 0
    record["harness_policy_failed"] = policy_failed
    record["final_container_exit_code"] = final_exit
    record["report_inclusive_exit_code"] = final_exit
    metrics = record.setdefault("process_metrics", {})
    if isinstance(metrics, dict):
        metrics["agent_report_exit_code"] = report_exit
        metrics["agent_report_failed"] = 1 if report_exit else 0
        metrics["harness_policy_failed"] = 1 if policy_failed else 0
        metrics["final_container_exit_code"] = final_exit
        metrics["report_inclusive_exit_code"] = final_exit
    write_json(config.final_record_path, record)
    write_run_summary(config.final_record_path, config.result_dir / "run_summary.json", print_summary=False)
    return final_exit


def exit_code_from_exception(exc: BaseException, default: int = 1) -> int:
    if isinstance(exc, SystemExit) and isinstance(exc.code, int):
        return exc.code
    return default


def harness_error_payload(exc: BaseException, exit_code: int, phase: str) -> dict[str, Any]:
    return {
        "timestamp": utc_timestamp(),
        "phase": phase,
        "exit_code": exit_code,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def write_failure_record(
    *,
    result_dir: Path,
    records_dir: Path,
    mode: str,
    exit_code: int,
    error_type: str,
    message: str,
    phase: str,
    agent: str = "unknown",
    agent_model: str = "unspecified_default",
    skills_enabled: bool | None = None,
) -> int:
    result_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    error = {
        "timestamp": utc_timestamp(),
        "phase": phase,
        "exit_code": exit_code,
        "error_type": error_type,
        "message": message,
    }
    record = {
        "schema_version": "1",
        "mode": mode,
        "run_mode": "unknown" if skills_enabled is None else "with_skills" if skills_enabled else "without_skills",
        "agent": agent,
        "source": "agent_benchmark_harness",
        "agent_process_passed": False,
        "agent_process_exit_code": exit_code,
        "agent_elapsed_seconds": 0,
        "elapsed_seconds": 0,
        "agent_model": agent_model,
        "skills_enabled": skills_enabled,
        "timestamp": error["timestamp"],
        "agent_record_present": False,
        "agent_record_valid": False,
        "harness_failure": True,
        "harness_error": error,
        "harness_errors": [error],
        "agent_report_exit_codes": {},
        "agent_report_exit_code": 0,
        "agent_report_failed": False,
        "final_container_exit_code": exit_code,
        "report_inclusive_exit_code": exit_code,
        "process_metrics": {
            "elapsed_seconds": 0,
            "agent_elapsed_seconds": 0,
            "token_count": None,
            "agent_exit_code": exit_code,
            "agent_process_passed": 0,
            "harness_failure": 1,
            "agent_report_exit_code": 0,
            "agent_report_failed": 0,
            "final_container_exit_code": exit_code,
            "report_inclusive_exit_code": exit_code,
        },
    }
    final_record_path = records_dir / f"{mode}_record.json"
    write_json(final_record_path, record)
    write_json(
        result_dir / "early_failure.json",
        {
            **error,
            "record_path": str(final_record_path),
        },
    )
    write_run_summary(final_record_path, result_dir / "run_summary.json", print_summary=False)
    make_tree_readable(result_dir)
    return exit_code


def merge_harness_failure(config: AgentRunConfig, exc: BaseException, exit_code: int, phase: str) -> int | None:
    record = load_json(config.final_record_path, {}) or {}
    if not isinstance(record, dict) or not record:
        return None
    error = harness_error_payload(exc, exit_code, phase)
    record["harness_failure"] = True
    record["agent"] = record.get("agent") or config.agent
    record["agent_model"] = record.get("agent_model") or config.agent_model
    record["harness_error"] = error
    errors = record.get("harness_errors")
    if not isinstance(errors, list):
        errors = []
    errors.append(error)
    record["harness_errors"] = errors
    record["final_container_exit_code"] = exit_code
    record["harness_failure_exit_code"] = exit_code
    metrics = record.setdefault("process_metrics", {})
    if isinstance(metrics, dict):
        metrics["harness_failure"] = 1
        metrics["harness_failure_exit_code"] = exit_code
        metrics["final_container_exit_code"] = exit_code
    write_json(config.final_record_path, record)
    write_json(
        config.result_dir / "late_harness_failure.json",
        {
            **error,
            "record_path": str(config.final_record_path),
            "preserved_existing_record": True,
        },
    )
    write_run_summary(config.final_record_path, config.result_dir / "run_summary.json", print_summary=False)
    make_tree_readable(config.result_dir)
    return exit_code


def write_failure_record_from_env(exc: BaseException, exit_code: int, phase: str) -> int:
    env = os.environ
    result_dir = Path(env.get("RESULT_DIR", "/workspace/results"))
    records_dir = Path(env.get("RECORDS_DIR", str(result_dir / "records")))
    agent = env.get("BENCHMARK_AGENT", "unknown")
    try:
        agent_model = load_agent_adapter(agent).model_from_env(env)
    except Exception:
        agent_model = env.get("BENCHMARK_AGENT_MODEL", "unspecified_default")
    return write_failure_record(
        result_dir=result_dir,
        records_dir=records_dir,
        mode=env.get("MODE", "unknown"),
        exit_code=exit_code,
        error_type=type(exc).__name__,
        message=str(exc),
        phase=phase,
        agent=agent,
        agent_model=agent_model,
    )


def write_configured_failure(
    config: AgentRunConfig,
    exc: BaseException,
    exit_code: int,
    phase: str,
    *,
    preserve_existing_record: bool = False,
) -> int:
    if preserve_existing_record:
        merged_exit = merge_harness_failure(config, exc, exit_code, phase)
        if merged_exit is not None:
            return merged_exit
    return write_failure_record(
        result_dir=config.result_dir,
        records_dir=config.records_dir,
        mode=config.mode,
        exit_code=exit_code,
        error_type=type(exc).__name__,
        message=str(exc),
        phase=phase,
        agent=config.agent,
        agent_model=config.agent_model,
        skills_enabled=config.use_preinstalled_skills,
    )


def run_agent_benchmark() -> int:
    try:
        config = AgentRunConfig.from_env()
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        return write_failure_record_from_env(exc, exit_code_from_exception(exc, 2), "config")

    phase = "input_validation"
    normal_record_written = False
    try:
        if not config.prompt_source.is_file():
            message = (
                f"Prompt file is not mounted or does not exist: {config.prompt_source}. "
                "Mount a prompt file to /workspace/prompts/benchmark_prompt.txt or pass --prompt through the host wrapper scripts."
            )
            print(message, file=sys.stderr)
            return write_configured_failure(config, RuntimeError(message), 2, phase)
        if not config.job_input_dir.is_dir():
            message = f"Job input folder does not exist: {config.job_input_dir}"
            print(message, file=sys.stderr)
            return write_configured_failure(config, RuntimeError(message), 2, phase)

        config.result_dir.mkdir(parents=True, exist_ok=True)
        config.records_dir.mkdir(parents=True, exist_ok=True)
        config.run_root.mkdir(parents=True, exist_ok=True)
        phase = "runtime_metadata_probe"
        agent_launch = build_agent_launch(config)
        persist_container_runtime_metadata(config, agent_launch)
        phase = "agent_availability_probe"
        run_agent_availability_probe(config)

        script_start = epoch_seconds()
        script_start_ns = epoch_nanoseconds()
        phase = "skill_availability_setup"
        skill_start, skill_end, skill_exposure = setup_skill_availability(config)
        phase = "input_copy"
        input_start, input_end = prepare_input_workspace(config)
        write_workspace_baseline(config.run_input_dir, config.result_dir / "input_baseline_manifest.json")
        write_workspace_baseline(config.run_workspace_dir, config.result_dir / "workspace_baseline_manifest.json")
        phase = "prompt_prepare"
        prompt_start, prompt_end = prepare_prompt(config)

        phase = "agent_exec"
        progress = ProgressWriter(config.mode, script_start, config.progress_log_path)
        agent_start, agent_end, agent_exit = run_agent(config, progress, skill_exposure, agent_launch)
        elapsed_seconds = agent_end - agent_start
        phase = "post_process"
        post_start, post_end = post_process(config, elapsed_seconds, agent_exit, script_start_ns)
        normal_record_written = True
        phase = "report_outcome"
        report_start = epoch_seconds()
        report_statuses: dict[str, int] = {}
        final_exit = write_report_outcome(config, agent_exit, report_statuses)
        report_end = epoch_seconds()

        script_end = epoch_seconds()
        phase = "finalize_timing"
        finalize_timing(
            config.result_dir / "run_summary.json",
            config.final_record_path,
            config.result_dir / "timing.json",
            config.agent_activity_path,
            LifecycleEpochs(
                script_start=script_start,
                skill_availability_start=skill_start,
                skill_availability_end=skill_end,
                input_copy_start=input_start,
                input_copy_end=input_end,
                prompt_prep_start=prompt_start,
                prompt_prep_end=prompt_end,
                agent_start=agent_start,
                agent_end=agent_end,
                post_process_start=post_start,
                post_process_end=post_end,
                report_outcome_start=report_start,
                report_outcome_end=report_end,
                script_end=script_end,
            ),
        )
        print(
            "Benchmark run complete: "
            f"mode={config.mode}; elapsed_seconds={script_end - script_start}; "
            f"agent_elapsed_seconds={elapsed_seconds}; final_exit={final_exit}; result_dir={config.result_dir}",
            flush=True,
        )
        return final_exit
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        return write_configured_failure(
            config,
            exc,
            exit_code_from_exception(exc),
            phase,
            preserve_existing_record=normal_record_written,
        )
    finally:
        make_tree_readable(config.result_dir)


def main() -> None:
    raise SystemExit(run_agent_benchmark())


if __name__ == "__main__":
    main()
