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

"""Shared host-side Docker benchmark helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..agents.base import AgentAdapter, DockerMount
from ..agents.registry import DEFAULT_BENCHMARK_AGENT, load_agent_adapter, validate_benchmark_agent
from ..common import write_json

SCRIPT_DIR = Path(__file__).resolve().parents[3]
PROMPT_FILE_NAME = "benchmark_prompt.txt"
CONTAINER_PROMPT_DIR = "/workspace/prompts"
CONTAINER_PROMPT_PATH = f"{CONTAINER_PROMPT_DIR}/{PROMPT_FILE_NAME}"
OUTPUT_LOCK = threading.Lock()
__all__ = [
    "SCRIPT_DIR",
    "PROMPT_FILE_NAME",
    "CONTAINER_PROMPT_DIR",
    "CONTAINER_PROMPT_PATH",
    "OUTPUT_LOCK",
    "HostCliOptions",
    "ImageConfig",
    "CaseConfig",
    "absolute_path",
    "add_agent_auth_mounts",
    "add_agent_passthrough_env",
    "benchmark_agent_adapter_from_env",
    "benchmark_agent_from_env",
    "case_config",
    "command_stdout_to_file",
    "default_results_root",
    "docker_args_for_case",
    "docker_env",
    "emit",
    "env_bool",
    "expand_home_path",
    "optional_int_env",
    "parse_host_cli_options",
    "print_usage",
    "stream_command",
    "timestamp_slug",
    "write_runtime_image",
]


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def expand_home_path(value: str) -> str:
    return str(Path(value).expanduser())


def absolute_path(value: str) -> Path:
    expanded = Path(expand_home_path(value))
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def env_bool(name: str, default: str) -> bool:
    value = os.environ.get(name, default)
    if value not in {"true", "false"}:
        raise SystemExit(f"{name} must be true or false; got {value}")
    return value == "true"


def default_results_root() -> Path:
    return Path(
        os.environ.get(
            "AGENT_BENCHMARK_RESULTS_ROOT",
            os.environ.get("CODEX_DOCKER_RESULTS_ROOT", str(SCRIPT_DIR / "results")),
        )
    )


def print_usage(command: str) -> None:
    usage = {
        "run-one": "Run one agent benchmark case against an arbitrary job folder.",
        "pair": "Run paired skills/no-skills benchmark cases against a job folder.",
        "interactive": "Start an interactive benchmark container with a job folder mounted.",
    }.get(command, "Run an agent benchmark command against a job folder.")
    print(
        f"Usage: {Path(sys.argv[0]).name} --prompt PATH [--training-code PATH] [--results-root PATH] [PATH]\n\n"
        f"{usage}\n\n"
        "Arguments:\n"
        "  PATH                    Job folder. Equivalent to --training-code.\n\n"
        "Options:\n"
        "  --prompt PATH           Prompt file to mount as the measured agent input.\n"
        "  --training-code PATH    Job folder to mount into the benchmark container.\n"
        "  --agent NAME            Agent profile to run. Defaults to codex.\n"
        "  --model NAME            Agent model to run. Required by agents without a default.\n"
        "  --mode NAME             run-one mode: with_skills or without_skills.\n"
        "  --workflow NAME         Workflow label for pair/run-one records. Defaults to default.\n"
        "  --job-scale NAME        Job size policy for pair/run-one. small, medium, or large.\n"
        "  --agent-home PATH       Host auth/config directory for the selected agent.\n"
        "  --no-agent-auth-mount   Do not mount host agent auth/config files.\n"
        "  --results-root PATH     Parent directory for generated timestamped result directories.\n"
        "  --output-dir PATH       Exact result directory for this run or comparison.\n"
        "  --result-root PATH      Exact result directory for pair comparisons.\n"
        "  --result-dir PATH       Exact result directory for run-one.\n"
        "  -h, --help              Show this help."
    )


@dataclass(frozen=True)
class HostCliOptions:
    job_input: Path
    prompt_path: Path
    results_root: Path | None = None
    result_root: Path | None = None
    result_dir: Path | None = None
    agent: str | None = None
    model: str | None = None
    mode: str | None = None
    workflow: str | None = None
    job_scale: str | None = None
    agent_home: Path | None = None
    mount_agent_auth: bool | None = None


def _option_value(argv: list[str], index: int, option: str) -> tuple[str, int]:
    arg = argv[index]
    if arg.startswith(f"{option}="):
        return arg.split("=", 1)[1], index + 1
    if index + 1 >= len(argv):
        raise SystemExit(f"{option} requires a path")
    return argv[index + 1], index + 2


def parse_host_cli_options(argv: list[str], command: str) -> HostCliOptions:
    job_input = os.environ.get("JOB_INPUT_DIR") or os.environ.get("TRAINING_CODE") or ""
    prompt_input = ""
    set_by_arg = False
    results_root: Path | None = None
    result_root: Path | None = None
    result_dir: Path | None = None
    agent: str | None = None
    model: str | None = None
    mode: str | None = None
    workflow: str | None = None
    job_scale: str | None = None
    agent_home: Path | None = None
    mount_agent_auth: bool | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--training-code" or arg.startswith("--training-code="):
            value, index = _option_value(argv, index, "--training-code")
            if set_by_arg:
                raise SystemExit("Expected only one job folder")
            job_input = value
            set_by_arg = True
        elif arg == "--prompt" or arg.startswith("--prompt="):
            value, index = _option_value(argv, index, "--prompt")
            if prompt_input:
                raise SystemExit("Expected only one --prompt")
            prompt_input = value
        elif arg == "--agent" or arg.startswith("--agent="):
            value, index = _option_value(argv, index, "--agent")
            if agent is not None:
                raise SystemExit("Expected only one --agent")
            agent = value
        elif arg == "--model" or arg.startswith("--model="):
            value, index = _option_value(argv, index, "--model")
            if model is not None:
                raise SystemExit("Expected only one --model")
            model = value
        elif arg == "--mode" or arg.startswith("--mode="):
            value, index = _option_value(argv, index, "--mode")
            if command != "run-one":
                raise SystemExit("--mode is only supported for run-one; use pair or scenario for comparisons.")
            if mode is not None:
                raise SystemExit("Expected only one --mode")
            mode = value
        elif arg == "--workflow" or arg.startswith("--workflow="):
            value, index = _option_value(argv, index, "--workflow")
            if workflow is not None:
                raise SystemExit("Expected only one --workflow")
            workflow = value
        elif arg == "--job-scale" or arg.startswith("--job-scale="):
            value, index = _option_value(argv, index, "--job-scale")
            if job_scale is not None:
                raise SystemExit("Expected only one --job-scale")
            job_scale = value
        elif arg == "--agent-home" or arg.startswith("--agent-home="):
            value, index = _option_value(argv, index, "--agent-home")
            if agent_home is not None:
                raise SystemExit("Expected only one --agent-home")
            agent_home = absolute_path(value)
        elif arg == "--no-agent-auth-mount":
            if mount_agent_auth is False:
                raise SystemExit("Expected only one --no-agent-auth-mount")
            mount_agent_auth = False
            index += 1
        elif arg == "--results-root" or arg.startswith("--results-root="):
            value, index = _option_value(argv, index, "--results-root")
            if results_root is not None:
                raise SystemExit("Expected only one --results-root")
            results_root = absolute_path(value)
        elif arg == "--result-root" or arg.startswith("--result-root="):
            value, index = _option_value(argv, index, "--result-root")
            if result_root is not None:
                raise SystemExit("Expected only one --result-root")
            result_root = absolute_path(value)
        elif arg == "--result-dir" or arg.startswith("--result-dir="):
            value, index = _option_value(argv, index, "--result-dir")
            if result_dir is not None:
                raise SystemExit("Expected only one --result-dir")
            result_dir = absolute_path(value)
        elif arg == "--output-dir" or arg.startswith("--output-dir="):
            value, index = _option_value(argv, index, "--output-dir")
            if command == "run-one":
                if result_dir is not None:
                    raise SystemExit("Expected only one exact output directory")
                result_dir = absolute_path(value)
            elif command == "pair":
                if result_root is not None:
                    raise SystemExit("Expected only one exact output directory")
                result_root = absolute_path(value)
            else:
                raise SystemExit(f"--output-dir is not supported for {command}")
        elif arg in {"-h", "--help"}:
            print_usage(command)
            raise SystemExit(0)
        elif arg == "--":
            rest = argv[index + 1 :]
            if len(rest) > 1:
                raise SystemExit("Expected at most one job folder after --")
            if rest:
                if set_by_arg:
                    raise SystemExit("Expected only one job folder")
                job_input = rest[0]
            break
        elif arg.startswith("-"):
            print_usage(command)
            raise SystemExit(f"Unknown option: {arg}")
        else:
            if set_by_arg:
                raise SystemExit("Expected only one job folder")
            job_input = arg
            set_by_arg = True
            index += 1

    if not job_input:
        print_usage(command)
        raise SystemExit("Job input folder is required. Pass PATH or --training-code PATH.")
    if not prompt_input:
        print_usage(command)
        raise SystemExit("Prompt file is required. Pass --prompt PATH.")
    for name, value in (
        ("--agent", agent),
        ("--model", model),
        ("--mode", mode),
        ("--workflow", workflow),
        ("--job-scale", job_scale),
    ):
        if value is not None and not value.strip():
            raise SystemExit(f"{name} requires a non-empty value")
    path = absolute_path(job_input)
    if not path.is_dir():
        raise SystemExit(f"Job input must be an existing folder: {path}")
    prompt_path = absolute_path(prompt_input)
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file must be an existing file: {prompt_path}")
    if results_root is not None and (result_root is not None or result_dir is not None):
        raise SystemExit("Use --results-root or an exact output option, not both.")
    if command == "run-one" and result_root is not None:
        raise SystemExit("--result-root is only supported for pair; use --result-dir for run-one.")
    if command == "pair" and result_dir is not None:
        raise SystemExit("--result-dir is only supported for run-one; use --result-root for comparisons.")
    if command == "interactive" and (results_root is not None or result_root is not None or result_dir is not None):
        raise SystemExit("Output directory options are not supported for interactive containers.")
    return HostCliOptions(
        job_input=path,
        prompt_path=prompt_path,
        results_root=results_root,
        result_root=result_root,
        result_dir=result_dir,
        agent=agent,
        model=model,
        mode=mode,
        workflow=workflow,
        job_scale=job_scale,
        agent_home=agent_home,
        mount_agent_auth=mount_agent_auth,
    )


def emit(message: str = "", *, logs: Iterable[Path] = (), prefix: str | None = None, stderr: bool = False) -> None:
    line = f"[{prefix}] {message}" if prefix else message
    stream = sys.stderr if stderr else sys.stdout
    with OUTPUT_LOCK:
        print(line, file=stream, flush=True)
        for log in logs:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def stream_command(
    command: list[str],
    *,
    logs: Iterable[Path] = (),
    prefix: str | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> int:
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except OSError as exc:
        program = command[0] if command else "<empty command>"
        emit(f"Failed to start command {program}: {type(exc).__name__}: {exc}", logs=logs, prefix=prefix, stderr=True)
        return 127
    if process.stdout is None:
        raise RuntimeError("subprocess stdout pipe was not created")
    reader_errors: list[BaseException] = []

    def _read_stdout() -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                emit(line.rstrip("\n"), logs=logs, prefix=prefix)
        except ValueError:
            pass
        except BaseException as exc:
            reader_errors.append(exc)

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()
    timed_out = False
    try:
        status = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        emit(
            f"Command timed out after {timeout_seconds} seconds; terminating process.",
            logs=logs,
            prefix=prefix,
            stderr=True,
        )
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            emit("Process did not terminate; killing process.", logs=logs, prefix=prefix, stderr=True)
            process.kill()
            process.wait()
        status = 124
    finally:
        if process.stdout is not None:
            with suppress(OSError, ValueError):
                process.stdout.close()
        reader.join(timeout=2)
    if reader.is_alive():
        emit("Output reader thread did not stop within 2 seconds.", logs=logs, prefix=prefix, stderr=True)
    for error in reader_errors:
        emit(f"Output reader error: {type(error).__name__}: {error}", logs=logs, prefix=prefix, stderr=True)
    return 124 if timed_out else status


def command_stdout_to_file(
    command: list[str],
    output_path: Path,
    *,
    logs: Iterable[Path] = (),
    prefix: str | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8", errors="replace") as stdout:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if process.stderr is None:
                raise RuntimeError("subprocess stderr pipe was not created")
            for line in process.stderr:
                emit(line.rstrip("\n"), logs=logs, prefix=prefix, stderr=True)
            return process.wait()
    except OSError as exc:
        program = command[0] if command else "<empty command>"
        emit(f"Failed to start command {program}: {type(exc).__name__}: {exc}", logs=logs, prefix=prefix, stderr=True)
        return 127


def add_agent_passthrough_env(args: list[str], adapter: AgentAdapter) -> None:
    for name in adapter.passthrough_env_names():
        if os.environ.get(name):
            args.extend(["-e", name])


def docker_env(name: str, value: str | int | bool | None = None) -> list[str]:
    if value is None:
        return ["-e", name]
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    return ["-e", f"{name}={rendered}"]


def add_agent_auth_mounts(
    args: list[str],
    *,
    mounts: list[DockerMount],
    logs: Iterable[Path] = (),
    prefix: str | None = None,
) -> None:
    for mount in mounts:
        if mount.host_path.is_file():
            suffix = ":ro" if mount.read_only else ""
            args.extend(["-v", f"{mount.host_path}:{mount.container_path}{suffix}"])
            label = mount.description or "agent auth/config"
            emit(f"Mounting {label}: {mount.host_path} -> {mount.container_path}", logs=logs, prefix=prefix)
        elif mount.required:
            raise SystemExit(f"Required agent auth/config file is missing: {mount.host_path}")
        else:
            label = mount.description or "agent auth/config"
            emit(f"{label} not mounted; missing {mount.host_path}", logs=logs, prefix=prefix, stderr=True)


@dataclass(frozen=True)
class ImageConfig:
    image_name: str
    baseline_image_name: str
    report_image_name: str

    @classmethod
    def from_env(cls) -> "ImageConfig":
        adapter = benchmark_agent_adapter_from_env()
        return cls.for_adapter(adapter)

    @classmethod
    def for_adapter(cls, adapter: AgentAdapter) -> "ImageConfig":
        targets = adapter.image_targets()
        image = targets.skills
        return cls(
            image_name=image,
            baseline_image_name=targets.baseline,
            report_image_name=targets.report,
        )


@dataclass(frozen=True)
class CaseConfig:
    mode: str
    use_preinstalled_skills: bool
    job_input_dir: Path
    result_dir: Path
    prompt_path: Path
    images: ImageConfig
    progress_interval_seconds: str
    agent: str
    agent_model: str
    model_was_explicit: bool
    adapter: AgentAdapter
    host_agent_home: Path
    mount_host_agent_auth: bool
    agent_timeout_seconds: int | None = None
    container_timeout_seconds: int | None = None
    result_size_budget_bytes: int | None = None

    @property
    def run_image(self) -> str:
        return self.images.image_name if self.use_preinstalled_skills else self.images.baseline_image_name

    @property
    def sdk_image_kind(self) -> str:
        return (
            "local_wheel_with_preinstalled_skills"
            if self.use_preinstalled_skills
            else "local_wheel_without_packaged_skills"
        )


def case_config(
    *,
    mode: str,
    use_preinstalled_skills: bool,
    job_input_dir: Path,
    result_dir: Path,
    prompt_path: Path,
    images: ImageConfig,
) -> CaseConfig:
    adapter = benchmark_agent_adapter_from_env()
    try:
        agent_model = adapter.model_from_env(os.environ)
        model_was_explicit = adapter.model_was_explicit(os.environ)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return CaseConfig(
        mode=mode,
        use_preinstalled_skills=use_preinstalled_skills,
        job_input_dir=job_input_dir,
        result_dir=result_dir,
        prompt_path=prompt_path,
        images=images,
        progress_interval_seconds=os.environ.get("PROGRESS_INTERVAL_SECONDS", "60"),
        agent=adapter.name,
        agent_model=agent_model,
        model_was_explicit=model_was_explicit,
        adapter=adapter,
        host_agent_home=absolute_path(str(adapter.host_home_from_env(os.environ))),
        mount_host_agent_auth=adapter.mount_auth_from_env(os.environ),
        agent_timeout_seconds=optional_int_env("AGENT_TIMEOUT_SECONDS"),
        container_timeout_seconds=optional_int_env("CONTAINER_TIMEOUT_SECONDS"),
        result_size_budget_bytes=optional_int_env("RESULT_SIZE_BUDGET_BYTES"),
    )


def optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer; got {value}") from exc
    if parsed <= 0:
        raise SystemExit(f"{name} must be positive; got {value}")
    return parsed


def docker_args_for_case(config: CaseConfig, logs: Iterable[Path] = (), prefix: str | None = None) -> list[str]:
    runtime_env = config.adapter.runtime_env(config)
    args = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{config.job_input_dir}:/workspace/input:ro",
        "-v",
        f"{config.result_dir}:/workspace/results",
        "-v",
        f"{config.prompt_path}:{CONTAINER_PROMPT_PATH}:ro",
        *docker_env("JOB_INPUT_DIR", "/workspace/input"),
        *docker_env("TRAINING_CODE", "/workspace/input"),
        *docker_env("PROMPT_SOURCE", CONTAINER_PROMPT_PATH),
        *docker_env("MODE", config.mode),
        *docker_env("USE_PREINSTALLED_SKILLS", config.use_preinstalled_skills),
        *docker_env("SDK_IMAGE_KIND", config.sdk_image_kind),
        *docker_env("PROGRESS_INTERVAL_SECONDS", config.progress_interval_seconds),
        *docker_env("RESULT_DIR", "/workspace/results"),
        *docker_env("RECORDS_DIR", "/workspace/results/records"),
        *docker_env("SDK_AGENT_RECORD", f"/workspace/results/records/{config.mode}_agent_record.json"),
    ]
    for name, value in sorted(runtime_env.items()):
        args.extend(docker_env(name, value))
    if config.agent_timeout_seconds is not None:
        args.extend(docker_env("AGENT_TIMEOUT_SECONDS", config.agent_timeout_seconds))
    add_agent_passthrough_env(args, config.adapter)
    if config.mount_host_agent_auth:
        add_agent_auth_mounts(args, mounts=config.adapter.auth_mounts(config), logs=logs, prefix=prefix)
    args.extend(
        [
            config.run_image,
            "/workspace/venv/bin/python",
            "-m",
            "assist_tools.skills_benchmark.skills.harness.container.agent_run",
        ]
    )
    return args


def write_runtime_image(config: CaseConfig) -> None:
    config.result_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        config.result_dir / "runtime_image.json",
        {
            "mode": config.mode,
            "use_preinstalled_skills": config.use_preinstalled_skills,
            "agent": config.agent,
            "agent_model": config.agent_model,
            "agent_model_explicit": config.model_was_explicit,
            "agent_adapter": config.adapter.metadata(),
            "runtime_image": config.run_image,
            "report_image": config.images.report_image_name,
            "sdk_image_kind": config.sdk_image_kind,
            "container_prompt_source": CONTAINER_PROMPT_PATH,
            "container_python": "/workspace/venv/bin/python",
            "container_virtual_env": "/workspace/venv",
            "resource_policy": {
                "agent_timeout_seconds": config.agent_timeout_seconds,
                "container_timeout_seconds": config.container_timeout_seconds,
                "result_size_budget_bytes": config.result_size_budget_bytes,
            },
        },
    )


def benchmark_agent_from_env() -> str:
    try:
        return validate_benchmark_agent(os.environ.get("BENCHMARK_AGENT", DEFAULT_BENCHMARK_AGENT))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def benchmark_agent_adapter_from_env() -> AgentAdapter:
    try:
        return load_agent_adapter(os.environ.get("BENCHMARK_AGENT", DEFAULT_BENCHMARK_AGENT))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
