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
"""Configuration, models, constants, and validation for Slurm launchers."""

from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Optional

from nvflare.app_opt.job_launcher.study_runtime import (
    SLURM_RESERVED_ENV_NAMES,
    SLURM_RESERVED_ENV_PREFIXES,
    SLURM_SANDBOXES,
)

CONTROL_DIR = ".nvflare_slurm"
SECRET_FILE = "secret.env"
BATCH_FILE = "batch.sh"
SANDBOX_ROOT = "sandbox_root"
SLURM_CHILD_PROCESS_ENV = "NVFLARE_SLURM_CHILD_PROCESS"
CONTAINER_RESOLV_CONF = "/etc/resolv.conf"

SQUEUE_FORMAT = "%i|%T|%U|%k|%j"
SACCT_FORMAT = "JobIDRaw%32,JobName%128,User%64,State%64,ExitCode%32"

_MAX_STDOUT_BYTES = 16 * 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_SUBMIT_ID_RE = re.compile(r"^([0-9]+)(?:;([A-Za-z0-9._-]+))?$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SLURM_SBATCH_DIRECTIVES = ("partition", "account", "qos", "time", "constraint", "reservation")
SLURM_PARENT_EXECUTABLES = ("sbatch", "squeue", "sacct", "scancel")
SLURM_COMPUTE_EXECUTABLES = ("apptainer", "srun")

_JOB_SLURM_KEYS = {"image", "nodes", "gpus_per_node", "cpus_per_node", "mem_per_node", "time", "pending_timeout"}

_PENDING_STATES = {"PENDING", "CONFIGURING", "REQUEUE_HOLD", "RESV_DEL_HOLD", "SPECIAL_EXIT"}
_APPLICATION_TERMINAL_STATES = {"COMPLETED", "FAILED"}
_INFRASTRUCTURE_TERMINAL_STATES = {
    "BOOT_FAIL",
    "DEADLINE",
    "LAUNCH_FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "RECONFIG_FAIL",
    "REVOKED",
    "TIMEOUT",
}


class SlurmLauncherError(RuntimeError):
    """Base error for a failed Slurm launch operation."""


class SlurmProtocolError(SlurmLauncherError):
    """Scheduler output contradicted the configured machine-readable protocol."""


class LookupStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SlurmConfig:
    workspace_path: str
    sandbox: str
    python_path: str
    executables: dict
    image: Optional[str] = None
    internal_port: int = 8102
    sbatch_directives: dict = field(default_factory=dict)
    setup: str = ""
    forward_env: tuple = ()
    parent_host: Optional[str] = None
    poll_interval: float = 10.0
    pending_timeout: float = 600.0


@dataclass(frozen=True)
class JobResources:
    nodes: int = 1
    gpus_per_node: Optional[int] = None
    cpus_per_node: Optional[int] = None
    mem_per_node: Optional[int] = None
    time_limit: Optional[str] = None
    pending_timeout: float = 600.0


@dataclass(frozen=True)
class BindMount:
    source: str
    destination: str
    mode: str

    def render(self) -> str:
        return f"{self.source}:{self.destination}:{self.mode}"


@dataclass(frozen=True)
class CommandResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def available(self) -> bool:
        return not self.timed_out and self.returncode == 0


@dataclass(frozen=True)
class SlurmRecord:
    job_id: str
    state: str
    exit_status: int = 0
    exit_signal: int = 0


@dataclass(frozen=True)
class QueryResult:
    status: LookupStatus
    records: tuple = ()


@dataclass(frozen=True)
class SubmissionResult:
    command: CommandResult
    job_id: Optional[str] = None
    cluster: Optional[str] = None


def _require_int(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SlurmLauncherError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _require_positive_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SlurmLauncherError(f"{name} must be a finite positive number")
    return float(value)


def _require_string(value: str, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise SlurmLauncherError(f"{name} must be {qualifier}")
    if "\x00" in value:
        raise SlurmLauncherError(f"{name} contains a forbidden NUL")
    return value


def _validate_env_name(name: str, label: str = "environment name") -> str:
    if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
        raise SlurmLauncherError(f"{label} must match [A-Za-z_][A-Za-z0-9_]*")
    if name in SLURM_RESERVED_ENV_NAMES or name.startswith(SLURM_RESERVED_ENV_PREFIXES):
        raise SlurmLauncherError(f"{label} '{name}' is launcher-owned")
    return name


def _validate_absolute_path(value: str, label: str) -> str:
    value = _require_string(value, label)
    if not os.path.isabs(value):
        raise SlurmLauncherError(f"{label} must be absolute")
    return value


def _paths_overlap(first: str, second: str) -> bool:
    first = os.path.normpath(first)
    second = os.path.normpath(second)
    try:
        return os.path.commonpath((first, second)) in (first, second)
    except ValueError:
        return False


def _validate_mount_operand(value: str, label: str) -> str:
    if ":" in value or "," in value:
        raise SlurmLauncherError(f"{label} contains ':' or ',', which is unsupported by Slurm container backends")
    return value


def _validate_mount_destination(value: str, label: str) -> str:
    value = _validate_mount_operand(_require_string(value, label), label)
    if value.startswith("//"):
        raise SlurmLauncherError(f"{label} must start with exactly one '/'")
    pure = PurePosixPath(value)
    if not pure.is_absolute() or ".." in pure.parts:
        raise SlurmLauncherError(f"{label} must be a normalized absolute POSIX path without '..'")
    normalized = str(pure)
    if normalized != value.rstrip("/") and not (value == "/" and normalized == "/"):
        raise SlurmLauncherError(f"{label} must be normalized")
    for forbidden in ("/", "/proc", "/sys", "/dev"):
        if normalized == forbidden or (forbidden != "/" and normalized.startswith(forbidden + "/")):
            raise SlurmLauncherError(f"{label} is under forbidden destination '{forbidden}'")
    return normalized


def _validate_mount_source(source: str, workspace: str, label: str) -> str:
    source = _validate_mount_operand(_validate_absolute_path(source, label), label)
    real = os.path.realpath(source)
    _validate_mount_operand(real, f"{label} canonical path")
    if _paths_overlap(real, workspace):
        raise SlurmLauncherError(f"{label} must be outside workspace_path")
    return real


def _mapping_or_empty(value, label: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SlurmLauncherError(f"{label} must be a mapping")
    return value


def normalize_slurm_directives(value: Optional[dict], label: str = "sbatch_directives") -> dict:
    value = _mapping_or_empty(value, label)
    unknown = set(value).difference(SLURM_SBATCH_DIRECTIVES)
    if unknown:
        raise SlurmLauncherError(f"unsupported {label}: {sorted(unknown)}")
    result = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise SlurmLauncherError(f"{label}.{key} must be a string or integer")
        text = _require_string(str(item), f"{label}.{key}")
        if any(character.isspace() for character in text):
            raise SlurmLauncherError(f"{label}.{key} must not contain whitespace")
        result[key] = text
    return result


def normalize_slurm_executables(value: dict) -> dict[str, Optional[str]]:
    if not isinstance(value, dict):
        raise SlurmLauncherError("executables must be a mapping")
    unknown = set(value).difference(SLURM_PARENT_EXECUTABLES, SLURM_COMPUTE_EXECUTABLES)
    if unknown:
        raise SlurmLauncherError(f"executables has unknown key(s): {sorted(unknown)}")
    result = {}
    for name in SLURM_PARENT_EXECUTABLES:
        path = value.get(name)
        result[name] = None if path is None else _validate_absolute_path(path, f"executables.{name}")
    for name in SLURM_COMPUTE_EXECUTABLES:
        path = value.get(name)
        if path is not None:
            path = _require_string(path, f"executables.{name}")
        result[name] = path
    return result


def resolve_slurm_parent_executables(value: dict) -> dict:
    """Resolve parent-side Slurm commands once on the host running the parent."""
    result = dict(value)
    for name in SLURM_PARENT_EXECUTABLES:
        configured = value.get(name)
        candidate = configured or shutil.which(name)
        if not candidate:
            raise SlurmLauncherError(f"required Slurm executable '{name}' was not found on the parent runtime PATH")
        if configured is not None:
            _validate_absolute_path(configured, f"executables.{name}")
        resolved = os.path.realpath(os.path.abspath(candidate))
        if not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
            raise SlurmLauncherError(
                f"required Slurm executable '{name}' is not an executable regular file on the parent runtime host: "
                f"{candidate}"
            )
        result[name] = resolved
    return result


def normalize_slurm_image(
    image: Optional[str], sandbox: str, *, require_file: bool = False, label: str = "image"
) -> Optional[str]:
    if sandbox not in SLURM_SANDBOXES:
        raise SlurmLauncherError(f"sandbox must be one of {sorted(SLURM_SANDBOXES)}")
    if image is None:
        if sandbox != "none":
            raise SlurmLauncherError(f"{label} is required for sandbox '{sandbox}'")
        return None
    image = _validate_absolute_path(image, label)
    if not require_file or sandbox == "none":
        return image
    real_path = os.path.realpath(image)
    if not os.path.isfile(real_path):
        raise SlurmLauncherError(f"{label} is not an existing regular file: {image}")
    return real_path


def normalize_slurm_workspace_path(value: str) -> str:
    value = os.path.normpath(_validate_absolute_path(value, "workspace_path"))
    if os.pathsep in value:
        raise SlurmLauncherError(f"workspace_path must not contain the path-list separator {os.pathsep!r}")
    if "," in value:
        raise SlurmLauncherError("workspace_path must not contain ','")
    return value


def normalize_slurm_launcher_settings(
    *,
    sandbox: str,
    python_path: str,
    executables: dict,
    image: Optional[str],
    internal_port: int,
    sbatch_directives: Optional[dict],
    setup: Optional[str],
    forward_env: Optional[list],
    parent_host: Optional[str],
    poll_interval: float,
    pending_timeout: float,
    require_image_file: bool = False,
) -> dict:
    sandbox = _require_string(sandbox, "sandbox")
    python_path = _validate_absolute_path(python_path, "python_path")
    internal_port = _require_int(internal_port, "internal_port")
    if internal_port > 65535:
        raise SlurmLauncherError("internal_port must be at most 65535")
    poll_interval = _require_positive_number(poll_interval, "poll_interval")
    pending_timeout = _require_positive_number(pending_timeout, "pending_timeout")
    setup = "" if setup is None else setup
    setup = _require_string(setup, "setup", allow_empty=True)
    forward_env = () if forward_env is None else forward_env
    if not isinstance(forward_env, (list, tuple)):
        raise SlurmLauncherError("forward_env must be a list")
    validated_forward = tuple(_validate_env_name(name, "forward_env entry") for name in forward_env)
    return {
        "sandbox": sandbox,
        "python_path": python_path,
        "executables": normalize_slurm_executables(executables),
        "image": normalize_slurm_image(image, sandbox, require_file=require_image_file),
        "internal_port": internal_port,
        "sbatch_directives": normalize_slurm_directives(sbatch_directives),
        "setup": setup,
        "forward_env": validated_forward,
        "parent_host": None if parent_host is None else _require_string(parent_host, "parent_host"),
        "poll_interval": poll_interval,
        "pending_timeout": pending_timeout,
    }


@dataclass(frozen=True)
class LaunchPlan:
    job_id: str
    run_dir: str
    exe_module: str
    module_args: tuple
    resources: JobResources
    directives: dict
    sandbox: str
    image: Optional[str]
    setup: str
    study_env: dict
    study_secret_env: dict
    mounts: tuple
    python_path: str
    python_env: str
    forward_env: tuple
