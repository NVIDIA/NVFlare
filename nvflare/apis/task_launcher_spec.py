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
"""Common contracts for launching disposable task execution units."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

from nvflare.apis.fl_component import FLComponent


class TaskLauncherError(RuntimeError):
    """Base error raised by a task launcher."""


class UnsupportedTaskResourceError(TaskLauncherError):
    """A launcher cannot provide admission and reservation for requested resources."""


class TaskSettlementError(TaskLauncherError):
    """A launcher could not confirm that an execution unit released its resources."""

    def __init__(self, message: str, status: "TaskExecutionStatus"):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class TaskResourceRequest:
    """Resources that must be admitted and reserved for one task attempt.

    ``None`` means that the caller did not request admission for that resource.
    A zero GPU count explicitly requests no GPUs and is therefore also empty.
    """

    cpu_cores: Optional[float] = None
    memory_mb: Optional[int] = None
    gpu_count: int = 0

    def __post_init__(self):
        if self.cpu_cores is not None and (
            isinstance(self.cpu_cores, bool)
            or not isinstance(self.cpu_cores, (int, float))
            or not math.isfinite(self.cpu_cores)
            or self.cpu_cores <= 0
        ):
            raise ValueError("cpu_cores must be a finite positive number or None")
        if self.memory_mb is not None and (
            isinstance(self.memory_mb, bool) or not isinstance(self.memory_mb, int) or self.memory_mb <= 0
        ):
            raise ValueError("memory_mb must be a positive integer or None")
        if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int) or self.gpu_count < 0:
            raise ValueError("gpu_count must be a non-negative integer")

    def is_empty(self) -> bool:
        return self.cpu_cores is None and self.memory_mb is None and self.gpu_count == 0


@dataclass(frozen=True)
class TaskLaunchRequest:
    """Backend-independent request for one physical task attempt.

    ``environment`` is the complete environment passed to the execution unit;
    launchers do not implicitly copy the parent process environment. This
    keeps credential and policy decisions with the supervising runtime.
    """

    job_id: str
    site_name: str
    task_id: str
    attempt_id: str
    argv: tuple[str, ...]
    environment: Mapping[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    resources: TaskResourceRequest = field(default_factory=TaskResourceRequest)

    def __post_init__(self):
        for name in ("job_id", "site_name", "task_id", "attempt_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{name} must be a non-empty string without NUL")

        if isinstance(self.argv, (str, bytes, bytearray)):
            raise ValueError("argv must be a non-empty sequence of strings, not a string or bytes")
        argv = tuple(self.argv)
        if not argv or any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
            raise ValueError("argv must be a non-empty sequence of strings without NUL")
        object.__setattr__(self, "argv", argv)

        environment = dict(self.environment)
        for name, value in environment.items():
            if not isinstance(name, str) or not name or "=" in name or "\x00" in name:
                raise ValueError("environment names must be non-empty strings without '=' or NUL")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("environment values must be strings without NUL")
        object.__setattr__(self, "environment", environment)

        if self.cwd is not None and (not isinstance(self.cwd, str) or not self.cwd or "\x00" in self.cwd):
            raise ValueError("cwd must be a non-empty string without NUL or None")
        if not isinstance(self.resources, TaskResourceRequest):
            raise TypeError("resources must be a TaskResourceRequest")

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """Logical identity that must not be reused for another physical launch."""
        return self.job_id, self.site_name, self.task_id, self.attempt_id


class TaskExecutionPhase(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class TaskExecutionStatus:
    """Observed execution facts; settlement is independent of leader exit."""

    phase: TaskExecutionPhase
    exit_code: Optional[int] = None
    termination_signal: Optional[int] = None
    cancel_requested: bool = False
    settled: bool = False
    failure_reason: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return (
            self.phase == TaskExecutionPhase.TERMINAL
            and self.exit_code == 0
            and self.termination_signal is None
            and not self.cancel_requested
            and self.settled
            and self.failure_reason is None
        )


class TaskHandleSpec(ABC):
    """Handle for one launched task execution unit."""

    @property
    @abstractmethod
    def request(self) -> TaskLaunchRequest:
        raise NotImplementedError()

    @property
    @abstractmethod
    def execution_id(self) -> str:
        """Backend physical identity used to correlate status and diagnostics."""
        raise NotImplementedError()

    @abstractmethod
    def poll(self) -> TaskExecutionStatus:
        """Return a non-blocking observation of execution and settlement."""
        raise NotImplementedError()

    @abstractmethod
    def cancel(self) -> TaskExecutionStatus:
        """Cancel the whole execution unit and return its resulting status."""
        raise NotImplementedError()

    @abstractmethod
    def wait_for_settlement(self, timeout: Optional[float] = None) -> TaskExecutionStatus:
        """Wait for terminal execution and confirmed resource settlement.

        Raises:
            TimeoutError: If the execution does not settle within ``timeout``.
            TaskSettlementError: If cleanup was attempted but settlement could
                not be confirmed.
        """
        raise NotImplementedError()


class TaskLauncherSpec(FLComponent, ABC):
    """Launch backend for disposable task execution units."""

    launch_mode = None

    @abstractmethod
    def launch_task(self, request: TaskLaunchRequest) -> TaskHandleSpec:
        raise NotImplementedError()
