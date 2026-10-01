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
"""Local process-group backend for disposable task execution.

The owned execution scope is one POSIX session/process group. The command may
launch any number of ranks inside that group. A task worker must not deliberately
detach descendants into another session: a portable process launcher cannot
discover or prove settlement of processes that escape its owned group.
"""

import logging
import math
import os
import signal
import threading
import time
from dataclasses import replace
from typing import Optional

import psutil

from nvflare.apis.launcher import LauncherMode
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskHandleSpec,
    TaskLauncherError,
    TaskLauncherSpec,
    TaskLaunchRequest,
    TaskSettlementError,
    UnsupportedTaskResourceError,
)
from nvflare.utils.process_utils import ProcessAdapter, spawn_process


def _positive_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


class ProcessTaskHandle(TaskHandleSpec):
    """Own and observe one task process group."""

    def __init__(
        self,
        request: TaskLaunchRequest,
        adapter: ProcessAdapter,
        *,
        stop_grace_period: float,
        descendant_settle_timeout: float,
        poll_interval: float,
    ):
        self._request = request
        self._adapter = adapter
        self._process_group_id = adapter.pid
        self._execution_id = f"process:{adapter.pid}"
        self._stop_grace_period = _positive_number(stop_grace_period, "stop_grace_period")
        self._descendant_settle_timeout = _positive_number(descendant_settle_timeout, "descendant_settle_timeout")
        self._poll_interval = _positive_number(poll_interval, "poll_interval")
        self._cancel_requested = False
        self._failure_reason = None
        self._settled_status = None
        self._lock = threading.RLock()
        self._termination_lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def request(self) -> TaskLaunchRequest:
        return self._request

    @property
    def execution_id(self) -> str:
        return self._execution_id

    @property
    def process_group_id(self) -> int:
        return self._process_group_id

    def _group_exists(self) -> bool:
        try:
            os.killpg(self._process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Lack of probe permission is not evidence that resources settled.
            return True

        # killpg also sees unreaped zombies. They cannot execute or hold compute
        # resources and cannot be removed with signals. Ignore a group only
        # when all observed members are dead; uncertain observations fail closed.
        dead_member_seen = False
        try:
            for process in psutil.process_iter():
                try:
                    if os.getpgid(process.pid) != self._process_group_id:
                        continue
                    if process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                        return True
                    dead_member_seen = True
                except (ProcessLookupError, psutil.NoSuchProcess):
                    continue
                except (PermissionError, psutil.AccessDenied):
                    return True
        except (PermissionError, psutil.AccessDenied):
            return True
        if not dead_member_seen:
            # The last member can disappear between killpg and enumeration.
            # Confirm absence afresh instead of retaining the stale probe.
            try:
                os.killpg(self._process_group_id, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
        return not dead_member_seen

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(self._process_group_id, sig)
        except ProcessLookupError:
            return
        except PermissionError as e:
            self.logger.warning("cannot signal task process group %s: %s", self._process_group_id, e)

    def _status_unlocked(self) -> TaskExecutionStatus:
        if self._settled_status is not None:
            return self._settled_status
        return_code = self._adapter.poll()
        group_exists = return_code is None or self._group_exists()
        if return_code is None:
            phase = TaskExecutionPhase.RUNNING
            exit_code = None
            termination_signal = None
        else:
            phase = TaskExecutionPhase.TERMINAL
            exit_code = return_code if return_code >= 0 else None
            termination_signal = -return_code if return_code < 0 else None
        status = TaskExecutionStatus(
            phase=phase,
            exit_code=exit_code,
            termination_signal=termination_signal,
            cancel_requested=self._cancel_requested,
            settled=return_code is not None and not group_exists,
            failure_reason=self._failure_reason,
        )
        if status.settled:
            # Settlement is final for this physical execution. Never reopen
            # its scope because a later scan is uncertain or its PGID is reused.
            self._settled_status = status
        return status

    def poll(self) -> TaskExecutionStatus:
        with self._lock:
            return self._status_unlocked()

    def _wait_for_group_exit(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            # Reap the leader as part of each observation. A zombie group leader
            # must not keep an otherwise empty execution scope looking live.
            if self.poll().settled:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(self._poll_interval, remaining))

    def _terminate_group(self) -> TaskExecutionStatus:
        with self._termination_lock:
            status = self.poll()
            if status.settled:
                return status
            self._signal_group(signal.SIGTERM)
            if not self._wait_for_group_exit(self._stop_grace_period):
                self._signal_group(signal.SIGKILL)
                self._wait_for_group_exit(self._stop_grace_period)

            with self._lock:
                status = self._status_unlocked()
                if not status.settled:
                    if self._failure_reason is None:
                        self._failure_reason = "task process group did not settle after SIGKILL"
                    raise TaskSettlementError(
                        self._failure_reason, replace(status, failure_reason=self._failure_reason)
                    )
            return status

    def cancel(self) -> TaskExecutionStatus:
        with self._lock:
            status = self._status_unlocked()
            if status.settled:
                return status
            self._cancel_requested = True
        return self._terminate_group()

    def wait_for_settlement(self, timeout: Optional[float] = None) -> TaskExecutionStatus:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite non-negative number or None")

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self.poll()
            if status.phase == TaskExecutionPhase.TERMINAL:
                break
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"task execution {self.execution_id} did not settle within {timeout} seconds")
                sleep_time = min(self._poll_interval, remaining)
            else:
                sleep_time = self._poll_interval
            time.sleep(sleep_time)

        if status.settled:
            return status

        # A leader exit is not settlement. Give normal group teardown a small
        # bound, then mark the attempt faulty and clean its scope. Long waits
        # never hold _lock, so cancellation and status remain available.
        settle_timeout = self._descendant_settle_timeout
        if deadline is not None:
            settle_timeout = min(settle_timeout, max(0.0, deadline - time.monotonic()))
        if self._wait_for_group_exit(settle_timeout):
            return self.poll()

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"task execution {self.execution_id} did not settle within {timeout} seconds")

        with self._lock:
            status = self._status_unlocked()
            if status.settled:
                return status
            if not self._cancel_requested:
                self._failure_reason = "task leader exited while descendants remained alive"
        return self._terminate_group()


class ProcessTaskLauncher(TaskLauncherSpec):
    """Launch trusted task workers as isolated local POSIX process groups.

    This backend has no CPU, memory, or GPU admission authority. It rejects
    non-empty resource requests rather than treating environment visibility as
    a reservation. An argv can still launch a multi-process group (for example,
    ``torchrun``) when admission is handled by a future site mechanism.
    Descendants must remain in the owned POSIX session; deliberate ``setsid``
    detachment is unsupported and cannot be included in settlement proof.
    """

    launch_mode = LauncherMode.PROCESS.value

    def __init__(
        self,
        stop_grace_period: float = 2.0,
        descendant_settle_timeout: float = 0.25,
        poll_interval: float = 0.05,
    ):
        super().__init__()
        self.stop_grace_period = _positive_number(stop_grace_period, "stop_grace_period")
        self.descendant_settle_timeout = _positive_number(descendant_settle_timeout, "descendant_settle_timeout")
        self.poll_interval = _positive_number(poll_interval, "poll_interval")
        self._attempt_identities = set()
        self._lock = threading.Lock()

    def launch_task(self, request: TaskLaunchRequest) -> ProcessTaskHandle:
        if not isinstance(request, TaskLaunchRequest):
            raise TypeError("request must be a TaskLaunchRequest")
        if os.name != "posix" or not hasattr(os, "killpg"):
            raise TaskLauncherError("ProcessTaskLauncher requires POSIX process-group support")
        if not request.resources.is_empty():
            raise UnsupportedTaskResourceError(
                "ProcessTaskLauncher has no CPU, memory, or GPU admission/reservation mechanism"
            )

        with self._lock:
            if request.identity in self._attempt_identities:
                raise TaskLauncherError(f"task attempt identity was already launched: {request.identity!r}")
            self._attempt_identities.add(request.identity)

        try:
            adapter = spawn_process(list(request.argv), dict(request.environment), cwd=request.cwd)
        except Exception:
            # No physical execution exists, so callers may correct a transient
            # launch problem and submit this attempt identity again.
            with self._lock:
                self._attempt_identities.remove(request.identity)
            raise

        handle = ProcessTaskHandle(
            request,
            adapter,
            stop_grace_period=self.stop_grace_period,
            descendant_settle_timeout=self.descendant_settle_timeout,
            poll_interval=self.poll_interval,
        )
        self.logger.info("launched task attempt %r as %s", request.identity, handle.execution_id)
        return handle
