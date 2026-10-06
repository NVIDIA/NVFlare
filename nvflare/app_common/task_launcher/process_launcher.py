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

The owned execution scope is one POSIX process group. The command may launch
any number of ranks inside that group. Moving descendants into another group,
even within the same session, is unsupported and outside settlement proof.

Requires Linux, or macOS with Python >= 3.13, and waitid/WNOWAIT support.
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


class _OwnedProcessAdapter:
    """Retain the child PID through cleanup and until the final group probe.

    An unreaped leader reserves the numeric PID/PGID, including after exit.
    Only this adapter may reap the child; an external reaper invalidates the
    ownership proof and must prevent further signals or successful settlement.
    """

    def __init__(self, adapter: ProcessAdapter):
        self._adapter = adapter
        self.pid = adapter.pid
        self._return_code = None
        self.reaped = False

    def _observe(self):
        try:
            return os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError as e:
            raise TaskLauncherError("task leader was externally reaped; process-group ownership is lost") from e

    def poll(self) -> Optional[int]:
        if self.reaped:
            return self._return_code
        observation = self._observe()
        if observation is not None and observation.si_pid:
            self._return_code = (
                observation.si_status & 0xFF if observation.si_code == os.CLD_EXITED else -observation.si_status
            )
        return self._return_code

    def verify_identity(self) -> None:
        if self.reaped:
            raise TaskLauncherError("task leader is already reaped; process-group ownership is no longer reserved")
        self._observe()

    @property
    def owns_exited_leader(self) -> bool:
        return self._return_code is not None and not self.reaped

    def reap(self) -> int:
        # Reap after no live group members were observed; the caller must then
        # confirm group absence. Populate the wrapped Popen/ProcessAdapter
        # return code so its finalizer cannot reap again.
        if not self.reaped:
            self.verify_identity()
            return_code = self._adapter.poll()
            if return_code is None:
                raise TaskLauncherError("task leader did not exit during reaping")
            # A completed reap is final. Its canonical waitpid/Popen return
            # code is authoritative; never try to observe the released PID.
            self._return_code = return_code
            self.reaped = True
        return self._return_code


class ProcessTaskHandle(TaskHandleSpec):
    """Own and observe one task process group."""

    def __init__(
        self,
        request: TaskLaunchRequest,
        adapter: _OwnedProcessAdapter,
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
        self._group_members = {}
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
        # Remember observed members by psutil identity (PID plus creation
        # time). Darwin can drop a dying member's PGID before status reports
        # death. A vanished group probe alone cannot retire that observation.
        live_member_seen = False
        for pid, process in list(self._group_members.items()):
            try:
                if process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    live_member_seen = True
                    continue
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                return True
            del self._group_members[pid]
        try:
            os.killpg(self._process_group_id, 0)
        except ProcessLookupError:
            return live_member_seen
        except PermissionError:
            # Lack of probe permission is not evidence that resources settled.
            # Darwin also reports EPERM for our retained zombie leader, whose
            # terminal state is known from waitid. Still inspect every member.
            if self._adapter.owns_exited_leader is not True:
                return True

        if self._adapter.reaped is True:
            # The last scan was non-atomic: a descendant might have forked a
            # worker after its PID snapshot. Only group absence can finalize
            # settlement after releasing the leader. A remaining/reused group
            # is uncertain, including when its snapshot shows only zombies.
            return True

        # killpg also sees unreaped zombies. They cannot execute or hold compute
        # resources and cannot be removed with signals. Ignore a group only
        # when all observed members are dead; uncertain observations fail closed.
        dead_member_seen = self._adapter.owns_exited_leader is True
        try:
            for process in psutil.process_iter():
                try:
                    if os.getpgid(process.pid) != self._process_group_id:
                        continue
                    if process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                        self._group_members[process.pid] = process
                        live_member_seen = True
                    else:
                        dead_member_seen = True
                except (ProcessLookupError, psutil.NoSuchProcess):
                    continue
                except (PermissionError, psutil.AccessDenied):
                    return True
        except (PermissionError, psutil.AccessDenied):
            return True
        if live_member_seen:
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
        with self._lock:
            if self._settled_status is not None:
                return
            try:
                # Keep the leader unreaped across this check and signal. The
                # lock also excludes our own settlement/reaping in poll().
                self._adapter.verify_identity()
            except TaskLauncherError as e:
                self._failure_reason = str(e)
                raise TaskSettlementError(self._failure_reason, self._status_unlocked()) from e
            try:
                os.killpg(self._process_group_id, sig)
            except ProcessLookupError:
                return
            except PermissionError as e:
                self.logger.warning("cannot signal task process group %s: %s", self._process_group_id, e)

    def _status_unlocked(self) -> TaskExecutionStatus:
        if self._settled_status is not None:
            return self._settled_status
        try:
            return_code = self._adapter.poll()
        except TaskLauncherError as e:
            self._failure_reason = str(e)
            return TaskExecutionStatus(
                phase=TaskExecutionPhase.TERMINAL,
                cancel_requested=self._cancel_requested,
                failure_reason=self._failure_reason,
            )
        group_exists = self._group_exists()
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
            try:
                reaped_code = self._adapter.reap()
            except TaskLauncherError as e:
                self._failure_reason = str(e)
                return replace(status, settled=False, failure_reason=self._failure_reason)
            if self._adapter.reaped is True:
                status = replace(
                    status,
                    exit_code=reaped_code if reaped_code >= 0 else None,
                    termination_signal=-reaped_code if reaped_code < 0 else None,
                )
                # Reaping releases numeric identity. Do not signal this group
                # again, and do not trust another non-atomic member snapshot.
                # A fresh absent-group probe closes the membership-churn race.
                status = replace(status, settled=not self._group_exists())
                if not status.settled:
                    return status
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
            # Observe without reaping while live group members are observed. A
            # zombie leader reserves identity but holds no compute resources.
            if self.poll().settled:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(self._poll_interval, remaining))

    def _terminate_group(self, deadline: Optional[float] = None) -> TaskExecutionStatus:
        if deadline is None:
            acquired = self._termination_lock.acquire()
        else:
            acquired = self._termination_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
        if not acquired:
            raise TimeoutError(f"task execution {self.execution_id} did not settle before its deadline")
        try:
            status = self.poll()
            if status.settled:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"task execution {self.execution_id} did not settle before its deadline")

            def remaining_grace():
                if deadline is None:
                    return self._stop_grace_period
                return min(self._stop_grace_period, max(0.0, deadline - time.monotonic()))

            self._signal_group(signal.SIGTERM)
            if not self._wait_for_group_exit(remaining_grace()):
                self._signal_group(signal.SIGKILL)
                self._wait_for_group_exit(remaining_grace())

            with self._lock:
                status = self._status_unlocked()
                if not status.settled:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(f"task execution {self.execution_id} did not settle before its deadline")
                    if self._failure_reason is None:
                        self._failure_reason = "task process group did not settle after SIGKILL"
                    raise TaskSettlementError(
                        self._failure_reason, replace(status, failure_reason=self._failure_reason)
                    )
            return status
        finally:
            self._termination_lock.release()

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
        return self._terminate_group(deadline=deadline)


class ProcessTaskLauncher(TaskLauncherSpec):
    """Launch trusted task workers as isolated local POSIX process groups.

    This backend has no CPU, memory, or GPU admission authority. It rejects
    non-empty resource requests rather than treating environment visibility as
    a reservation. Multi-process applications are supported only when every
    descendant remains in the owned process group. Changing group or session
    is unsupported, including a different group in the same POSIX session.
    Requires Linux, or macOS with Python >= 3.13, and waitid/WNOWAIT support.

    Attempt identities are retained for this launcher's lifetime and never
    evicted. Once ``max_attempt_identities`` is reached, new launches fail
    closed. Use a new launcher for a new trusted scope instead of forgetting
    identities and permitting physical relaunches within the existing scope.
    """

    launch_mode = "process"

    def __init__(
        self,
        stop_grace_period: float = 2.0,
        descendant_settle_timeout: float = 0.25,
        poll_interval: float = 0.05,
        max_attempt_identities: int = 10000,
    ):
        super().__init__()
        self.stop_grace_period = _positive_number(stop_grace_period, "stop_grace_period")
        self.descendant_settle_timeout = _positive_number(descendant_settle_timeout, "descendant_settle_timeout")
        self.poll_interval = _positive_number(poll_interval, "poll_interval")
        if (
            isinstance(max_attempt_identities, bool)
            or not isinstance(max_attempt_identities, int)
            or max_attempt_identities <= 0
        ):
            raise ValueError("max_attempt_identities must be a positive integer")
        self._max_attempt_identities = max_attempt_identities
        self._attempt_identities = set()
        self._lock = threading.Lock()

    def launch_task(self, request: TaskLaunchRequest) -> ProcessTaskHandle:
        if not isinstance(request, TaskLaunchRequest):
            raise TypeError("request must be a TaskLaunchRequest")
        if os.name != "posix" or not all(
            hasattr(os, name) for name in ("killpg", "waitid", "WNOWAIT", "P_PID", "WEXITED", "WNOHANG")
        ):
            raise TaskLauncherError("ProcessTaskLauncher requires POSIX process groups and waitid/WNOWAIT support")
        if not request.resources.is_empty():
            raise UnsupportedTaskResourceError(
                "ProcessTaskLauncher has no CPU, memory, or GPU admission/reservation mechanism"
            )

        with self._lock:
            if request.identity in self._attempt_identities:
                raise TaskLauncherError(f"task attempt identity was already launched: {request.identity!r}")
            if len(self._attempt_identities) >= self._max_attempt_identities:
                raise TaskLauncherError("task attempt identity capacity exhausted; use a new launcher for a new scope")
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
            _OwnedProcessAdapter(adapter),
            stop_grace_period=self.stop_grace_period,
            descendant_settle_timeout=self.descendant_settle_timeout,
            poll_interval=self.poll_interval,
        )
        self.logger.info("launched task attempt %r as %s", request.identity, handle.execution_id)
        return handle
