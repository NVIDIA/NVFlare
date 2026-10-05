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

"""Role-neutral ownership of disposable task attempts and their artifacts.

The caller supplies the trusted launch invocation and translates its own run,
abort, and publication events. This service has no client engine, FLContext,
federation event, or application analytics dispatch responsibilities.
"""

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_execution import TaskArtifactCleanup
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskLauncherSpec, TaskLaunchRequest
from nvflare.apis.task_state import TaskState

from .artifacts import FileTaskArtifactStore, TaskCompletion
from .protocol import TaskAttemptIdentity, WorkerBootstrap, write_bootstrap


@dataclass(frozen=True)
class TaskSupervisorOptions:
    worker_timeout: Optional[float] = None
    poll_interval: float = 0.05
    result_wait_timeout: Optional[float] = None

    def __post_init__(self):
        for name in ("worker_timeout", "result_wait_timeout"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number or None")
        if (
            isinstance(self.poll_interval, bool)
            or not isinstance(self.poll_interval, (int, float))
            or not math.isfinite(self.poll_interval)
            or self.poll_interval <= 0
        ):
            raise ValueError("poll_interval must be a finite positive number")


@dataclass(frozen=True)
class TaskSupervisorOutcome:
    identity: TaskAttemptIdentity
    result: Optional[Shareable] = None
    completion: Optional[TaskCompletion] = None
    analytics: tuple[dict, ...] = field(default_factory=tuple)
    aborted: bool = False


@dataclass(frozen=True)
class _PendingAcknowledgement:
    store: FileTaskArtifactStore
    runtime_root: str
    diagnostic: dict
    completion: TaskCompletion
    state_store: Optional[object]
    result_succeeded: bool


class TaskSupervisor:
    """Stage, settle, read, and retain attempts independently of their caller's role."""

    def __init__(self, artifact_cleanup=TaskArtifactCleanup.JOB):
        self.artifact_cleanup = TaskArtifactCleanup.validate(artifact_cleanup)
        self._execution_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._diagnostic_lock = threading.Lock()
        self._retention_lock = threading.RLock()
        self._active_handle = None
        self._active_abort_signal = None
        self._stopping = False
        self._settlement_unconfirmed = False
        self._pending_publication = {}
        self._retained_attempts = {}
        self._job_cleanup_requested = False
        self._issues = []

    @property
    def settlement_unconfirmed(self):
        with self._state_lock:
            return self._settlement_unconfirmed

    def start_run(self):
        with self._state_lock:
            self._stopping = False
            self._settlement_unconfirmed = False
            self._job_cleanup_requested = False

    def _fail_unconfirmed_settlement(self):
        with self._state_lock:
            self._stopping = True
            self._settlement_unconfirmed = True

    def cancel_active(self):
        with self._state_lock:
            handle = self._active_handle
            if self._active_abort_signal is not None:
                self._active_abort_signal.trigger(True)
        if handle is not None:
            try:
                status = handle.cancel()
                if not status.settled:
                    self._fail_unconfirmed_settlement()
            except BaseException:
                self._fail_unconfirmed_settlement()
                raise

    def end_run(self):
        with self._state_lock:
            self._stopping = True
            self._job_cleanup_requested = True
        try:
            self.cancel_active()
        finally:
            self.cleanup_job_payloads()

    def _record_issue(self, level, message):
        with self._state_lock:
            self._issues.append((level, message))

    def take_issues(self):
        """Drain ordinary diagnostics for a caller-owned logging adapter."""
        with self._state_lock:
            issues, self._issues = self._issues, []
        return issues

    def cleanup_job_payloads(self):
        """Release only owned attempts, after all worker and read activity settles."""
        if self.artifact_cleanup != TaskArtifactCleanup.JOB or not self._job_cleanup_requested:
            return
        if not self._execution_lock.acquire(blocking=False):
            return
        try:
            if not self._retention_lock.acquire(blocking=False):
                return
            try:
                with self._state_lock:
                    if self._active_handle is not None:
                        return
                    attempts = list(self._retained_attempts.items())
                for identity, (_, store) in attempts:
                    with self._state_lock:
                        pending = self._pending_publication.get(identity)
                    # Declared state is a candidate paired with this result, not
                    # transient worker scratch. Its admission may race END_RUN.
                    if pending is not None and pending.state_store is not None:
                        continue
                    try:
                        store.release_payloads(identity)
                    except Exception as e:
                        self._record_issue("warning", f"failed to remove job-ended task worker artifacts: {e}")
                    else:
                        with self._state_lock:
                            self._retained_attempts.pop(identity, None)
            finally:
                self._retention_lock.release()
        finally:
            self._execution_lock.release()

    def _append_diagnostic(self, runtime_root: str, record: dict):
        os.makedirs(runtime_root, mode=0o700, exist_ok=True)
        path = os.path.join(runtime_root, "diagnostics.jsonl")
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._diagnostic_lock:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    @staticmethod
    def _new_diagnostic(identity: TaskAttemptIdentity):
        return {
            **identity.to_dict(),
            "supervisor_pid": os.getpid(),
            "worker_pid": None,
            "worker_ppid": None,
            "execution_id": None,
            "launch_timestamp": None,
            "settled_timestamp": None,
            "worker_started_timestamp": None,
            "worker_completed_timestamp": None,
            "publication_timestamp": None,
            "publication_outcome": None,
        }

    def run(
        self,
        *,
        bootstrap: WorkerBootstrap,
        store: FileTaskArtifactStore,
        input_payload: Shareable,
        runtime_root: str,
        launcher: TaskLauncherSpec,
        abort_signal: Signal,
        request_factory: Callable[[TaskAttemptIdentity, str], TaskLaunchRequest],
        options: TaskSupervisorOptions = TaskSupervisorOptions(),
        state_store=None,
    ) -> TaskSupervisorOutcome:
        if not isinstance(input_payload, Shareable):
            raise TypeError("task worker input must be a Shareable")
        if not isinstance(bootstrap, WorkerBootstrap):
            raise TypeError("bootstrap must be a WorkerBootstrap")
        if not isinstance(options, TaskSupervisorOptions):
            raise TypeError("options must be TaskSupervisorOptions")
        if not isinstance(launcher, TaskLauncherSpec):
            raise TypeError("launcher must be a TaskLauncherSpec")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("this task supervisor already has an active task worker")
        with self._state_lock:
            self._active_abort_signal = abort_signal
        try:
            with self._state_lock:
                if abort_signal.triggered:
                    return TaskSupervisorOutcome(identity=bootstrap.identity, aborted=True)
                if self._stopping:
                    raise RuntimeError("Job is stopping and rejects a new task worker")
            if options.worker_timeout == 0:
                raise TimeoutError("task worker result timeout expired before launch")
            if bootstrap.state_names and state_store is None:
                raise ValueError("declared task state requires a supervising state store")
            if state_store is not None:
                if state_store.names != bootstrap.state_names:
                    raise ValueError("supervisor state declarations do not match the worker bootstrap")
                revision, records = state_store.snapshot()
                bootstrap = replace(bootstrap, state_revision=revision, state_records=records)
            return self._run(
                bootstrap,
                store,
                input_payload,
                runtime_root,
                launcher,
                abort_signal,
                request_factory,
                options,
                state_store,
            )
        finally:
            with self._state_lock:
                self._active_abort_signal = None
            self._execution_lock.release()
            self.cleanup_job_payloads()

    def _run(
        self,
        bootstrap,
        store,
        input_payload,
        runtime_root,
        launcher,
        abort_signal,
        request_factory,
        options,
        state_store,
    ):
        identity = bootstrap.identity
        with self._state_lock:
            if identity in self._pending_publication:
                raise RuntimeError(f"attempt {identity.attempt_id!r} already has a result pending publication")
        store.create_attempt(identity)
        with self._state_lock:
            self._retained_attempts[identity] = (identity, store)
        store.write_input(identity, input_payload)
        bootstrap_path = store.bootstrap_path(identity)
        write_bootstrap(bootstrap_path, bootstrap)

        diagnostic = self._new_diagnostic(identity)
        diagnostic["launcher_class"] = f"{type(launcher).__module__}.{type(launcher).__qualname__}"
        diagnostic["launcher_mode"] = launcher.launch_mode
        diagnostic["artifact_cleanup"] = self.artifact_cleanup
        request = request_factory(identity, bootstrap_path)
        if request.identity != (identity.job_id, identity.site_name, identity.task_id, identity.attempt_id):
            raise ValueError("launch request does not match staged task attempt identity")
        diagnostic["launch_timestamp"] = time.time()
        handle = launcher.launch_task(request)
        with self._state_lock:
            self._active_handle = handle
            stopping = self._stopping
        try:
            diagnostic["execution_id"] = handle.execution_id
            diagnostic["worker_pid"] = getattr(handle, "process_group_id", None)
            self._append_diagnostic(runtime_root, {**diagnostic, "event": "launched"})
            deadline = None if options.worker_timeout is None else time.monotonic() + options.worker_timeout
            cancellation_reason = None
            while True:
                if stopping or self._stopping or abort_signal.triggered:
                    cancellation_reason = "aborted"
                    status = handle.cancel()
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if options.result_wait_timeout is not None:
                    wait_started, result_sent = store.result_wait_state(identity)
                    if wait_started is not None and (
                        result_sent is None or result_sent > wait_started + options.result_wait_timeout
                    ):
                        result_remaining = wait_started + options.result_wait_timeout - time.monotonic()
                        remaining = result_remaining if remaining is None else min(remaining, result_remaining)
                if remaining is not None and remaining <= 0:
                    cancellation_reason = "timed out"
                    status = handle.cancel()
                    break
                status = handle.poll()
                if status.phase == TaskExecutionPhase.TERMINAL:
                    try:
                        status = handle.wait_for_settlement(timeout=remaining)
                    except TimeoutError:
                        cancellation_reason = "timed out"
                        status = handle.cancel()
                    break
                time.sleep(options.poll_interval if remaining is None else min(options.poll_interval, remaining))

            diagnostic["settled_timestamp"] = time.time()
            self._append_diagnostic(runtime_root, {**diagnostic, "event": "settled", "status": status.phase.value})
            if not status.settled:
                raise RuntimeError(f"task worker did not settle: {status.failure_reason or status}")
            if cancellation_reason == "aborted" or (cancellation_reason is None and status.cancel_requested):
                return TaskSupervisorOutcome(identity=identity, aborted=True)
            if cancellation_reason:
                raise RuntimeError(f"task worker {cancellation_reason}")
            if not status.succeeded:
                details = status.failure_reason or (
                    f"exit code {status.exit_code}"
                    if status.exit_code is not None
                    else f"termination signal {status.termination_signal}"
                )
                raise RuntimeError(f"task worker failed: {details}")

            result, completion = store.read_result(identity)
            if bootstrap.state_names:
                if completion.state is None:
                    raise ValueError("declared task state is missing from the committed task completion")
                if completion.state_revision != bootstrap.state_revision:
                    raise ValueError("task completion state revision does not match the staged input revision")
                TaskState.from_wire(bootstrap.state_names, store.read_state(identity, completion.state))
            launched_pid = diagnostic["worker_pid"]
            if launched_pid is not None and completion.worker_pid != launched_pid:
                raise ValueError(
                    f"task completion worker PID {completion.worker_pid} does not match launched process {launched_pid}"
                )
            diagnostic["worker_pid"] = completion.worker_pid
            diagnostic["worker_ppid"] = completion.worker_ppid
            diagnostic["worker_started_timestamp"] = completion.started_at
            diagnostic["worker_completed_timestamp"] = completion.completed_at
            analytics = tuple(store.read_analytics(identity, completion))
            with self._state_lock:
                self._pending_publication[identity] = _PendingAcknowledgement(
                    store,
                    runtime_root,
                    diagnostic,
                    completion,
                    state_store,
                    result.get_return_code(default=ReturnCode.OK) == ReturnCode.OK,
                )
            return TaskSupervisorOutcome(identity=identity, result=result, completion=completion, analytics=analytics)
        except BaseException:
            try:
                observed = handle.poll()
                if not observed.settled:
                    handle.cancel()
            except BaseException:
                self._fail_unconfirmed_settlement()
                raise
            raise
        finally:
            try:
                settled = handle.poll().settled
            except Exception:
                settled = False
            with self._state_lock:
                if settled and self._active_handle is handle:
                    self._active_handle = None
                elif not settled:
                    self._fail_unconfirmed_settlement()

    def acknowledge(
        self, identity: TaskAttemptIdentity, *, sent: bool, admitted: Optional[bool], result_succeeded: bool = True
    ):
        """Record caller publication facts for exactly one physical attempt."""
        if not isinstance(identity, TaskAttemptIdentity):
            raise TypeError("identity must be a TaskAttemptIdentity")
        with self._retention_lock:
            try:
                self._acknowledge(identity, sent=sent, admitted=admitted, result_succeeded=result_succeeded)
            finally:
                self.cleanup_job_payloads()

    def _acknowledge(self, identity, *, sent, admitted, result_succeeded):
        with self._state_lock:
            pending = self._pending_publication.pop(identity, None)
        if pending is None:
            return
        store, runtime_root, diagnostic = pending.store, pending.runtime_root, pending.diagnostic
        completion, state_store = pending.completion, pending.state_store
        if state_store is not None and (sent is not True or not isinstance(admitted, bool)):
            with self._state_lock:
                self._pending_publication[identity] = pending
                self._stopping = True
            raise RuntimeError("task state admission is unconfirmed; retaining the candidate and stopping the job")
        succeeded = sent is True and admitted is True
        if succeeded and pending.result_succeeded and result_succeeded is True and state_store is not None:
            try:
                state_store.commit(identity, completion, store)
            except Exception:
                with self._state_lock:
                    self._pending_publication[identity] = pending
                    self._stopping = True
                raise
        diagnostic["publication_timestamp"] = time.time()
        diagnostic["publication_outcome"] = "accepted" if succeeded else "not_accepted"
        try:
            self._append_diagnostic(runtime_root, {**diagnostic, "event": "publication"})
        except Exception as e:
            self._record_issue("error", f"failed to record task worker publication outcome: {e}")
            return
        if succeeded and self.artifact_cleanup == TaskArtifactCleanup.ACCEPTED:
            try:
                store.release_payloads(identity)
                with self._state_lock:
                    self._retained_attempts.pop(identity, None)
            except Exception as e:
                self._record_issue("warning", f"failed to remove accepted task worker artifacts: {e}")
