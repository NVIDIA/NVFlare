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

"""Plain ownership of a task worker and its validated in-memory result.

The trusted caller supplies bootstrap configuration and a launch request factory,
translates its own lifecycle events, and validates receipt replies before passing
them here. This module neither participates in federation nor dispatches events.
Workers share the site's UID and filesystem; this is not a sandbox.
"""

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from nvflare.apis.fl_constant import ReturnCode, TaskResultReceipt
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskHandleSpec,
    TaskLaunchError,
    TaskLauncherSpec,
    TaskLaunchRequest,
    TaskSettlementError,
)

from .artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError, TaskCompletion
from .protocol import TaskAttemptIdentity, WorkerBootstrap, write_bootstrap


@dataclass(frozen=True)
class TaskSupervisorOptions:
    worker_timeout: Optional[float] = None
    settlement_timeout: float = 10.0
    poll_interval: float = 0.05

    def __post_init__(self):
        for name in ("worker_timeout", "settlement_timeout", "poll_interval"):
            value = getattr(self, name)
            if name == "worker_timeout" and value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name != "worker_timeout" and value == 0)
            ):
                bound = "nonnegative" if name == "worker_timeout" else "positive"
                raise ValueError(f"invalid {name}: expected a finite {bound} number")


@dataclass(frozen=True)
class TaskSupervisorOutcome:
    identity: TaskAttemptIdentity
    result: Optional[Shareable] = None
    completion: Optional[TaskCompletion] = None
    aborted: bool = False
    execution_status: Optional[TaskExecutionStatus] = None


@dataclass
class _Attempt:
    identity: TaskAttemptIdentity
    store: FileTaskArtifactStore
    runtime_root: str
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    handle: Optional[TaskHandleSpec] = None
    settled: bool = False
    validated: bool = False
    completion: Optional[TaskCompletion] = None
    failed: bool = False
    receipt: Optional[str] = None
    releasing: bool = False
    released: bool = False
    result_readers: int = 0


class TaskSupervisor:
    """Own one active physical attempt; retain failed or unresolved evidence.

    Instances have one job lifetime. ``end_run`` permanently closes admission.
    Unconfirmed settlement permanently fences this instance, even if a later
    cleanup succeeds. There is no reset or automatic physical retry.

    Default cleanup follows confirmed settlement, a decoded successful result
    and terminal receipt/closure. Uploads use the decoded in-memory result.
    ``retain_artifacts=True`` disables payload release. Completion and
    supervisor diagnostics remain after payload release in either case.
    """

    def __init__(self, *, retain_artifacts: bool = False):
        if not isinstance(retain_artifacts, bool):
            raise TypeError("retain_artifacts must be bool")
        self._retain_artifacts = retain_artifacts
        self._execution_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._diagnostic_lock = threading.Lock()
        self._active = None
        self._attempts = {}
        self._launch_identities = set()
        self._stopping = False
        self._settlement_unconfirmed = False
        self._issues = []

    @property
    def settlement_unconfirmed(self) -> bool:
        with self._state_lock:
            return self._settlement_unconfirmed

    @property
    def owns_execution(self) -> bool:
        """Staging/launch or physical execution is still outstanding."""
        with self._state_lock:
            return self._active is not None and not self._active.settled

    def take_issues(self):
        """Drain diagnostic/cleanup errors for caller-owned logging."""
        with self._state_lock:
            issues, self._issues = self._issues, []
        return issues

    def _diagnostic(self, attempt, event, **facts):
        record = {**attempt.identity.to_dict(), "event": event, "timestamp": time.time(), **facts}
        try:
            os.makedirs(attempt.runtime_root, mode=0o700, exist_ok=True)
            with self._diagnostic_lock:
                with open(os.path.join(attempt.runtime_root, "diagnostics.jsonl"), "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        except Exception as e:
            with self._state_lock:
                attempt.failed = True
                self._issues.append(("error", f"failed to record task diagnostic: {e}"))

    def _fence(self, attempt):
        with self._state_lock:
            attempt.failed = True
            self._stopping = True
            self._settlement_unconfirmed = True
        self._diagnostic(attempt, "settlement_unconfirmed")

    def _confirm_settlement(self, attempt, status):
        if not isinstance(status, TaskExecutionStatus) or not (
            status.settled and status.phase == TaskExecutionPhase.TERMINAL
        ):
            self._fence(attempt)
            raise TaskSettlementError("task execution settlement is unconfirmed", status)
        with self._state_lock:
            attempt.settled = True
        return status

    def _cancel(self, attempt):
        try:
            return self._confirm_settlement(attempt, attempt.handle.cancel())
        except BaseException:
            if not self.settlement_unconfirmed:
                self._fence(attempt)
            raise

    def cancel_active(self):
        """Cancel even when staging or launch has not returned its handle yet."""
        with self._state_lock:
            attempt = self._active
            if attempt is None:
                return
            attempt.cancel_requested.set()
            if attempt.settled:
                return
            handle = attempt.handle
        if handle is not None:
            self._cancel(attempt)

    def end_run(self):
        """Close admission, cancel owned execution, then attempt safe cleanup.

        A concurrent staging/launch call may still be acquiring its handle.
        The caller must join its running ``run`` call before discarding this
        owner. That call cancels a late handle and retains unresolved ownership.
        """
        with self._state_lock:
            self._stopping = True
        try:
            self.cancel_active()
        finally:
            self.cleanup_job_payloads()

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
        deadline_provider: Optional[Callable[[], Optional[float]]] = None,
    ) -> TaskSupervisorOutcome:
        if not isinstance(bootstrap, WorkerBootstrap) or not isinstance(options, TaskSupervisorOptions):
            raise TypeError("bootstrap and options must be worker bootstrap and supervisor options")
        if not isinstance(input_payload, Shareable) or not isinstance(abort_signal, Signal):
            raise TypeError("input_payload and abort_signal must be Shareable and Signal")
        if not isinstance(launcher, TaskLauncherSpec):
            raise TypeError("launcher must be a TaskLauncherSpec")
        if not isinstance(store, FileTaskArtifactStore):
            raise TypeError("store must be a FileTaskArtifactStore")
        if os.path.realpath(bootstrap.artifact_root) != store.root_dir:
            raise ValueError("bootstrap artifact root does not match the owned store")
        if bootstrap.max_payload_bytes != store.max_payload_bytes:
            raise ValueError("bootstrap and owned store must use the same payload policy")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("this task supervisor already has an active attempt")
        attempt = None
        try:
            identity = bootstrap.identity
            launch_identity = (identity.job_id, identity.site_name, identity.task_id, identity.attempt_id)
            with self._state_lock:
                if self._stopping:
                    raise RuntimeError("task supervisor is stopping and rejects new attempts")
                if abort_signal.triggered:
                    return TaskSupervisorOutcome(identity=identity, aborted=True)
                if options.worker_timeout == 0:
                    raise TimeoutError("task worker timeout expired before launch")
                if launch_identity in self._launch_identities:
                    raise RuntimeError("task attempt identity was already owned; resubmit its saved result")
                self._launch_identities.add(launch_identity)
                attempt = _Attempt(identity, store, runtime_root)
                self._attempts[identity] = attempt
                self._active = attempt
            store.create_attempt(identity)
            store.write_input(identity, input_payload)
            bootstrap_path = store.bootstrap_path(identity)
            write_bootstrap(bootstrap_path, bootstrap)
            request = request_factory(identity, bootstrap_path)
            if not isinstance(request, TaskLaunchRequest) or request.identity != launch_identity:
                raise ValueError("launch request does not match staged task attempt identity")
            if self._aborted(attempt, abort_signal):
                attempt.failed = True
                return TaskSupervisorOutcome(identity=identity, aborted=True)
            try:
                handle = launcher.launch_task(request)
            except TaskLaunchError as e:
                # Acquiring a handle is ownership, including post-spawn errors.
                with self._state_lock:
                    attempt.handle = e.handle
                self._diagnostic(attempt, "launch_failed", execution_id=getattr(e.handle, "execution_id", None))
                raise
            with self._state_lock:
                attempt.handle = handle
            self._diagnostic(attempt, "launched", execution_id=handle.execution_id, supervisor_pid=os.getpid())
            if handle.request.identity != launch_identity:
                raise ValueError("launcher handle does not match owned task attempt identity")
            status = self._wait(attempt, abort_signal, options, deadline_provider)
            self._diagnostic(
                attempt,
                "settled",
                execution_id=handle.execution_id,
                exit_code=status.exit_code,
                termination_signal=status.termination_signal,
                failure_reason=status.failure_reason,
                cancel_requested=status.cancel_requested,
            )
            if self._aborted(attempt, abort_signal):
                attempt.failed = True
                return TaskSupervisorOutcome(identity=identity, aborted=True, execution_status=status)
            # Only a settled execution can authorize reading its completion.
            # A later process failure cannot erase a fully verified completion.
            # Keep abnormal-exit and post-publication failure artifacts anyway.
            attempt.failed |= not status.succeeded or os.path.lexists(
                os.path.join(store.attempt_dir(identity), "failure.json")
            )
            try:
                result, completion = store.read_result(identity)
            except IncompleteTaskArtifactError as e:
                failure = store.read_failure(identity)
                if failure is not None:
                    raise RuntimeError(
                        f"task worker {identity.attempt_id} failed: {failure['error_type']}: {failure['message']} "
                        f"(diagnostics: {store.attempt_dir(identity)})"
                    ) from e
                if status.failure_reason:
                    raise RuntimeError(
                        f"task worker {identity.attempt_id} failed: {status.failure_reason} "
                        f"(diagnostics: {store.attempt_dir(identity)})"
                    ) from e
                raise
            if completion.identity != identity:
                raise ValueError("task completion identity does not match owned attempt")
            worker_pid = getattr(handle, "process_group_id", None)
            if worker_pid is not None and completion.worker_pid != worker_pid:
                raise ValueError("task completion worker PID does not match launched process")
            with self._state_lock:
                if self._aborted(attempt, abort_signal):
                    attempt.failed = True
                    return TaskSupervisorOutcome(identity=identity, aborted=True, execution_status=status)
                attempt.validated = True
                attempt.completion = completion
                attempt.failed |= result.get_return_code(default=ReturnCode.OK) != ReturnCode.OK
            self._diagnostic(attempt, "validated", worker_pid=completion.worker_pid, worker_ppid=completion.worker_ppid)
            return TaskSupervisorOutcome(identity, result, completion, execution_status=status)
        except BaseException as e:
            if attempt is not None:
                attempt.failed = True
                self._diagnostic(attempt, "failed", error_type=type(e).__name__, message=str(e))
                if attempt.handle is not None and not attempt.settled and not self.settlement_unconfirmed:
                    self._cancel(attempt)
            raise
        finally:
            with self._state_lock:
                if attempt is not None and (attempt.handle is None or attempt.settled):
                    if not self._settlement_unconfirmed:
                        self._active = None
            self._execution_lock.release()
            self.cleanup_job_payloads()

    def _aborted(self, attempt, abort_signal):
        with self._state_lock:
            return self._stopping or attempt.cancel_requested.is_set() or abort_signal.triggered

    def _wait(self, attempt, abort_signal, options, deadline_provider=None):
        deadline = None if options.worker_timeout is None else time.monotonic() + options.worker_timeout
        while True:
            if self._aborted(attempt, abort_signal):
                return self._cancel(attempt)
            remaining = None if deadline is None else deadline - time.monotonic()
            if deadline_provider is not None:
                caller_deadline = deadline_provider()
                if caller_deadline is not None:
                    if (
                        isinstance(caller_deadline, bool)
                        or not isinstance(caller_deadline, (int, float))
                        or not math.isfinite(caller_deadline)
                    ):
                        raise ValueError("caller execution deadline must be finite or None")
                    caller_remaining = caller_deadline - time.monotonic()
                    remaining = caller_remaining if remaining is None else min(remaining, caller_remaining)
            if remaining is not None and remaining <= 0:
                self._cancel(attempt)
                raise TimeoutError("task worker timed out")
            status = attempt.handle.poll()
            if status.phase == TaskExecutionPhase.TERMINAL:
                timeout = (
                    options.settlement_timeout if remaining is None else min(options.settlement_timeout, remaining)
                )
                try:
                    status = attempt.handle.wait_for_settlement(timeout=timeout)
                except TimeoutError:
                    self._cancel(attempt)
                    raise
                except TaskSettlementError:
                    self._fence(attempt)
                    raise
                return self._confirm_settlement(attempt, status)
            attempt.cancel_requested.wait(
                options.poll_interval if remaining is None else min(options.poll_interval, remaining)
            )

    def record_receipt(self, identity: TaskAttemptIdentity, receipt: Optional[str]) -> bool:
        """Record an already identity-validated P1 receipt; never aggregation admission.

        The caller verifies workflow, task and attempt on the transport reply.
        RETRY/unknown keep the saved result. Terminal receipt is monotonic;
        closure makes no claim about an earlier upload's receipt or use.
        """
        if receipt not in (None, TaskResultReceipt.RETRY, TaskResultReceipt.RECEIVED, TaskResultReceipt.TASK_CLOSED):
            raise ValueError("invalid task result receipt")
        with self._state_lock:
            attempt = self._attempts.get(identity)
            if attempt is None or not attempt.validated:
                return False
            if attempt.receipt is None and receipt in (TaskResultReceipt.RECEIVED, TaskResultReceipt.TASK_CLOSED):
                attempt.receipt = receipt
        self._diagnostic(attempt, "receipt", receipt=receipt)
        self.cleanup_job_payloads()
        return True

    def read_saved_result(self, identity: TaskAttemptIdentity) -> Shareable:
        """Re-read a settled result without launching or replaying worker side effects."""
        with self._state_lock:
            attempt = self._attempts.get(identity)
            if (
                self._stopping
                or self._settlement_unconfirmed
                or attempt is None
                or not attempt.settled
                or not attempt.validated
                or attempt.releasing
                or attempt.released
            ):
                raise RuntimeError("task attempt has no retained, settled result to resubmit")
            attempt.result_readers += 1
        try:
            result, completion = attempt.store.read_result(identity)
            if completion != attempt.completion:
                raise ValueError("saved task completion changed after validation")
            return result
        except BaseException:
            with self._state_lock:
                attempt.failed = True
            raise
        finally:
            with self._state_lock:
                attempt.result_readers -= 1
            self.cleanup_job_payloads()

    def retain_attempt(self, identity: TaskAttemptIdentity):
        """Keep evidence when caller-owned result processing fails after compute."""
        with self._state_lock:
            attempt = self._attempts.get(identity)
            if attempt is not None:
                attempt.failed = True
        if attempt is not None:
            self._diagnostic(attempt, "retained_by_caller")

    def cleanup_job_payloads(self):
        """Idempotently release eligible payloads; keep evidence on uncertainty."""
        with self._state_lock:
            if self._retain_artifacts or self._settlement_unconfirmed:
                return
            attempts = list(self._attempts.values())
        for attempt in attempts:
            with self._state_lock:
                if (
                    self._settlement_unconfirmed
                    or not attempt.settled
                    or not attempt.validated
                    or attempt.failed
                    or attempt.result_readers
                    or attempt.receipt is None
                    or attempt.releasing
                    or attempt.released
                ):
                    continue
                attempt.releasing = True
            try:
                attempt.store.release_payloads(attempt.identity)
            except Exception as e:
                with self._state_lock:
                    attempt.failed = True
                    self._issues.append(("warning", f"failed to release task payloads: {e}"))
                self._diagnostic(attempt, "release_failed", message=str(e))
            else:
                with self._state_lock:
                    attempt.released = True
                self._diagnostic(attempt, "released")
                with self._state_lock:
                    if not attempt.failed:
                        # Replay prevention needs only _launch_identities;
                        # released attempts must not retain handles or join
                        # every later cleanup scan.
                        self._attempts.pop(attempt.identity, None)
