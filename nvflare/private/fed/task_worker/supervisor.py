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
import uuid
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

from .artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError, TaskCompletion, _write_json_exclusive
from .protocol import SESSION_END_MARKER, TaskAttemptIdentity, WorkerBootstrap, session_entry_name, write_bootstrap


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
    settled: bool = False  # physical: the process that owned this record has exited and been reaped
    published: bool = False  # the attempt's artifacts are final; the worker never touches them again
    validated: bool = False
    completion: Optional[TaskCompletion] = None
    failed: bool = False
    receipt: Optional[str] = None
    releasing: bool = False
    released: bool = False


@dataclass
class _Session(_Attempt):
    """One worker process; ``identity`` is its first attempt. Without a group it serves that attempt only."""

    group_id: Optional[str] = None
    session_dir: Optional[str] = None
    launcher: Optional[TaskLauncherSpec] = None
    request_factory: Optional[Callable] = None
    staged: int = 0


class TaskSupervisor:
    """Own one active worker session and its attempts; retain failed or unresolved evidence.

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
        identity = attempt.identity.to_dict() if attempt.identity is not None else {}
        record = {**identity, "event": event, "timestamp": time.time(), **facts}
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
        stage_hook: Optional[Callable[[TaskAttemptIdentity], None]] = None,
    ) -> TaskSupervisorOutcome:
        """Run one attempt in a worker of its own: a session of one that ends when the worker exits."""
        if not isinstance(bootstrap, WorkerBootstrap):
            raise TypeError("bootstrap must be a WorkerBootstrap")
        if bootstrap.session_dir is not None:
            raise ValueError("a one-shot worker bootstrap cannot name a session directory")
        session = self.open_session(None, store, runtime_root, launcher, request_factory)
        return self.run_attempt(session, bootstrap, input_payload, abort_signal, options, deadline_provider, stage_hook)

    def open_session(self, group_id, store, runtime_root, launcher, request_factory) -> _Session:
        """Record a session; its worker is launched by the first ``run_attempt``.

        Without a group the session has no directory: its worker serves one attempt and exits.
        """
        if not isinstance(launcher, TaskLauncherSpec):
            raise TypeError("launcher must be a TaskLauncherSpec")
        if not isinstance(store, FileTaskArtifactStore):
            raise TypeError("store must be a FileTaskArtifactStore")
        session_dir = None
        if group_id is not None:
            session_dir = os.path.join(runtime_root, "sessions", str(uuid.uuid4()))
            os.makedirs(session_dir, mode=0o700)
        return _Session(
            identity=None,
            store=store,
            runtime_root=runtime_root,
            group_id=group_id,
            session_dir=session_dir,
            launcher=launcher,
            request_factory=request_factory,
        )

    def run_attempt(
        self,
        session: _Session,
        bootstrap: WorkerBootstrap,
        input_payload: Shareable,
        abort_signal: Signal,
        options: TaskSupervisorOptions = TaskSupervisorOptions(),
        deadline_provider: Optional[Callable[[], Optional[float]]] = None,
        stage_hook: Optional[Callable[[TaskAttemptIdentity], None]] = None,
    ) -> TaskSupervisorOutcome:
        """Stage one attempt for the session's worker, launching it first if needed, and wait for its completion."""
        if not isinstance(bootstrap, WorkerBootstrap) or not isinstance(options, TaskSupervisorOptions):
            raise TypeError("bootstrap and options must be worker bootstrap and supervisor options")
        if not isinstance(input_payload, Shareable) or not isinstance(abort_signal, Signal):
            raise TypeError("input_payload and abort_signal must be Shareable and Signal")
        if not isinstance(session, _Session):
            raise TypeError("session must come from open_session")
        store = session.store
        if bootstrap.session_dir != session.session_dir:
            raise ValueError("bootstrap does not name the session's directory")
        if os.path.realpath(bootstrap.artifact_root) != store.root_dir:
            raise ValueError("bootstrap artifact root does not match the owned store")
        if bootstrap.max_payload_bytes != store.max_payload_bytes:
            raise ValueError("bootstrap and owned store must use the same payload policy")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("this task supervisor already has an active attempt")
        identity = bootstrap.identity
        attempt = None
        try:
            launch_identity = (identity.job_id, identity.site_name, identity.task_id, identity.attempt_id)
            with self._state_lock:
                if self._stopping:
                    raise RuntimeError("task supervisor is stopping and rejects new attempts")
                if session.settled:
                    raise RuntimeError("task worker session has ended")
                if abort_signal.triggered:
                    return TaskSupervisorOutcome(identity=identity, aborted=True)
                if options.worker_timeout == 0:
                    raise TimeoutError("task worker timeout expired before launch")
                if launch_identity in self._launch_identities:
                    raise RuntimeError("task attempt identity was already owned; resubmit its saved result")
                self._launch_identities.add(launch_identity)
                attempt = _Attempt(identity, store, session.runtime_root)
                self._attempts[identity] = attempt
                if session.identity is None:
                    session.identity = identity
                self._active = session
            store.create_attempt(identity)
            store.write_input(identity, input_payload)
            if stage_hook is not None:
                stage_hook(identity)
            if self._aborted(session, abort_signal):
                attempt.failed = True
                return TaskSupervisorOutcome(identity=identity, aborted=True)
            session.staged += 1
            if session.handle is None:
                self._launch(session, attempt, bootstrap, launch_identity)
            else:
                _write_json_exclusive(session.session_dir, session_entry_name(session.staged), identity.to_dict())
                self._diagnostic(attempt, "staged", session=session.session_dir, sequence=session.staged)
            # A one-shot worker reports completion by exiting; a session worker publishes it and lives on.
            done_path = store.completion_path(identity) if session.session_dir is not None else None
            status = self._wait(session, abort_signal, options, deadline_provider, done_path=done_path)
            if status is not None:
                self._diagnostic(
                    attempt,
                    "settled",
                    execution_id=session.handle.execution_id,
                    exit_code=status.exit_code,
                    termination_signal=status.termination_signal,
                    failure_reason=status.failure_reason,
                    cancel_requested=status.cancel_requested,
                )
            if self._aborted(session, abort_signal):
                attempt.failed = True
                return TaskSupervisorOutcome(identity=identity, aborted=True, execution_status=status)
            # Only a settled execution or a published completion authorizes reading a
            # result. A later process failure cannot erase a fully verified completion.
            # Keep abnormal-exit and post-publication failure artifacts anyway.
            attempt.failed |= (status is not None and not status.succeeded) or store.has_failure(identity)
            try:
                result, completion = store.read_result(identity)
            except IncompleteTaskArtifactError as e:
                failure = store.read_failure(identity)
                if failure is not None:
                    raise RuntimeError(
                        f"task worker {identity.attempt_id} failed: {failure['error_type']}: {failure['message']} "
                        f"(diagnostics: {store.attempt_dir(identity)})"
                    ) from e
                raise
            if completion.identity != identity:
                raise ValueError("task completion identity does not match owned attempt")
            worker_pid = getattr(session.handle, "process_group_id", None)
            if worker_pid is not None and completion.worker_pid != worker_pid:
                raise ValueError("task completion worker PID does not match launched process")
            with self._state_lock:
                if self._aborted(session, abort_signal):
                    attempt.failed = True
                    return TaskSupervisorOutcome(identity=identity, aborted=True, execution_status=status)
                # The worker never touches a published attempt again, so its artifacts are final
                # even while the session's process lives on; physical settlement is the session's.
                attempt.published = True
                attempt.validated = True
                attempt.completion = completion
                attempt.failed |= result.get_return_code(default=ReturnCode.OK) != ReturnCode.OK
            self._diagnostic(attempt, "validated", worker_pid=completion.worker_pid, worker_ppid=completion.worker_ppid)
            return TaskSupervisorOutcome(identity, result, completion, execution_status=status)
        except BaseException as e:
            if attempt is not None:
                attempt.failed = True
                self._diagnostic(attempt, "failed", error_type=type(e).__name__, message=str(e))
            if session.handle is not None and not session.settled and not self.settlement_unconfirmed:
                self._cancel(session)
            raise
        finally:
            with self._state_lock:
                if (session.handle is None or session.settled) and not self._settlement_unconfirmed:
                    self._active = None
            self._execution_lock.release()
            self.cleanup_job_payloads()

    def _launch(self, session, attempt, bootstrap, launch_identity):
        store = session.store
        bootstrap_path = store.bootstrap_path(attempt.identity)
        write_bootstrap(bootstrap_path, bootstrap)
        request = session.request_factory(attempt.identity, bootstrap_path)
        if not isinstance(request, TaskLaunchRequest) or request.identity != launch_identity:
            raise ValueError("launch request does not match staged task attempt identity")
        try:
            handle = session.launcher.launch_task(request)
        except TaskLaunchError as e:
            # Acquiring a handle is ownership, including post-spawn errors.
            with self._state_lock:
                session.handle = e.handle
            self._diagnostic(attempt, "launch_failed", execution_id=getattr(e.handle, "execution_id", None))
            raise
        with self._state_lock:
            session.handle = handle
        self._diagnostic(
            attempt,
            "launched",
            execution_id=handle.execution_id,
            supervisor_pid=os.getpid(),
            session=session.session_dir,
        )
        if handle.request.identity != launch_identity:
            raise ValueError("launcher handle does not match owned task attempt identity")

    def close_session(
        self, session: _Session, timeout: float, options: TaskSupervisorOptions = TaskSupervisorOptions()
    ):
        """Ask the worker to finalize and exit, then confirm settlement; cancel if it does not exit in time."""
        if not isinstance(session, _Session):
            raise TypeError("session must come from open_session")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("this task supervisor already has an active attempt")
        try:
            with self._state_lock:
                if session.handle is None or session.settled:
                    return None
            if session.session_dir is not None:
                try:
                    _write_json_exclusive(session.session_dir, SESSION_END_MARKER, {})
                except FileExistsError:
                    pass
            try:
                status = self._wait(session, None, options, timeout=timeout)
            except TimeoutError:
                with self._state_lock:
                    self._issues.append(("warning", "task worker did not exit after its group ended; cancelled"))
                status = session.handle.poll()
            self._diagnostic(
                session,
                "session_closed",
                execution_id=session.handle.execution_id,
                exit_code=status.exit_code,
                termination_signal=status.termination_signal,
                failure_reason=status.failure_reason,
            )
            return status
        finally:
            with self._state_lock:
                if session.settled and not self._settlement_unconfirmed:
                    self._active = None
            self._execution_lock.release()
            self.cleanup_job_payloads()

    def _aborted(self, owner, abort_signal):
        with self._state_lock:
            return (
                self._stopping
                or owner.cancel_requested.is_set()
                or (abort_signal is not None and abort_signal.triggered)
            )

    def _wait(self, owner, abort_signal, options, deadline_provider=None, *, done_path=None, timeout=None):
        """Wait until the worker exits (returning its settled status) or, with ``done_path``, publishes it (None)."""
        limit = options.worker_timeout if timeout is None else timeout
        deadline = None if limit is None else time.monotonic() + limit
        while True:
            if done_path is not None and os.path.lexists(done_path):
                return None
            if self._aborted(owner, abort_signal):
                return self._cancel(owner)
            remaining = self._remaining(deadline, deadline_provider)
            if remaining is not None and remaining <= 0:
                self._cancel(owner)
                raise TimeoutError("task worker timed out")
            status = owner.handle.poll()
            if status.phase == TaskExecutionPhase.TERMINAL:
                return self._settle(owner, options, remaining)
            owner.cancel_requested.wait(
                options.poll_interval if remaining is None else min(options.poll_interval, remaining)
            )

    def _settle(self, owner, options, remaining):
        timeout = options.settlement_timeout if remaining is None else min(options.settlement_timeout, remaining)
        try:
            status = owner.handle.wait_for_settlement(timeout=timeout)
        except TimeoutError:
            self._cancel(owner)
            raise
        except TaskSettlementError:
            self._fence(owner)
            raise
        return self._confirm_settlement(owner, status)

    @staticmethod
    def _remaining(deadline, deadline_provider):
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
        return remaining

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
                or not attempt.published
                or not attempt.validated
                or attempt.releasing
                or attempt.released
            ):
                raise RuntimeError("task attempt has no retained, published result to resubmit")
        result, completion = attempt.store.read_result(identity)
        if completion != attempt.completion:
            raise ValueError("saved task completion changed after validation")
        return result

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
                    or not attempt.published
                    or not attempt.validated
                    or attempt.failed
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
