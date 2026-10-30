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

"""CJ-owned adapter: filters and publication stay in the resident Client Job."""

import copy
import hashlib
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from functools import partial

from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode, ServerCommandNames
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.task_launcher_spec import TaskLaunchRequest
from nvflare.apis.utils.analytix_utils import send_analytic_dxo
from nvflare.apis.utils.task_utils import materialize_lazy_download_refs
from nvflare.app_common.utils.tensor_disk_offload_context import _TENSOR_DISK_OFFLOAD_ROOT_DIR
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.private.defs import CellChannel, RequestHeader
from nvflare.private.fed.task_worker import cell_session
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import (
    TaskAttemptIdentity,
    WorkerBootstrap,
    _open_directory,
    _open_regular,
)
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions

from .client_api_task_runner import result_wait_deadline
from .task_execution_policy import TaskExecutionPolicy, is_native_task_launcher

TASK_ATTEMPT_IDENTITY = "__task_worker_attempt_identity__"
_TASK_ROUTE = (CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK)


def _workspace(fl_ctx):
    return fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT) or fl_ctx.get_workspace()


class TaskWorkerExecutor(Executor):
    """Trusted runtime adapter constructed from authorized inert application specs.

    Without a task group every assignment is one worker process (#5405). Inside a
    group one worker session serves the group's assignments until the group closes,
    the job ends, or a different group arrives. A worker that dies inside a group is
    not replaced; the group's remaining assignments fail on this client.
    """

    def __init__(self, executor, components, policy, *, worker_timeout=None, result_wait_timeout=None):
        super().__init__()
        if not isinstance(policy, TaskExecutionPolicy):
            raise TypeError("task adapter requires trusted site policy")
        self.executor_spec = copy.deepcopy(executor)
        self.component_specs = copy.deepcopy(components)
        self.policy = policy
        limits = [value for value in (worker_timeout, policy.worker_timeout) if value is not None]
        self.options = TaskSupervisorOptions(
            worker_timeout=min(limits) if limits else None,
            settlement_timeout=policy.settlement_timeout,
            poll_interval=policy.poll_interval,
        )
        self.result_wait_timeout = result_wait_timeout
        self.supervisor = TaskSupervisor(retain_artifacts=policy.artifact_cleanup == "retain")
        self._gate = threading.Lock()
        self._state_lock = threading.Lock()
        self._done = threading.Event()
        self._done.set()
        self._started = False
        self._start_error = None
        self._ending = False
        self._panic_reported = False
        self._completed_attempts = {}
        self._session = None
        self._session_ctx = None
        self._ended_groups = set()
        self._metrics = None
        self._attempt_ctx = None

    def handle_event(self, event_type, fl_ctx):
        try:
            if event_type == EventType.START_RUN:
                with self._state_lock:
                    if self._ending:
                        raise RuntimeError("task worker owner cannot restart after ending")
                    cell = fl_ctx.get_engine().get_cell()
                    if cell is None:
                        raise RuntimeError("task lifetime requires a Cell from the engine")
                    self._started = False
                try:
                    self.policy.launcher.handle_event(event_type, fl_ctx)
                    # Workers join CJ with site internal security and the
                    # launcher's worker-reachable address on CJ's own port.
                    cell_session.listen(cell, self.policy.launcher.get_worker_listener_host())
                    cell.decode_pass_through_topics.add(_TASK_ROUTE)
                except Exception as error:
                    self._start_error = str(error)
                    self.system_panic(f"task launcher START_RUN failed: {error}", fl_ctx)
                    raise
                with self._state_lock:
                    self._started = True
                    self._start_error = None
            elif event_type == EventType.ABORT_TASK:
                self.supervisor.cancel_active()
            elif event_type == EventType.TASK_GROUP_CLOSED:
                self._close_session_if(group_id=fl_ctx.get_prop(FLContextKey.TASK_GROUP_ID))
            elif event_type in (EventType.ABOUT_TO_END_RUN, EventType.END_RUN):
                if event_type == EventType.END_RUN:
                    fl_ctx.get_engine().get_cell().decode_pass_through_topics.discard(_TASK_ROUTE)
                with self._state_lock:
                    self._ending = True
                self._close_session_if()
                parent_owned = getattr(self.policy.launcher, "cleanup_owned_by_parent", False) is True
                if event_type == EventType.END_RUN and parent_owned:
                    # CP already owns every accepted request, including lost replies
                    # and in-flight creates. CJ exit cannot discard that owner.
                    try:
                        self.policy.launcher.handle_event(event_type, fl_ctx)
                    finally:
                        self.supervisor.end_run()
                elif event_type == EventType.END_RUN and is_native_task_launcher(self.policy.launcher):
                    self._end_native_run(fl_ctx)
                else:
                    self.supervisor.end_run()
                    if (
                        event_type == EventType.END_RUN
                        and not self._done.wait(self.policy.settlement_timeout)
                        and self.supervisor.owns_execution
                    ):
                        self.system_panic("task worker launch/execution is still owned during job shutdown", fl_ctx)
            elif event_type == EventType.CHECK_END_RUN_READINESS:
                if not self._done.is_set() or self.supervisor.settlement_unconfirmed:
                    fl_ctx.set_prop(FLContextKey.NOT_READY_TO_END_RUN, True, private=True, sticky=False)
            elif event_type == EventType.AFTER_SEND_TASK_RESULT:
                self._record_receipt(fl_ctx)
        finally:
            self._report_issues(fl_ctx)

    def _end_native_run(self, fl_ctx):
        # Readiness has a finite budget and event dispatch catches exceptions.
        # The native cleanup owner must therefore hold END_RUN itself until a
        # late launch returns and physical settlement has actually been proved.
        while True:
            try:
                self.supervisor.end_run()
            except Exception as error:
                self.log_warning(fl_ctx, f"native task cleanup failed; retaining CJ and retrying: {error}")
            self._report_issues(fl_ctx)
            if not self.supervisor.owns_execution and self._done.is_set():
                return
            time.sleep(self.policy.poll_interval)

    def _report_issues(self, fl_ctx):
        for level, message in self.supervisor.take_issues():
            (self.log_error if level == "error" else self.log_warning)(fl_ctx, message)
        panic = False
        with self._state_lock:
            if self.supervisor.settlement_unconfirmed and not self._panic_reported:
                self._panic_reported = panic = True
        if panic:
            self.system_panic("task worker settlement is unconfirmed; retaining ownership and stopping the job", fl_ctx)

    def _attempt_id(self, fl_ctx):
        attempt_id = fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID)
        launch_id = ConfigService.get_str_var(RequestHeader.CJ_LAUNCH_ID)
        if is_native_task_launcher(self.policy.launcher) and launch_id and isinstance(attempt_id, str) and attempt_id:
            # The server assignment survives CJ recovery; each physical worker
            # needs a fresh identity after CP confirms prior worker shutdown.
            # Process keeps its existing artifact fence: CJ group exit alone
            # cannot establish that a separately grouped task worker stopped.
            return hashlib.sha256(json.dumps([launch_id, attempt_id]).encode()).hexdigest()
        return attempt_id

    def _identity(self, task_name, fl_ctx):
        return TaskAttemptIdentity(
            fl_ctx.get_job_id(),
            fl_ctx.get_identity_name(),
            fl_ctx.get_prop(FLContextKey.TASK_ID),
            task_name,
            self._attempt_id(fl_ctx),
        )

    @staticmethod
    def _runtime_root(workspace, job_id):
        return os.path.join(workspace.get_run_dir(job_id), ".nvflare", "task-execution")

    def resolve_task_data(self, data, fl_ctx, *, abort_signal):
        cell = fl_ctx.get_engine().get_cell()
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT) or fl_ctx.get_workspace()
        root = os.path.realpath(os.path.join(self._runtime_root(workspace, fl_ctx.get_job_id()), "offload"))
        os.close(_open_directory(root, create=True))
        owners = []

        def release():
            for owner in owners:
                owner.cleanup()

        try:
            resolved = materialize_lazy_download_refs(
                data,
                cell,
                abort_signal,
                download_props={FOBSContextKey.TENSOR_DISK_OFFLOAD: True, _TENSOR_DISK_OFFLOAD_ROOT_DIR: root},
                owners=owners,
            )
        except BaseException:
            release()
            raise
        return resolved, release

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("task worker executor already has an active attempt")
        self._done.clear()
        identity = workspace = outcome = None
        group_id = None
        metrics = None
        try:
            with self._state_lock:
                if not self._started or self._ending:
                    raise RuntimeError(self._start_error or "task worker executor is not running")
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            identity = self._identity(task_name, fl_ctx)
            if identity in self._completed_attempts:
                saved_identity = self._completed_attempts[identity]
                result = self.supervisor.read_saved_result(saved_identity)
                with self._checkpoint_commit_guard(abort_signal):
                    fl_ctx.set_prop(TASK_ATTEMPT_IDENTITY, saved_identity, private=True, sticky=False)
                    return result
            self.log_info(
                fl_ctx,
                f"Task {identity.task_id}: server attempt {fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID)}, "
                f"worker attempt {identity.attempt_id}",
            )
            resources = self.policy.task_resources(fl_ctx.get_prop(FLContextKey.JOB_META, {}) or {}, identity.site_name)
            workspace = _workspace(fl_ctx)
            runtime_root = self._runtime_root(workspace, identity.job_id)
            store = FileTaskArtifactStore(
                os.path.join(runtime_root, "attempts"), max_payload_bytes=self.policy.max_payload_bytes
            )
            checkpoint_key = hashlib.sha256(
                json.dumps([self.executor_spec, task_name], sort_keys=True).encode("utf-8")
            ).hexdigest()
            checkpoint_file = os.path.join(runtime_root, "checkpoints", checkpoint_key + ".checkpoint")
            assignment = None
            if is_native_task_launcher(self.policy.launcher):
                assignment = replace(identity, attempt_id=fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID))
                outcome = self.supervisor.recover_checkpoint_result(assignment, store, checkpoint_file, runtime_root)
                if outcome is not None:
                    self._completed_attempts[identity] = outcome.identity
                    with self._checkpoint_commit_guard(abort_signal):
                        fl_ctx.set_prop(TASK_ATTEMPT_IDENTITY, outcome.identity, private=True, sticky=False)
                        return outcome.result

            stage_hook = partial(store.prepare_checkpoint, checkpoint_file=checkpoint_file)
            request_factory = partial(self._launch_request, workspace, resources)
            deadline_provider = partial(
                result_wait_deadline, store, identity, self.result_wait_timeout, self.policy.finalization_timeout
            )
            group_id = fl_ctx.get_prop(FLContextKey.TASK_GROUP_ID)
            if group_id is None:
                session = self.supervisor.open_session(None, store, runtime_root, self.policy.launcher, request_factory)
            else:
                session = self._session_for(group_id, store, runtime_root, request_factory, fl_ctx)
            if group_id is None:
                # Metrics logged by the worker are fired into the job while the attempt is still owned.
                metrics = cell_session.CellWorkerSession(fl_ctx.get_engine().get_cell(), self._fire_current_analytics)
            else:
                metrics = self._metrics
            bootstrap = WorkerBootstrap(
                identity,
                store.root_dir,
                workspace.get_root_dir(),
                self.executor_spec,
                self.component_specs,
                max_payload_bytes=self.policy.max_payload_bytes,
                session_dir=session.session_dir,
                cell_session=metrics.bootstrap_fields(),
            )
            with self._state_lock:
                self._attempt_ctx = fl_ctx
            outcome = self.supervisor.run_attempt(
                session, bootstrap, shareable, abort_signal, self.options, deadline_provider, stage_hook
            )
            # Finish fallible CJ replay/reporting before promoting application
            # state. The success path's finally block only releases ownership.
            self._report_issues(fl_ctx)
            if group_id is None and not self.supervisor.owns_execution and not self.supervisor.settlement_unconfirmed:
                self._replay_worker_log(workspace, identity, fl_ctx)
            if outcome.aborted or abort_signal.triggered:
                self.supervisor.retain_attempt(identity)
                return make_reply(ReturnCode.TASK_ABORTED)
            fl_ctx.set_prop(TASK_ATTEMPT_IDENTITY, identity, private=True, sticky=False)
            if outcome.result.get_return_code(default=ReturnCode.OK) == ReturnCode.OK:
                store.commit_checkpoint(
                    outcome.completion,
                    checkpoint_file,
                    publication_guard=partial(self._checkpoint_commit_guard, abort_signal),
                    assignment=assignment,
                )
            self._completed_attempts[identity] = identity
            return outcome.result
        finally:
            if group_id is None and metrics is not None:
                # The worker has settled: a record arriving now belongs to no owned attempt.
                metrics.close(timeout=self.policy.settlement_timeout)
            try:
                if group_id is not None:
                    self._settle_session(fl_ctx)
                elif (
                    outcome is None
                    and workspace is not None
                    and not self.supervisor.owns_execution
                    and not self.supervisor.settlement_unconfirmed
                ):
                    self._replay_worker_log(workspace, identity, fl_ctx)
            finally:
                self._done.set()
                self._gate.release()
                if outcome is None:
                    self._report_issues(fl_ctx)

    def _launch_request(self, workspace, resources, attempt, path):
        return TaskLaunchRequest(
            attempt.job_id,
            attempt.site_name,
            attempt.task_id,
            attempt.attempt_id,
            (
                sys.executable,
                "-m",
                "nvflare.private.fed.app.client.task_worker_process",
                "--bootstrap",
                path,
                "--parent_pid",
                str(os.getpid()),
            ),
            environment=self.policy.worker_environment(),
            cwd=workspace.get_run_dir(attempt.job_id),
            resources=resources,
        )

    # ---- task group sessions ----

    def _session_for(self, group_id, store, runtime_root, request_factory, fl_ctx):
        """Return the open session for this group, closing a session of another group first.

        A group's worker is never replaced: executors keep state in memory across the
        group's assignments, so once its session has ended for any reason (worker death,
        abort, closure, or another group taking over) the group's later assignments fail
        on this client instead of silently running with fresh memory.
        """
        if group_id in self._ended_groups:
            raise RuntimeError(f"task group {group_id} already ended on this client; its later assignments cannot run")
        session = self._session
        if session is not None and (session.group_id != group_id or not self._reusable(session)):
            self._close_session(fl_ctx)
            session = None
            if group_id in self._ended_groups:
                raise RuntimeError(
                    f"task group {group_id} lost its worker on this client; its later assignments cannot run"
                )
        if session is None:
            session = self.supervisor.open_session(group_id, store, runtime_root, self.policy.launcher, request_factory)
            with self._state_lock:
                self._session = session
                self._session_ctx = fl_ctx
                self._metrics = cell_session.CellWorkerSession(
                    fl_ctx.get_engine().get_cell(), self._fire_current_analytics
                )
            self.log_info(fl_ctx, f"opened task worker session for group {group_id}")
        return session

    @staticmethod
    def _reusable(session):
        return not (session.settled or session.cancel_requested.is_set())

    def _settle_session(self, fl_ctx):
        """After a group attempt, a session whose worker is gone ends the group on this client."""
        with self._state_lock:
            session = self._session
        if session is not None and not self._reusable(session):
            self._close_session(fl_ctx)

    def _close_session_if(self, group_id=None):
        """Close the open session when it serves ``group_id``, or any session when no group is named."""
        with self._state_lock:
            session = self._session
        if session is None or (group_id is not None and session.group_id != group_id):
            return
        # A running attempt holds the gate; its own settlement path drops a dead session.
        if self._gate.acquire(timeout=self.policy.finalization_timeout):
            try:
                if self._session is session:
                    self._close_session(self._session_ctx)
            finally:
                self._gate.release()

    def _close_session(self, fl_ctx):
        """Caller holds the gate. The session is dropped only once its worker has exited."""
        with self._state_lock:
            session = self._session
        if session is None:
            return
        try:
            status = self.supervisor.close_session(session, self.policy.finalization_timeout, self.options)
            if status is not None and not status.succeeded:
                self.log_warning(fl_ctx, f"task worker session for group {session.group_id} ended abnormally: {status}")
        finally:
            with self._state_lock:
                self._session = None
                self._ended_groups.add(session.group_id)
                metrics, self._metrics = self._metrics, None
            if metrics is not None:
                metrics.close(timeout=self.policy.settlement_timeout)
            self._report_issues(fl_ctx)
            if session.identity is not None and not self.supervisor.settlement_unconfirmed:
                self._replay_worker_log(_workspace(fl_ctx), session.identity, fl_ctx)

    @contextmanager
    def _checkpoint_commit_guard(self, abort_signal):
        with self._state_lock:
            if self._ending or abort_signal.triggered or self.supervisor.settlement_unconfirmed:
                raise RuntimeError("task checkpoint commit was cancelled")
            yield

    def _replay_worker_log(self, workspace, identity, fl_ctx):
        """Replay bounded diagnostics from a regular file to the CJ log streamer."""
        log_root = os.path.realpath(workspace.get_log_root(identity.job_id))
        path = os.path.join(log_root, f"task_{identity.attempt_id}_log.txt")
        max_bytes, max_lines = 1024 * 1024, 4096
        try:
            # The worker controls this file. Reject links/devices/FIFOs without
            # blocking and bound reads even when a file keeps growing.
            with _open_regular(path) as stream:
                data = stream.read(max_bytes + 1)
            lines = data[:max_bytes].decode("utf-8", errors="replace").splitlines()
            for line in lines[:max_lines]:
                self.log_info(fl_ctx, f"[task {identity.attempt_id}] {line.rstrip()}")
            if len(data) > max_bytes or len(lines) > max_lines:
                self.log_warning(fl_ctx, f"task worker log replay truncated; full log remains at {path}")
        except FileNotFoundError:
            pass
        except Exception as e:
            self.log_warning(fl_ctx, f"could not replay task worker log {path}: {e}")

    def _fire_current_analytics(self, data):
        """A group worker outlives attempts; its metrics belong to whichever attempt is owned now."""
        with self._state_lock:
            fl_ctx = self._attempt_ctx
        if fl_ctx is None:
            raise RuntimeError("no owned attempt to fire task worker analytics into")
        self._fire_analytics(fl_ctx, data)

    def _fire_analytics(self, fl_ctx, data):
        """Fire one metric the worker logged; runs on a cell thread while ``execute`` owns the attempt.

        Raises only for a metric it cannot fire. Receiver failures are logged and recorded on ``fl_ctx`` by event
        dispatch, as for a resident executor in job lifetime, so a failing tracking backend never fails training.
        """
        if not isinstance(data, Shareable):
            raise ValueError("task worker analytics must be a DXO Shareable")
        send_analytic_dxo(self, from_shareable(data), fl_ctx)

    def _record_receipt(self, fl_ctx):
        identity = fl_ctx.get_prop(TASK_ATTEMPT_IDENTITY)
        if not isinstance(identity, TaskAttemptIdentity):
            return
        current = self._identity(fl_ctx.get_prop(FLContextKey.TASK_NAME), fl_ctx)
        if self._completed_attempts.get(current, current) != identity:
            return
        result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
        if not isinstance(result, Shareable) or result.get_return_code(default=ReturnCode.OK) != ReturnCode.OK:
            self.supervisor.retain_attempt(identity)
        self.supervisor.record_receipt(identity, fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT))
