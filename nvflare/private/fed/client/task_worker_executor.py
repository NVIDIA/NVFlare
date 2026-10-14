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
import os
import sys
import threading
from functools import partial

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.task_launcher_spec import TaskLaunchRequest
from nvflare.apis.utils.analytix_utils import send_analytic_dxo
from nvflare.app_common.executors.client_api_executor import FED_ANALYTIC_EVENT_TYPE
from nvflare.job_config.task_execution import validate_task_resources
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions

from .client_api_task_runner import result_wait_deadline
from .task_execution_policy import TaskExecutionPolicy

TASK_ATTEMPT_IDENTITY = "__task_worker_attempt_identity__"


class TaskWorkerExecutor(Executor):
    """Trusted runtime adapter constructed from authorized inert application specs."""

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
        self._ending = False
        self._panic_reported = False

    def handle_event(self, event_type, fl_ctx):
        try:
            if event_type == EventType.START_RUN:
                with self._state_lock:
                    if self._ending:
                        raise RuntimeError("task worker owner cannot restart after ending")
                    self._started = True
            elif event_type == EventType.ABORT_TASK:
                self.supervisor.cancel_active()
            elif event_type in (EventType.ABOUT_TO_END_RUN, EventType.END_RUN):
                with self._state_lock:
                    self._ending = True
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

    def _report_issues(self, fl_ctx):
        for level, message in self.supervisor.take_issues():
            (self.log_error if level == "error" else self.log_warning)(fl_ctx, message)
        panic = False
        with self._state_lock:
            if self.supervisor.settlement_unconfirmed and not self._panic_reported:
                self._panic_reported = panic = True
        if panic:
            self.system_panic("task worker settlement is unconfirmed; retaining ownership and stopping the job", fl_ctx)

    @staticmethod
    def _identity(task_name, fl_ctx):
        return TaskAttemptIdentity(
            fl_ctx.get_job_id(),
            fl_ctx.get_identity_name(),
            fl_ctx.get_prop(FLContextKey.TASK_ID),
            task_name,
            fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID),
        )

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("task worker executor already has an active attempt")
        self._done.clear()
        identity = workspace = None
        try:
            with self._state_lock:
                if not self._started or self._ending:
                    raise RuntimeError("task worker executor is not running")
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            identity = self._identity(task_name, fl_ctx)
            validate_task_resources(fl_ctx.get_prop(FLContextKey.JOB_META, {}) or {}, identity.site_name)
            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT) or fl_ctx.get_workspace()
            runtime_root = os.path.join(workspace.get_run_dir(identity.job_id), ".nvflare", "task-execution")
            store = FileTaskArtifactStore(
                os.path.join(runtime_root, "attempts"), max_payload_bytes=self.policy.max_payload_bytes
            )
            bootstrap = WorkerBootstrap(
                identity,
                store.root_dir,
                workspace.get_root_dir(),
                self.executor_spec,
                self.component_specs,
                max_payload_bytes=self.policy.max_payload_bytes,
            )

            def request_factory(attempt, path):
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
                )

            outcome = self.supervisor.run(
                bootstrap=bootstrap,
                store=store,
                input_payload=shareable,
                runtime_root=runtime_root,
                launcher=self.policy.launcher,
                abort_signal=abort_signal,
                request_factory=request_factory,
                options=self.options,
                deadline_provider=partial(
                    result_wait_deadline, store, identity, self.result_wait_timeout, self.policy.finalization_timeout
                ),
            )
            if outcome.aborted or abort_signal.triggered:
                self.supervisor.retain_attempt(identity)
                return make_reply(ReturnCode.TASK_ABORTED)
            fl_ctx.set_prop(TASK_ATTEMPT_IDENTITY, identity, private=True, sticky=False)
            self._replay_analytics(outcome.analytics, fl_ctx)
            return outcome.result
        finally:
            if (
                workspace is not None
                and not self.supervisor.owns_execution
                and not self.supervisor.settlement_unconfirmed
            ):
                self._replay_worker_log(workspace, identity, fl_ctx)
            self._done.set()
            self._gate.release()
            self._report_issues(fl_ctx)

    def _replay_worker_log(self, workspace, identity, fl_ctx):
        """Make worker diagnostics visible to the existing CJ log streamer."""
        path = os.path.join(workspace.get_log_root(identity.job_id), f"task_{identity.attempt_id}_log.txt")
        try:
            with open(path, encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    self.log_info(fl_ctx, f"[task {identity.attempt_id}] {line.rstrip()}")
        except FileNotFoundError:
            pass
        except Exception as e:
            self.log_warning(fl_ctx, f"could not replay task worker log {path}: {e}")

    def _replay_analytics(self, records, fl_ctx):
        for record in records:
            if record.get("event_type") not in (ANALYTIC_EVENT_TYPE, FED_ANALYTIC_EVENT_TYPE) or not isinstance(
                record.get("federated"), bool
            ):
                raise ValueError("unsupported task worker analytics event")
            send_analytic_dxo(
                self,
                from_shareable(record["data"]),
                fl_ctx,
                event_type=record["event_type"],
                fire_fed_event=record["federated"],
            )

    def _record_receipt(self, fl_ctx):
        identity = fl_ctx.get_prop(TASK_ATTEMPT_IDENTITY)
        if not isinstance(identity, TaskAttemptIdentity):
            return
        expected = (identity.job_id, identity.site_name, identity.task_id, identity.task_name, identity.attempt_id)
        actual = (
            fl_ctx.get_job_id(),
            fl_ctx.get_identity_name(),
            fl_ctx.get_prop(FLContextKey.TASK_ID),
            fl_ctx.get_prop(FLContextKey.TASK_NAME),
            fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID),
        )
        if actual != expected:
            return
        result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
        if not isinstance(result, Shareable) or result.get_return_code(default=ReturnCode.OK) != ReturnCode.OK:
            self.supervisor.retain_attempt(identity)
        self.supervisor.record_receipt(identity, fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT))
