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

"""Client Job adapter for the role-neutral disposable-task supervisor.

This runtime-owned adapter is not an application component or extension point.
Sites customize the injected TaskLauncherSpec, not this supervisor.
"""

import copy
import json
import os
import re
import sys
import threading
from typing import Optional

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.task_execution import TaskArtifactCleanup
from nvflare.apis.task_launcher_spec import TaskLauncherSpec, TaskLaunchRequest, TaskResourceRequest
from nvflare.apis.task_state import TaskState
from nvflare.apis.utils.analytix_utils import create_analytic_dxo, send_analytic_dxo
from nvflare.private.fed.task_worker import FileTaskArtifactStore, TaskAttemptIdentity, WorkerBootstrap
from nvflare.private.fed.task_worker.state import FileTaskStateStore
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions

_RUNTIME_DIR = ".nvflare/task-execution"
_WORKER_MODULE = "nvflare.private.fed.app.client.task_worker_process"
_ATTEMPT_DIR = "attempts"
_TASK_ATTEMPT_IDENTITY = "__task_worker_attempt_identity__"
_GPU_ENVIRONMENT_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
}
_PASSTHROUGH_ENVIRONMENT_NAMES = {
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "FL_LOG_LEVEL",
    "NVFLARE_SECURE_LOGGING",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "TEMP",
    "TMP",
    "TMPDIR",
    "VIRTUAL_ENV",
}


class TaskWorkerExecutor(Executor):
    """Translate client task/context and publication events for TaskSupervisor.

    The original Executor and its explicitly task-scoped components arrive as
    inert, authorization-checked configuration specs. They are constructed
    in the worker after the input artifact is committed. The CJ retains filters
    and publishes the returned Shareable through ClientRunner.
    """

    def __init__(
        self,
        executor: dict,
        components: list,
        worker_timeout: Optional[float] = None,
        poll_interval: float = 0.05,
        result_wait_timeout: Optional[float] = None,
        state_names=(),
    ):
        super().__init__()
        TaskSupervisorOptions(worker_timeout, poll_interval, result_wait_timeout)

        if not isinstance(executor, dict):
            raise TypeError("task worker executor spec must be a dict")
        if not isinstance(components, list) or not all(isinstance(item, dict) for item in components):
            raise TypeError("task worker component specs must be a list of dicts")

        self.executor_spec = copy.deepcopy(executor)
        self.component_specs = copy.deepcopy(components)
        self.state_names = TaskState.validate_names(state_names)
        self.worker_timeout = None if worker_timeout is None else float(worker_timeout)
        self.result_wait_timeout = result_wait_timeout
        self.poll_interval = float(poll_interval)
        self._launcher = None
        self._environment_variables = ()
        self._state_lock = threading.RLock()
        self._settlement_panic_reported = False
        self._artifact_cleanup = TaskArtifactCleanup.JOB
        self._supervisor = TaskSupervisor()

    @staticmethod
    def validate_environment_variables(names):
        if not isinstance(names, (list, tuple)):
            raise ValueError("task_launcher.environment_variables must be a list of environment variable names")
        forbidden = set(JobProcessEnv.ALL) | _GPU_ENVIRONMENT_NAMES | {"NVFLARE_CLIENT_API_BOOTSTRAP"}
        for name in names:
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError("task_launcher.environment_variables contains an invalid variable name")
            if name in forbidden:
                raise ValueError(f"task_launcher.environment_variables cannot forward protected variable {name!r}")
        return tuple(dict.fromkeys(names))

    def set_task_launcher(
        self, launcher: TaskLauncherSpec, environment_variables=(), artifact_cleanup=TaskArtifactCleanup.JOB
    ):
        """Inject the launcher selected by the trusted site runtime.

        Launcher construction and backend selection deliberately stay outside
        job configuration. All supervisors in one client runtime may share the
        same launcher instance so admission and backend policy remain
        site-owned.
        """
        if not isinstance(launcher, TaskLauncherSpec):
            raise TypeError(f"launcher must be a TaskLauncherSpec but got {type(launcher)}")
        names = self.validate_environment_variables(environment_variables)
        cleanup = TaskArtifactCleanup.validate(artifact_cleanup)
        with self._state_lock:
            if self._launcher is not None and (
                self._launcher is not launcher
                or self._environment_variables != names
                or self._artifact_cleanup != cleanup
            ):
                raise RuntimeError("task launcher has already been configured")
            self._launcher = launcher
            self._environment_variables = names
            self._artifact_cleanup = cleanup
            self._supervisor.artifact_cleanup = cleanup

    def _get_task_launcher(self) -> TaskLauncherSpec:
        with self._state_lock:
            launcher = self._launcher
        if launcher is None:
            raise RuntimeError("no task launcher is configured; the client site runtime must inject a TaskLauncherSpec")
        return launcher

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        try:
            if event_type == EventType.START_RUN:
                self._supervisor.start_run()
                with self._state_lock:
                    self._settlement_panic_reported = False
            elif event_type == EventType.ABORT_TASK:
                self._supervisor.cancel_active()
            elif event_type == EventType.END_RUN:
                self._supervisor.end_run()
            elif event_type == EventType.AFTER_SEND_TASK_RESULT:
                self._record_publication_outcome(fl_ctx)
        finally:
            self._report_supervisor_issues(fl_ctx)

    def _report_supervisor_issues(self, fl_ctx):
        for level, message in self._supervisor.take_issues():
            log = self.log_error if level == "error" else self.log_warning
            log(fl_ctx, message)
        if self._supervisor.settlement_unconfirmed:
            with self._state_lock:
                if fl_ctx is None or self._settlement_panic_reported:
                    return
                self._settlement_panic_reported = True
            self.system_panic("Task worker process settlement is unconfirmed; stopping the job", fl_ctx)

    @staticmethod
    def _worker_environment(environment_variables=()):
        # TaskLauncherSpec treats this as the complete child environment.
        # This first CPU-only slice deliberately masks inherited GPU visibility
        # and drops federation bootstrap credentials.
        allowed = _PASSTHROUGH_ENVIRONMENT_NAMES | set(environment_variables)
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in allowed and name not in _GPU_ENVIRONMENT_NAMES and name not in JobProcessEnv.ALL
        }
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "none",
                "ROCR_VISIBLE_DEVICES": "",
            }
        )
        return environment

    @staticmethod
    def _workspace(fl_ctx: FLContext):
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        return workspace if workspace is not None else fl_ctx.get_workspace()

    @classmethod
    def _runtime_paths(cls, fl_ctx: FLContext):
        workspace = cls._workspace(fl_ctx)
        job_id = fl_ctx.get_job_id()
        runtime_root = os.path.join(workspace.get_run_dir(job_id), _RUNTIME_DIR)
        return runtime_root, os.path.join(runtime_root, _ATTEMPT_DIR)

    @classmethod
    def _validate_cpu_only_runtime(cls, fl_ctx: FLContext):
        workspace = cls._workspace(fl_ctx)
        meta_path = workspace.get_job_meta_path(fl_ctx.get_job_id())
        try:
            with open(meta_path, encoding="utf-8") as stream:
                job_meta = json.load(stream)
        except FileNotFoundError:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META, {})
        if not isinstance(job_meta, dict):
            return

        site_name = fl_ctx.get_identity_name()
        for setting_name, default_key in (("resource_spec", "@default"), ("launcher_spec", "default")):
            effective = cls._effective_site_settings(job_meta.get(setting_name, {}), site_name, default_key)
            gpu_setting = cls._find_nonempty_gpu_setting(effective, setting_name)
            if gpu_setting:
                raise RuntimeError(
                    "execution_lifetime='task' currently supports CPU Process workers only; "
                    f"the effective client job resources request GPU resources at {gpu_setting!r}"
                )

    @classmethod
    def _effective_site_settings(cls, settings, site_name, default_key):
        if not isinstance(settings, dict):
            return {}
        default = settings.get(default_key, {})
        site = settings.get(site_name, {})
        if not isinstance(default, dict):
            default = {}
        if not isinstance(site, dict):
            site = {}
        return cls._merge_settings(default, site)

    @classmethod
    def _merge_settings(cls, default, override):
        result = copy.deepcopy(default)
        for key, value in override.items():
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key] = cls._merge_settings(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    @classmethod
    def _find_nonempty_gpu_setting(cls, value, path="resource_spec"):
        if isinstance(value, dict):
            for key, item in value.items():
                item_path = f"{path}.{key}"
                if "gpu" in str(key).lower() and cls._has_resource_value(item):
                    return item_path
                found = cls._find_nonempty_gpu_setting(item, item_path)
                if found:
                    return found
        elif isinstance(value, list):
            for index, item in enumerate(value):
                found = cls._find_nonempty_gpu_setting(item, f"{path}[{index}]")
                if found:
                    return found
        return None

    @staticmethod
    def _has_resource_value(value):
        if value is None or value is False:
            return False
        if isinstance(value, (int, float)):
            return value > 0
        return bool(value)

    def _attempt_identity(self, task_name, fl_ctx):
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("task execution requires a current task ID")
        attempt_id = fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID)
        if not isinstance(attempt_id, str) or not attempt_id:
            raise RuntimeError("task execution requires a server-issued task attempt ID")
        return TaskAttemptIdentity(
            job_id=fl_ctx.get_job_id(),
            site_name=fl_ctx.get_identity_name(),
            task_id=task_id,
            task_name=task_name,
            attempt_id=attempt_id,
        )

    def _launch_request(self, identity, bootstrap_path, fl_ctx):
        """Trusted client invocation seam; backend-specific packaging stays here."""
        return TaskLaunchRequest(
            job_id=identity.job_id,
            site_name=identity.site_name,
            task_id=identity.task_id,
            attempt_id=identity.attempt_id,
            argv=(
                sys.executable,
                "-m",
                _WORKER_MODULE,
                "--bootstrap",
                bootstrap_path,
                "--parent_pid",
                str(os.getpid()),
            ),
            environment=self._worker_environment(self._environment_variables),
            cwd=self._workspace(fl_ctx).get_run_dir(identity.job_id),
            resources=TaskResourceRequest(),
        )

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not isinstance(shareable, Shareable):
            raise TypeError("task worker input must be a Shareable")
        identity = self._attempt_identity(task_name, fl_ctx)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self._validate_cpu_only_runtime(fl_ctx)
        launcher = self._get_task_launcher()
        runtime_root, artifact_root = self._runtime_paths(fl_ctx)
        bootstrap = WorkerBootstrap(
            identity=identity,
            artifact_root=artifact_root,
            workspace_root=self._workspace(fl_ctx).get_root_dir(),
            executor=self.executor_spec,
            components=self.component_specs,
            state_names=self.state_names,
        )
        state_store = (
            FileTaskStateStore(os.path.join(runtime_root, "state"), self.state_names) if self.state_names else None
        )
        try:
            outcome = self._supervisor.run(
                bootstrap=bootstrap,
                store=FileTaskArtifactStore(artifact_root),
                input_payload=shareable,
                runtime_root=runtime_root,
                launcher=launcher,
                abort_signal=abort_signal,
                request_factory=lambda identity, path: self._launch_request(identity, path, fl_ctx),
                options=TaskSupervisorOptions(self.worker_timeout, self.poll_interval, self.result_wait_timeout),
                state_store=state_store,
            )
            if outcome.aborted:
                return make_reply(ReturnCode.TASK_ABORTED)
            fl_ctx.set_prop(_TASK_ATTEMPT_IDENTITY, outcome.identity, private=True, sticky=False)
            self._replay_analytics(outcome.analytics, fl_ctx)
            return outcome.result
        finally:
            self._report_supervisor_issues(fl_ctx)

    def _replay_analytics(self, records, fl_ctx):
        for message in records:
            try:
                message = dict(message)
                if "event_type" in message:
                    if message["event_type"] != ANALYTIC_EVENT_TYPE or not isinstance(message["federated"], bool):
                        raise ValueError("unsupported task worker analytics event")
                    send_analytic_dxo(
                        self,
                        from_shareable(message["data"]),
                        fl_ctx,
                        event_type=message["event_type"],
                        fire_fed_event=message["federated"],
                    )
                else:
                    message["tag"] = message.pop("key")
                    dxo = create_analytic_dxo(**message)
                    send_analytic_dxo(self, dxo, fl_ctx, event_type=ANALYTIC_EVENT_TYPE, fire_fed_event=False)
            except Exception as e:
                self.log_error(fl_ctx, f"failed to emit task worker analytics: {e}")

    def _record_publication_outcome(self, fl_ctx: FLContext):
        identity = fl_ctx.get_prop(_TASK_ATTEMPT_IDENTITY)
        if not isinstance(identity, TaskAttemptIdentity):
            return
        if (
            identity.job_id != fl_ctx.get_job_id()
            or identity.site_name != fl_ctx.get_identity_name()
            or identity.task_id != fl_ctx.get_prop(FLContextKey.TASK_ID)
            or identity.attempt_id != fl_ctx.get_prop(FLContextKey.TASK_ATTEMPT_ID)
        ):
            return
        effective_result = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
        result_succeeded = effective_result is None or (
            isinstance(effective_result, Shareable)
            and effective_result.get_return_code(default=ReturnCode.OK) == ReturnCode.OK
        )
        try:
            self._supervisor.acknowledge(
                identity,
                sent=fl_ctx.get_prop(FLContextKey.TASK_RESULT_SEND_SUCCESS) is True,
                admitted=fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED),
                result_succeeded=result_succeeded,
            )
        except Exception as e:
            self.system_panic(f"Failed to promote admitted task worker state: {e}", fl_ctx)
            raise
