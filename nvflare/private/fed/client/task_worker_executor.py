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

"""Internal CJ supervisor for one fresh application process per task.

This runtime-owned adapter is not an application component or extension point.
Sites customize the injected TaskLauncherSpec, not this supervisor.
"""

import copy
import json
import math
import os
import re
import sys
import threading
import time
import uuid
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
from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskLauncherSpec, TaskLaunchRequest, TaskResourceRequest
from nvflare.apis.utils.analytix_utils import create_analytic_dxo, send_analytic_dxo
from nvflare.private.fed.task_worker import FileTaskArtifactStore, TaskAttemptIdentity, WorkerBootstrap, write_bootstrap
from nvflare.private.fed.task_worker.protocol import WORKER_MODULE

_RUNTIME_DIR = ".nvflare/task-execution"
_ATTEMPT_DIR = "attempts"
_DIAGNOSTICS_FILE = "diagnostics.jsonl"
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
    """Stage, launch, settle, and validate one application task attempt.

    The original Executor and its transitive component dependencies arrive as
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
    ):
        super().__init__()
        if worker_timeout is not None and (
            isinstance(worker_timeout, bool)
            or not isinstance(worker_timeout, (int, float))
            or not math.isfinite(worker_timeout)
            or worker_timeout < 0
        ):
            raise ValueError("worker_timeout must be a finite nonnegative number or None")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(poll_interval)
            or poll_interval <= 0
        ):
            raise ValueError("poll_interval must be a finite positive number")

        if not isinstance(executor, dict):
            raise TypeError("task worker executor spec must be a dict")
        if not isinstance(components, list) or not all(isinstance(item, dict) for item in components):
            raise TypeError("task worker component specs must be a list of dicts")

        self.executor_spec = copy.deepcopy(executor)
        self.component_specs = copy.deepcopy(components)
        self.worker_timeout = None if worker_timeout is None else float(worker_timeout)
        if result_wait_timeout is not None and (
            isinstance(result_wait_timeout, bool)
            or not isinstance(result_wait_timeout, (int, float))
            or not math.isfinite(result_wait_timeout)
            or result_wait_timeout < 0
        ):
            raise ValueError("result_wait_timeout must be a finite nonnegative number or None")
        self.result_wait_timeout = result_wait_timeout
        self.poll_interval = float(poll_interval)
        self._launcher = None
        self._environment_variables = ()
        self._execution_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._diagnostic_lock = threading.Lock()
        self._active_handle = None
        self._active_abort_signal = None
        self._stopping = False
        self._settlement_panic_reported = False
        self._pending_publication = {}
        self._artifact_cleanup = TaskArtifactCleanup.JOB
        self._retained_attempts = {}
        self._job_cleanup_requested = False

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

    def _get_task_launcher(self) -> TaskLauncherSpec:
        with self._state_lock:
            launcher = self._launcher
        if launcher is None:
            raise RuntimeError("no task launcher is configured; the client site runtime must inject a TaskLauncherSpec")
        return launcher

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            with self._state_lock:
                self._stopping = False
                self._settlement_panic_reported = False
                self._job_cleanup_requested = False
        elif event_type == EventType.ABORT_TASK:
            self._cancel_active(fl_ctx)
        elif event_type == EventType.END_RUN:
            with self._state_lock:
                self._stopping = True
                self._job_cleanup_requested = True
            self._cancel_active(fl_ctx)
            self._cleanup_job_payloads(fl_ctx)
        elif event_type == EventType.AFTER_SEND_TASK_RESULT:
            self._record_publication_outcome(fl_ctx)

    def _cancel_active(self, fl_ctx=None):
        with self._state_lock:
            handle = self._active_handle
            if self._active_abort_signal is not None:
                self._active_abort_signal.trigger(True)
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                self._fail_unconfirmed_settlement(fl_ctx)
                raise

    def _fail_unconfirmed_settlement(self, fl_ctx):
        with self._state_lock:
            self._stopping = True
            if fl_ctx is None or self._settlement_panic_reported:
                return
            self._settlement_panic_reported = True
        self.system_panic("Task worker process settlement is unconfirmed; stopping the job", fl_ctx)

    def _cleanup_job_payloads(self, fl_ctx):
        """Release owned attempts only after END_RUN and all worker/read activity settles.

        END_RUN can race execution. In that case execute's finally block retries
        cleanup after releasing the gate. An uninspectable or live worker keeps
        its active handle and blocks cleanup; no workspace-wide scan is used.
        """
        if self._artifact_cleanup != TaskArtifactCleanup.JOB or not self._job_cleanup_requested:
            return
        if not self._execution_lock.acquire(blocking=False):
            return
        try:
            with self._state_lock:
                if self._active_handle is not None:
                    return
                attempts = list(self._retained_attempts.items())
            for attempt_id, (identity, store) in attempts:
                try:
                    store.release_payloads(identity)
                except Exception as e:
                    self.log_warning(fl_ctx, f"failed to remove job-ended task worker artifacts: {e}")
                else:
                    with self._state_lock:
                        self._retained_attempts.pop(attempt_id, None)
        finally:
            self._execution_lock.release()

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

    def _append_diagnostic(self, runtime_root: str, record: dict):
        os.makedirs(runtime_root, mode=0o700, exist_ok=True)
        path = os.path.join(runtime_root, _DIAGNOSTICS_FILE)
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

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not isinstance(shareable, Shareable):
            raise TypeError("task worker input must be a Shareable")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("this task supervisor already has an active task worker")
        with self._state_lock:
            self._active_abort_signal = abort_signal
        try:
            return self._execute(task_name, shareable, fl_ctx, abort_signal)
        finally:
            with self._state_lock:
                self._active_abort_signal = None
            self._execution_lock.release()
            self._cleanup_job_payloads(fl_ctx)

    def _execute(self, task_name, shareable, fl_ctx, abort_signal):
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("task execution requires a current task ID")
        with self._state_lock:
            if self._stopping or abort_signal.triggered:
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                raise RuntimeError("Client Job is stopping and rejects a new task worker")
        self._validate_cpu_only_runtime(fl_ctx)
        launcher = self._get_task_launcher()
        if self.worker_timeout == 0:
            raise TimeoutError("task worker result timeout expired before launch")

        job_id = fl_ctx.get_job_id()
        site_name = fl_ctx.get_identity_name()
        identity = TaskAttemptIdentity(
            job_id=job_id,
            site_name=site_name,
            task_id=task_id,
            task_name=task_name,
            attempt_id=uuid.uuid4().hex,
        )
        runtime_root, artifact_root = self._runtime_paths(fl_ctx)
        store = FileTaskArtifactStore(artifact_root)
        store.create_attempt(identity)
        with self._state_lock:
            self._retained_attempts[identity.attempt_id] = (identity, store)
        store.write_input(identity, shareable)
        bootstrap = WorkerBootstrap(
            identity=identity,
            artifact_root=artifact_root,
            workspace_root=self._workspace(fl_ctx).get_root_dir(),
            executor=self.executor_spec,
            components=self.component_specs,
        )
        bootstrap_path = store.bootstrap_path(identity)
        write_bootstrap(bootstrap_path, bootstrap)

        diagnostic = self._new_diagnostic(identity)
        diagnostic["launcher_class"] = f"{type(launcher).__module__}.{type(launcher).__qualname__}"
        diagnostic["launcher_mode"] = launcher.launch_mode
        diagnostic["artifact_cleanup"] = self._artifact_cleanup
        request = TaskLaunchRequest(
            job_id=job_id,
            site_name=site_name,
            task_id=task_id,
            attempt_id=identity.attempt_id,
            argv=(sys.executable, "-m", WORKER_MODULE, "--bootstrap", bootstrap_path, "--parent_pid", str(os.getpid())),
            environment=self._worker_environment(self._environment_variables),
            cwd=self._workspace(fl_ctx).get_run_dir(job_id),
            resources=TaskResourceRequest(),
        )
        diagnostic["launch_timestamp"] = time.time()
        handle = launcher.launch_task(request)
        with self._state_lock:
            self._active_handle = handle
            stopping = self._stopping
        status = None
        try:
            diagnostic["execution_id"] = handle.execution_id
            diagnostic["worker_pid"] = getattr(handle, "process_group_id", None)
            self._append_diagnostic(runtime_root, {**diagnostic, "event": "launched"})

            deadline = None if self.worker_timeout is None else time.monotonic() + self.worker_timeout
            cancellation_reason = None
            while True:
                if stopping or self._stopping or abort_signal.triggered:
                    cancellation_reason = "aborted"
                    status = handle.cancel()
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if self.result_wait_timeout is not None:
                    wait_started, result_sent = store.result_wait_state(identity)
                    if wait_started is not None and (
                        result_sent is None or result_sent > wait_started + self.result_wait_timeout
                    ):
                        result_remaining = wait_started + self.result_wait_timeout - time.monotonic()
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
                time.sleep(self.poll_interval if remaining is None else min(self.poll_interval, remaining))

            diagnostic["settled_timestamp"] = time.time()
            self._append_diagnostic(runtime_root, {**diagnostic, "event": "settled", "status": status.phase.value})
            if not status.settled:
                raise RuntimeError(f"task worker did not settle: {status.failure_reason or status}")
            if cancellation_reason == "aborted" or (cancellation_reason is None and status.cancel_requested):
                return make_reply(ReturnCode.TASK_ABORTED)
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
            launched_pid = diagnostic["worker_pid"]
            if launched_pid is not None and completion.worker_pid != launched_pid:
                raise ValueError(
                    f"task completion worker PID {completion.worker_pid} does not match launched process {launched_pid}"
                )
            diagnostic["worker_pid"] = completion.worker_pid
            diagnostic["worker_ppid"] = completion.worker_ppid
            diagnostic["worker_started_timestamp"] = completion.started_at
            diagnostic["worker_completed_timestamp"] = completion.completed_at
            for message in store.read_analytics(identity, completion):
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
            with self._state_lock:
                if task_id in self._pending_publication:
                    raise RuntimeError(f"task {task_id!r} already has a result pending publication")
                self._pending_publication[task_id] = (identity, store, runtime_root, diagnostic)
            return result
        except BaseException:
            try:
                observed = handle.poll()
                if not observed.settled:
                    handle.cancel()
            except BaseException:
                self._fail_unconfirmed_settlement(fl_ctx)
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
                    self._stopping = True
            if not settled:
                self._fail_unconfirmed_settlement(fl_ctx)

    def _record_publication_outcome(self, fl_ctx: FLContext):
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        with self._state_lock:
            pending = self._pending_publication.pop(task_id, None)
        if pending is None:
            return
        identity, store, runtime_root, diagnostic = pending
        succeeded = (
            fl_ctx.get_prop(FLContextKey.TASK_RESULT_SEND_SUCCESS) is True
            and fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is True
        )
        diagnostic["publication_timestamp"] = time.time()
        diagnostic["publication_outcome"] = "accepted" if succeeded else "not_accepted"
        try:
            self._append_diagnostic(runtime_root, {**diagnostic, "event": "publication"})
        except Exception as e:
            self.log_error(fl_ctx, f"failed to record task worker publication outcome: {e}")
            return
        if succeeded and self._artifact_cleanup == TaskArtifactCleanup.ACCEPTED:
            try:
                store.release_payloads(identity)
                with self._state_lock:
                    self._retained_attempts.pop(identity.attempt_id, None)
            except Exception as e:
                self.log_warning(fl_ctx, f"failed to remove accepted task worker artifacts: {e}")
