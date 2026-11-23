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

"""CP-side ownership and lifecycle for coordinators with disposable task workers."""

import logging
import os
import secrets
import threading
from dataclasses import dataclass, field, replace

from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobReturnCode
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.config_service import ConfigService

JOB_LAUNCH_OPTIONS = "_job_launch_options"


@dataclass(frozen=True)
class JobLaunchOptions:
    """Trusted parent settings for one launch, never job-selected site policy.

    ``environment`` replaces the launcher's initial environment when supplied.
    Launchers still add job credentials and application import paths. Paths and
    container options affect this launch only, not the shared launcher instance.
    ``publish_failed_launch_handle`` opts into receiving an uncertain execution
    identity on launch errors; the caller must retain it for reconciliation.
    ``container_kwargs`` replaces site container defaults when supplied.
    """

    environment: dict | None = None
    apply_study_runtime: bool = True
    publish_failed_launch_handle: bool = False
    workspace_mount: str | None = None
    container_kwargs: dict | None = None


def get_job_launch_options(fl_ctx):
    options = fl_ctx.get_prop(JOB_LAUNCH_OPTIONS)
    if options is None:
        return JobLaunchOptions()
    return options


@dataclass
class _TaskExecutionState:
    """CP-owned admission fence and task-service registry, shared by its jobs."""

    lock: object = field(default_factory=threading.Lock)
    closing: bool = False
    services: dict = field(default_factory=dict)


class JobHandleWithTaskWorkers(JobHandleSpec):
    """Preserve CJ outcome while bounding the wait for parent-owned cleanup.

    A managed cleanup timeout raises so the shared executor retains registration,
    admission resources and credentials while CP's physical owner keeps retrying.
    """

    def __init__(self, handle, owner, cleanup_timeout=30.0, *, allow_restart=True):
        self.handle = handle
        self.owner = owner
        self.cleanup_timeout = cleanup_timeout
        self._allow_restart = allow_restart

    def terminate(self):
        self.owner.release()
        if getattr(self.owner, "manage_cj", False):
            return self.owner.terminate_cj()
        return self.handle.terminate()

    def _terminate_for_heartbeat_cleanup(self):
        self.owner.release()
        if getattr(self.owner, "manage_cj", False):
            return self.owner.terminate_cj()
        cleanup = getattr(self.handle, "_terminate_for_heartbeat_cleanup", self.handle.terminate)
        return cleanup()

    def can_restart(self):
        """Replace the CJ only after CP-owned workers have all settled."""
        if not self._allow_restart or self.handle.can_restart() is not True:
            return False
        self.owner.release()
        self.owner.thread.join(self.cleanup_timeout)
        if self.owner.thread.is_alive():
            with self.owner.lock:
                pending = [a.execution_id for a in self.owner.attempts.values() if not a.status.settled]
            logging.getLogger(__name__).warning(
                "CJ recovery skipped for job %s: worker allocations %s not settled within %ss",
                self.owner.job_id,
                pending,
                self.cleanup_timeout,
            )
            return False
        return True

    def poll(self):
        code = self.handle.poll()
        if code != JobReturnCode.UNKNOWN:
            self.owner.release()
        return code

    def wait(self):
        if getattr(self.owner, "manage_cj", False):
            # Normal training has no cleanup deadline. Observation or abort
            # releases the owner; only its subsequent finalization is bounded.
            self.owner.released_event.wait()
        else:
            self.handle.wait()
        self.owner.release()
        self.owner.thread.join(self.cleanup_timeout)
        if self.owner.thread.is_alive():
            if getattr(self.owner, "manage_cj", False):
                self.owner.warn_cleanup_pending()
                raise TimeoutError(f"job {self.owner.job_id}: coordinator or task-worker cleanup remains unconfirmed")
            with self.owner.lock:
                pending = [a.execution_id for a in self.owner.attempts.values() if not a.status.settled]
            logging.getLogger(__name__).warning(
                "Task cleanup unconfirmed for job %s allocations %s; CP retains cleanup", self.owner.job_id, pending
            )


@dataclass
class TaskExecutionBinding:
    """Prepared backend binding and settings for one CJ launch."""

    backend: object
    service_class: type
    owner_env: str
    token_env: str
    meta: dict
    options: JobLaunchOptions


def _launch_job_with_task_workers(prepared, launcher, fl_ctx, *, cleanup_timeout):
    engine = fl_ctx.get_engine()
    cell = engine.get_cell()
    state = engine._task_execution_state
    token = secrets.token_hex(32)
    with state.lock:
        if state.closing:
            raise RuntimeError("parent task runtime is shutting down")
        service = prepared.service_class.get_or_create(cell)
        # Register and admit atomically with CP shutdown, so shutdown cannot
        # miss an owner created just after its services snapshot.
        state.services[prepared.service_class] = (service, cleanup_timeout)
        owner = service.add_job(
            prepared.backend,
            prepared.meta[JobConstants.JOB_ID],
            fl_ctx.get_identity_name(),
            token,
            cj_launch_pending=True,
        )
    previous = {key: fl_ctx.get_prop_detail(key) for key in (JOB_LAUNCH_OPTIONS, FLContextKey.JOB_HANDLE_CALLBACK)}
    previous_callback = fl_ctx.get_prop(FLContextKey.JOB_HANDLE_CALLBACK)
    result = None

    def attach(native):
        nonlocal result
        if result is not None:
            if result.handle is not native:
                raise RuntimeError("launcher published a different coordinator handle")
            return result
        result = JobHandleWithTaskWorkers(native, owner, cleanup_timeout, allow_restart=False)
        with state.lock:
            owner.attach_cj(native)
        # Publish even when shutdown has begun so the executor retains the
        # registration until the task owner confirms physical cleanup.
        if previous_callback is not None:
            previous_callback(result)
        with state.lock:
            if state.closing:
                raise RuntimeError("parent task runtime shut down during coordinator creation")
        return result

    environment = dict(prepared.options.environment or {})
    environment.update({prepared.owner_env: cell.get_fqcn(), prepared.token_env: token})
    options = replace(prepared.options, environment=environment, publish_failed_launch_handle=True)
    try:
        fl_ctx.remove_prop(JOB_LAUNCH_OPTIONS)
        fl_ctx.remove_prop(FLContextKey.JOB_HANDLE_CALLBACK)
        fl_ctx.set_prop(JOB_LAUNCH_OPTIONS, options, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, attach, private=True, sticky=False)
        return attach(launcher.launch_job(prepared.meta, fl_ctx))
    except BaseException:
        owner.release()
        if result is not None:
            try:
                result.terminate()
            except Exception:
                logging.getLogger(__name__).exception("Coordinator termination failed; CP retains cleanup")
            owner.finish_cj_launch()
            owner.thread.join(cleanup_timeout)
            if owner.thread.is_alive():
                logging.getLogger(__name__).warning("Coordinator cleanup unconfirmed; CP retains job ownership")
        raise
    finally:
        owner.finish_cj_launch()
        for key, detail in previous.items():
            fl_ctx.remove_prop(key)
            if detail is not None:
                fl_ctx.set_prop(key, **detail)


def _prepare_process_slurm_cj(launcher, job_meta, fl_ctx, launcher_config=None):
    from nvflare.app_opt.job_launcher.slurm.task_launcher import _prepare_task_binding

    options = launcher._task_coordinator_options(job_meta, fl_ctx)
    return _prepare_task_binding(job_meta, fl_ctx, options, launcher_config=launcher_config)


def _prepare_docker_slurm_cj(launcher, job_meta, fl_ctx, launcher_config=None):
    from nvflare.app_opt.job_launcher.slurm.task_launcher import _prepare_task_binding

    options = launcher._task_coordinator_options(job_meta, fl_ctx)
    return _prepare_task_binding(
        job_meta,
        fl_ctx,
        options,
        workspace=Workspace(launcher.workspace, fl_ctx.get_identity_name()),
        launcher_config=launcher_config,
    )


def validate_task_site(workspace):
    """Validate optional canonical task policy before CP accepts jobs."""
    path = workspace.get_resources_file_path()
    site_config = ConfigService.load_config_dict(path) if path and os.path.isfile(path) else {}
    if "task_execution" in site_config:
        from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy

        return TaskExecutionPolicy.from_workspace(workspace)
    return None


def launch_task_job(launcher, job_meta, fl_ctx, workspace):
    """Validate the launcher pairing and set up CP ownership for task execution."""
    from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy, _prepare_cp_host_cj

    policy = TaskExecutionPolicy.from_workspace(workspace)
    binding = _prepare_cp_host_cj(policy.launcher, launcher, job_meta, fl_ctx, launcher_config=policy.launcher_config)
    if binding is not None:
        return _launch_job_with_task_workers(binding, launcher, fl_ctx, cleanup_timeout=policy.finalization_timeout)
    return launcher.launch_job(job_meta, fl_ctx)


def shutdown_task_execution(engine, timeout=None):
    if engine is None:
        return
    state = engine._task_execution_state
    with state.lock:
        state.closing = True
        services = tuple(state.services.values())
    for service, configured_timeout in services:
        try:
            service.shutdown(timeout=configured_timeout if timeout is None else timeout)
        except Exception:
            logging.getLogger(__name__).exception("Task service shutdown failed; continuing CP shutdown")
