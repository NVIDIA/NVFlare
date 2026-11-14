# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import copy
import json
import math
import os
import shutil
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import (
    AdminCommandNames,
    ConnPropKey,
    FLContextKey,
    ReservedKey,
    RunProcessKey,
    SystemConfigs,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobLauncherSpec, JobProcessArgs, JobReturnCode
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.exit_codes import PROCESS_EXIT_REASON, ProcessExitCode
from nvflare.fuel.f3.cellnet.core_cell import FQCN
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.job_config.task_execution import get_client_execution_lifetime
from nvflare.private.defs import CellChannel, CellChannelTopic, JobOutcomeMsgKey, RequestHeader, new_cell_message
from nvflare.private.fed.utils.fed_utils import get_job_launcher, get_return_code
from nvflare.private.fed.utils.job_cert_utils import remove_job_cert
from nvflare.security.logging import secure_format_exception, secure_log_traceback
from nvflare.utils.job_launcher_utils import has_cj_resource_spec, validate_cj_execution_lifetime

from .client_status import ClientStatus, get_status_message

REPORTABLE_JOB_FAILURES = {
    ProcessExitCode.EXCEPTION: PROCESS_EXIT_REASON[ProcessExitCode.EXCEPTION],
    ProcessExitCode.UNSAFE_COMPONENT: PROCESS_EXIT_REASON[ProcessExitCode.UNSAFE_COMPONENT],
    ProcessExitCode.CONFIG_ERROR: PROCESS_EXIT_REASON[ProcessExitCode.CONFIG_ERROR],
    ProcessExitCode.INFRASTRUCTURE_ERROR: PROCESS_EXIT_REASON[ProcessExitCode.INFRASTRUCTURE_ERROR],
    JobReturnCode.ABORTED: "aborted",
}

_ABORT_REQUESTED_KEY = "_abort_requested"
_RESTART_KEY = "_restart"
_CLEANUP_PENDING_KEY = "_cleanup_pending"
# Escrow the held admission identity after a failed start unwinds its caller.
# Release remains an operator responsibility when cleanup is unconfirmed.
_RESOURCE_OWNER_KEY = "_resource_owner"
_CLEANUP_RETRY_KEY = "_cleanup_retry_running"


class JobCleanupPending(RuntimeError):
    """Launch failed with a runtime still owned by JobExecutor, including its admission resources."""


class _CleanupCall:
    """One control operation; a stuck launcher must not block its parent or CP shutdown."""

    def __init__(self, operation):
        self.error = None
        self.done = threading.Event()

        def run():
            try:
                operation()
            except BaseException as e:
                self.error = e
            finally:
                self.done.set()

        self.thread = threading.Thread(target=run, daemon=True)
        try:
            self.thread.start()
        except BaseException as e:
            self.error = e
            self.done.set()


@dataclass
class _JobRestart:
    launcher: JobLauncherSpec
    meta: dict
    args: dict
    launch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attempts: int = 0  # Replacement launches since the last successful initialization.


class _PendingJobHandle(JobHandleSpec):
    """Hold abort intent until a launcher publishes or returns the real handle."""

    def __init__(self):
        self._lock = threading.Lock()
        self._job_handle = None
        self._pending_heartbeat_cleanup: bool | None = None
        self.cleanup_calls = None
        self.cancel_call = None

    def attach(self, job_handle: JobHandleSpec) -> bool | None:
        with self._lock:
            if self._job_handle is not None:
                if self._job_handle is not job_handle:
                    raise RuntimeError("launcher published a different job handle")
                return None
            self._job_handle = job_handle
            heartbeat_cleanup = self._pending_heartbeat_cleanup
            self._pending_heartbeat_cleanup = None
            return heartbeat_cleanup

    def terminate(self, heartbeat_cleanup=False):
        with self._lock:
            job_handle = self._job_handle
            if job_handle is None:
                # Preserve a real abort if stop requests race before attachment.
                if self._pending_heartbeat_cleanup is None or not heartbeat_cleanup:
                    self._pending_heartbeat_cleanup = heartbeat_cleanup
                return
        if heartbeat_cleanup:
            terminate_for_heartbeat_cleanup = getattr(job_handle, "_terminate_for_heartbeat_cleanup", None)
            if terminate_for_heartbeat_cleanup:
                return terminate_for_heartbeat_cleanup()
        return job_handle.terminate()

    def poll(self):
        with self._lock:
            job_handle = self._job_handle
        if job_handle is None:
            return None
        return job_handle.poll()

    def wait(self):
        with self._lock:
            job_handle = self._job_handle
        if job_handle is None:
            raise RuntimeError("cannot wait for a pending job handle before it is attached")
        return job_handle.wait()

    def can_restart(self):
        with self._lock:
            job_handle = self._job_handle
        return job_handle is not None and job_handle.can_restart() is True


class ClientExecutor(ABC):
    @abstractmethod
    def start_app(
        self,
        client,
        job_id,
        job_meta,
        args,
        allocated_resource,
        token,
        resource_manager,
        fl_ctx: FLContext,
    ):
        """Starts the client app.

        Args:
            client: the FL client object
            job_id: the job_id
            args: admin command arguments for starting the FL client training
            allocated_resource: allocated resources
            token: token from resource manager
            resource_manager: resource manager
            fl_ctx: FLContext
        """
        pass

    @abstractmethod
    def check_status(self, job_id) -> str:
        """Checks the status of the running client.

        Args:
            job_id: the job_id

        Returns:
            A client status message
        """
        pass

    @abstractmethod
    def abort_app(self, job_id):
        """Aborts the running app.

        Args:
            job_id: the job_id
        """
        pass

    @abstractmethod
    def abort_task(self, job_id):
        """Aborts the client executing task.

        Args:
            job_id: the job_id
        """
        pass

    @abstractmethod
    def get_run_info(self, job_id):
        """Gets the run information.

        Args:
            job_id: the job_id

        Returns:
            A dict of run information.
        """

    @abstractmethod
    def get_errors(self, job_id):
        """Get the error information.

        Returns:
            A dict of error information.

        """

    @abstractmethod
    def reset_errors(self, job_id):
        """Resets the error information.

        Args:
            job_id: the job_id
        """


class JobExecutor(ClientExecutor):
    """Run the Client executor in a child process."""

    def __init__(self, client, startup):
        """To init the ProcessExecutor.

        Args:
            startup: startup folder
        """
        self.client = client
        self.logger = get_obj_logger(self)
        self.startup = startup
        self.run_processes = {}
        self._deleting_runs = set()
        self.lock = threading.Lock()

        self.job_query_timeout = ConfigService.get_float_var(
            name="job_query_timeout", conf=SystemConfigs.APPLICATION_CONF, default=5.0
        )
        self.max_cj_restart_failures = ConfigService.get_int_var(
            name="max_cj_restart_failures", conf=SystemConfigs.APPLICATION_CONF, default=3
        )
        if self.max_cj_restart_failures < 0:
            raise ValueError("max_cj_restart_failures must be nonnegative")
        self.job_launch_cleanup_timeout = ConfigService.get_float_var(
            name="job_launch_cleanup_timeout", conf=SystemConfigs.APPLICATION_CONF, default=10.0
        )
        if not math.isfinite(self.job_launch_cleanup_timeout) or self.job_launch_cleanup_timeout <= 0:
            raise ValueError("job_launch_cleanup_timeout must be finite and positive")

    def start_app(
        self,
        client,
        job_id,
        job_meta,
        args,
        allocated_resource,
        token,
        resource_manager: ResourceManagerSpec,
        fl_ctx: FLContext,
    ):
        """Starts the app.

        Args:
            client: the FL client object
            job_id: the job_id
            job_meta: job metadata
            args: admin command arguments for starting the worker process
            allocated_resource: allocated resources
            token: token from resource manager
            resource_manager: resource manager
            fl_ctx: FLContext
        """
        # use a deep copy of the args for operation since its content will be changed!
        args = copy.deepcopy(args)

        # update the job meta
        workspace = Workspace(args.workspace, site_name=client.client_name)
        meta_file = workspace.get_job_meta_path(job_id)
        if not os.path.exists(meta_file):
            raise RuntimeError(f"missing deployed job metadata file for job '{job_id}': {meta_file}")
        with open(meta_file) as f:
            deployed_job_meta = json.load(f)
        for meta_key in (
            JobMetaKey.JOB_ID.value,
            JobMetaKey.RESOURCE_SPEC.value,
            JobMetaKey.CJ_RESOURCE_SPEC.value,
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
            JobMetaKey.SCOPE.value,
            JobMetaKey.STUDY.value,
        ):
            deployed_value = deployed_job_meta.get(meta_key)
            if deployed_value != job_meta.get(meta_key):
                raise RuntimeError(f"START_JOB metadata differs from deployed job metadata for '{meta_key}'")
            if meta_key in deployed_job_meta:
                job_meta[meta_key] = copy.deepcopy(deployed_value)
        if has_cj_resource_spec(job_meta, client.client_name):
            client_config = ConfigService.load_config_dict(workspace.get_client_app_config_file_path(job_id))
            validate_cj_execution_lifetime(job_meta, client.client_name, get_client_execution_lifetime(client_config))
        if deployed_job_meta.get(AppValidationKey.BYOC, False):
            job_meta[AppValidationKey.BYOC] = True
        else:
            job_meta.pop(AppValidationKey.BYOC, None)

        # Preserve deploy-time launch metadata while recording scheduler-maintained start metadata.
        with open(meta_file, "w") as f:
            json.dump(job_meta, f, indent=4)

        fl_ctx.set_prop(FLContextKey.ALLOCATED_RESOURCES, allocated_resource, private=True, sticky=False)
        job_launcher: JobLauncherSpec = get_job_launcher(job_meta, fl_ctx)
        if has_cj_resource_spec(job_meta, client.client_name):
            from .task_execution_policy import TaskExecutionPolicy

            TaskExecutionPolicy.from_workspace(workspace).validate_job_launcher(job_launcher)

        # prepare command args for the job process
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        server_config = fl_ctx.get_prop(FLContextKey.SERVER_CONFIG)
        if not server_config:
            raise RuntimeError(f"missing {FLContextKey.SERVER_CONFIG} in FL context")
        service = server_config[0].get("service", {})
        if not isinstance(service, dict):
            raise RuntimeError(f"expect server config data to be dict but got {type(service)}")
        command_options = ""
        for t in args.set:
            command_options += " " + t
        command_options += " print_conf=True"
        args.set.append("print_conf=True")
        fl_ctx.remove_prop(FLContextKey.ARGS, force_removal=True)
        fl_ctx.set_prop(FLContextKey.ARGS, args, private=True, sticky=False)

        # Job process args are the same for all job launchers! Letting each job launcher compute the job
        # args would be error-prone and would require access to internal server components (e.g. cell).
        # We prepare job process args here and save the prepared result in the fl_ctx.
        # This way, the job launcher won't need to compute these args again.
        # The job launcher will only need to use the args properly to launch the job process!
        #
        # Each arg is a tuple of (arg_option, arg_value).
        # Note that the arg_option is fixed for each arg, and is not launcher specific!
        job_args = {
            JobProcessArgs.EXE_MODULE: ("-m", "nvflare.private.fed.app.client.worker_process"),
            JobProcessArgs.JOB_ID: ("-n", job_id),
            JobProcessArgs.CLIENT_NAME: ("-c", client.client_name),
            JobProcessArgs.AUTH_TOKEN: ("-t", client.token),
            JobProcessArgs.TOKEN_SIGNATURE: ("-ts", client.token_signature),
            JobProcessArgs.SSID: ("-d", client.ssid),
            JobProcessArgs.WORKSPACE: ("-m", args.workspace),
            JobProcessArgs.STARTUP_DIR: ("-w", workspace_obj.get_startup_kit_dir()),
            JobProcessArgs.PARENT_URL: ("-p", str(client.cell.get_internal_listener_url())),
            JobProcessArgs.SCHEME: ("-scheme", service.get("scheme", "grpc")),
            JobProcessArgs.TARGET: ("-g", service.get("target")),
            JobProcessArgs.STARTUP_CONFIG_FILE: ("-s", "fed_client.json"),
            JobProcessArgs.OPTIONS: ("--set", command_options),
        }

        params = client.cell.get_internal_listener_params()
        if params:
            parent_conn_sec = params.get(ConnPropKey.CONNECTION_SECURITY)
            if parent_conn_sec:
                job_args[JobProcessArgs.PARENT_CONN_SEC] = ("--parent_conn_sec", parent_conn_sec)

        restart = _JobRestart(job_launcher, copy.deepcopy(job_meta), job_args)
        pending_handle = _PendingJobHandle()
        with self.lock:
            if job_id in self._deleting_runs:
                raise RuntimeError(f"job {job_id}: deletion in progress")
            if job_id in self.run_processes:
                raise RuntimeError(f"client app for job '{job_id}' is still registered")
            self.run_processes[job_id] = {
                RunProcessKey.JOB_HANDLE: pending_handle,
                RunProcessKey.STATUS: ClientStatus.STARTING,
                _ABORT_REQUESTED_KEY: False,
                _RESTART_KEY: restart,
                _RESOURCE_OWNER_KEY: (allocated_resource, token, resource_manager),
            }

        try:
            self._launch_job(job_id, restart, pending_handle, fl_ctx)
        except JobCleanupPending:
            raise  # Keep the owner even if a timed-out observation has since returned.
        except BaseException:
            with self.lock:
                if self.run_processes.get(job_id, {}).get(RunProcessKey.JOB_HANDLE) is pending_handle:
                    self.run_processes.pop(job_id, None)
            raise

        client.multi_gpu = False

        thread = threading.Thread(
            target=self._wait_child_process_finish,
            args=(client, job_id, allocated_resource, token, resource_manager, args.workspace, fl_ctx),
            daemon=True,
        )
        try:
            thread.start()
        except BaseException as e:
            self.logger.error(f"job {job_id}: monitor startup failed: {secure_format_exception(e)}")
            if not self._cleanup_failed_launch(job_id, pending_handle):
                raise JobCleanupPending(f"job {job_id}: monitor startup failed; cleanup remains owned by CP") from e
            with self.lock:
                self.run_processes.pop(job_id, None)
            raise

    @staticmethod
    def _launch_cleanup_settled(pending_handle):
        cancel, observe = pending_handle.cleanup_calls
        return cancel.done.is_set() and observe.done.is_set() and observe.error is None

    def _retain_cleanup_owner(self, job_id):
        with self.lock:
            process = self.run_processes.get(job_id)
            if process is not None:
                process[_CLEANUP_PENDING_KEY] = True
        self.logger.critical(
            f"job {job_id}: cleanup unconfirmed; retaining handle, admission resources and job credentials. "
            "Operator reconciliation is required before removing the run or restarting CP; do not resubmit."
        )

    def _cleanup_failed_launch(self, job_id, pending_handle, retry=False):
        # One attempt only, with separate bounds. Always observe even if cancellation
        # fails or blocks. The retained registration owns both calls after timeout;
        # neither elapsed time nor cancellation acknowledgement authorizes release.
        previous = pending_handle.cleanup_calls
        if retry and previous and any(not call.done.is_set() for call in previous):
            cancel, observe = previous
        else:
            cancel = None if retry else pending_handle.cancel_call
            cancel = cancel or _CleanupCall(pending_handle.terminate)
            pending_handle.cancel_call = cancel
            cancel.done.wait(self.job_launch_cleanup_timeout)
            observe = _CleanupCall(pending_handle.wait)
            pending_handle.cleanup_calls = (cancel, observe)
        observe.done.wait(self.job_launch_cleanup_timeout)
        for name, call in (("cancellation", cancel), ("observation", observe)):
            if not call.done.is_set():
                self.logger.error(f"job {job_id}: launch cleanup {name} exceeded job_launch_cleanup_timeout")
            elif call.error is not None:
                self.logger.error(f"job {job_id}: launch cleanup {name} failed: {secure_format_exception(call.error)}")
        if not self._launch_cleanup_settled(pending_handle):
            self._retain_cleanup_owner(job_id)
            return False
        return True

    def _launch_job(self, job_id, restart, pending_handle, fl_ctx):
        job_args = dict(restart.args)
        launch_option = f"{RequestHeader.CJ_LAUNCH_ID}={restart.launch_id}"
        flag, options = job_args[JobProcessArgs.OPTIONS]
        job_args[JobProcessArgs.OPTIONS] = (flag, f"{options} {launch_option}")
        # Container launchers take their --set list from ARGS instead of JOB_PROCESS_ARGS.
        args = copy.copy(fl_ctx.get_prop(FLContextKey.ARGS))
        args.set = [option for option in args.set if not option.startswith(f"{RequestHeader.CJ_LAUNCH_ID}=")]
        args.set.append(launch_option)
        fl_ctx.set_prop(FLContextKey.ARGS, args, private=True, sticky=False)
        # CP may have refreshed its registration while the previous allocation ran.
        for key, value in (
            (JobProcessArgs.AUTH_TOKEN, self.client.token),
            (JobProcessArgs.TOKEN_SIGNATURE, self.client.token_signature),
            (JobProcessArgs.SSID, self.client.ssid),
        ):
            job_args[key] = (job_args[key][0], value)
        fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args, private=True, sticky=False)
        job_handle = None

        def register_handle(handle):
            nonlocal job_handle
            heartbeat_cleanup = pending_handle.attach(handle)
            job_handle = handle
            if heartbeat_cleanup is not None:
                # Attach and remember ownership before applying the saved abort.
                # Reuse this cancellation during launch cleanup if it fails or hangs.
                cancel = _CleanupCall(lambda: pending_handle.terminate(heartbeat_cleanup=heartbeat_cleanup))
                pending_handle.cancel_call = cancel
                cancel.done.wait(self.job_launch_cleanup_timeout)
                # Do not raise through publication: some launchers catch callback
                # errors and enter their own blocking cleanup. Check after return.

        # Initial and replacement launches can both wait after acquiring a
        # runtime. Publish its handle before that wait so abort remains live.
        fl_ctx.set_prop(FLContextKey.JOB_HANDLE_CALLBACK, register_handle, private=True, sticky=False)
        try:
            handle = restart.launcher.launch_job(copy.deepcopy(restart.meta), fl_ctx)
            if handle is None:
                raise RuntimeError(f"job launcher returned no job handle for job '{job_id}'")
            register_handle(handle)
            cancel = pending_handle.cancel_call
            if cancel is not None:
                if not cancel.done.is_set():
                    raise TimeoutError("saved abort did not complete within job_launch_cleanup_timeout")
                if cancel.error is not None:
                    raise cancel.error
            self.logger.info(f"Launched job {job_id} with job launcher: {type(restart.launcher)}")
            fl_ctx.set_prop(FLContextKey.JOB_META, restart.meta, private=True, sticky=False)
            fl_ctx.get_engine().fire_event(EventType.AFTER_JOB_LAUNCH, fl_ctx)
        except BaseException as e:
            # Once launched, retain ownership until the worker has exited.
            if job_handle is not None:
                self.logger.error(f"job {job_id}: launch failed: {secure_format_exception(e)}")
                if not self._cleanup_failed_launch(job_id, pending_handle):
                    raise JobCleanupPending(f"job {job_id}: launch failed; cleanup remains owned by CP") from e
            raise
        finally:
            fl_ctx.remove_prop(FLContextKey.JOB_HANDLE_CALLBACK)

    def _get_job_launcher(self, job_meta: dict, fl_ctx: FLContext) -> JobLauncherSpec:
        engine = fl_ctx.get_engine()
        fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
        engine.fire_event(EventType.BEFORE_JOB_LAUNCH, fl_ctx)

        job_launcher = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER)
        if not (job_launcher and isinstance(job_launcher, list)):
            raise RuntimeError(f"There's no job launcher can handle this job: {job_meta}.")

        return job_launcher[0]

    def notify_job_status(self, job_id, job_status, launch_id=None):
        with self.lock:
            run_process = self.run_processes.get(job_id)
            if not run_process or run_process.get(RunProcessKey.STATUS) == ClientStatus.STOPPED:
                return
            restart = run_process.get(_RESTART_KEY)
            if restart:
                # Older CJs and custom launchers may omit the launch ID.
                if launch_id is not None and launch_id != restart.launch_id:
                    return  # delayed notification from an earlier physical execution
                if job_status == ClientStatus.READY:
                    restart.attempts = 0
            run_process[RunProcessKey.STATUS] = ClientStatus.STARTED if job_status == ClientStatus.READY else job_status

    def _job_fqcn(self, job_id: str):
        return FQCN.join([self.client.cell.get_fqcn(), job_id])

    def check_status(self, job_id):
        """Checks the status of the running client.

        Args:
            job_id: the job_id

        Returns:
            A client status message
        """
        try:
            with self.lock:
                process = self.run_processes.get(job_id, {})
                process_status = (
                    ClientStatus.CLEANUP_PENDING
                    if process.get(_CLEANUP_PENDING_KEY)
                    else process.get(RunProcessKey.STATUS, ClientStatus.NOT_STARTED)
                )
            return get_status_message(process_status)
        except Exception as e:
            self.logger.error(f"check_status execution exception: {secure_format_exception(e)}.")
            secure_log_traceback()
            return "execution exception. Please try again."

    def get_run_info(self, job_id):
        """Gets the run information.

        Args:
            job_id: the job_id

        Returns:
            A dict of run information.
        """
        try:
            data = {}
            request = new_cell_message({}, data)
            return_data = self.client.cell.send_request(
                target=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.SHOW_STATS,
                request=request,
                optional=True,
                timeout=self.job_query_timeout,
            )
            return_code = return_data.get_header(MessageHeaderKey.RETURN_CODE)
            if return_code == ReturnCode.OK:
                run_info = return_data.payload
                return run_info
            else:
                return {}
        except Exception as e:
            self.logger.error(f"get_run_info execution exception: {secure_format_exception(e)}.")
            secure_log_traceback()
            return {"error": "no info collector. Please try again."}

    def get_errors(self, job_id):
        """Get the error information.

        Args:
            job_id: the job_id

        Returns:
            A dict of error information.
        """
        try:
            data = {"command": AdminCommandNames.SHOW_ERRORS, "data": {}}
            request = new_cell_message({}, data)
            return_data = self.client.cell.send_request(
                target=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.SHOW_ERRORS,
                request=request,
                optional=True,
                timeout=self.job_query_timeout,
            )
            return_code = return_data.get_header(MessageHeaderKey.RETURN_CODE)
            if return_code == ReturnCode.OK:
                errors_info = return_data.payload
                return errors_info
            else:
                return None
        except Exception as e:
            self.logger.error(f"get_errors execution exception: {secure_format_exception(e)}.")
            secure_log_traceback()
            return None

    def configure_job_log(self, job_id, config):
        """Configure the job log.

        Args:
            job_id: the job_id
            config: log config

         Returns:
            configure_job_log command message
        """
        try:
            request = new_cell_message({}, config)
            return_data = self.client.cell.send_request(
                target=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.CONFIGURE_JOB_LOG,
                request=request,
                optional=True,
                timeout=self.job_query_timeout,
            )
            return_code = return_data.get_header(MessageHeaderKey.RETURN_CODE)
            if return_code == ReturnCode.OK:
                return return_data.payload
            else:
                return f"failed to configure_job_log with return code: {return_code}"
        except Exception as e:
            err = f"configure_job_log execution exception: {secure_format_exception(e)}."
            self.logger.error(err)
            secure_log_traceback()
            return err

    def reset_errors(self, job_id):
        """Resets the error information.

        Args:
            job_id: the job_id
        """
        try:
            data = {"command": AdminCommandNames.RESET_ERRORS, "data": {}}
            request = new_cell_message({}, data)
            self.client.cell.fire_and_forget(
                targets=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.RESET_ERRORS,
                message=request,
                optional=True,
            )

        except Exception as e:
            self.logger.error(f"reset_errors execution exception: {secure_format_exception(e)}.")
            secure_log_traceback()

    def abort_app(self, job_id, heartbeat_cleanup=False):
        """Aborts the running app.

        Args:
            job_id: the job_id
            heartbeat_cleanup: whether heartbeat cleanup requested the abort rather than a user or administrator
        """
        # When the HeartBeat cleanup process try to abort the worker process, the job maybe already terminated,
        # Use retry to avoid print out the error stack trace.
        retry = 1
        while retry >= 0:
            with self.lock:
                process = self.run_processes.get(job_id)
                if process:
                    process[_ABORT_REQUESTED_KEY] = True
                process_status = (
                    process.get(RunProcessKey.STATUS, ClientStatus.NOT_STARTED) if process else ClientStatus.NOT_STARTED
                )
                job_handle = process.get(RunProcessKey.JOB_HANDLE) if process else None
                cleanup_pending = process.get(_CLEANUP_PENDING_KEY, False) if process else False
            if cleanup_pending:
                # Heartbeats must not repeatedly issue control calls or warnings.
                # An explicit admin abort may retry completed failed operations.
                if heartbeat_cleanup:
                    return
                with self.lock:
                    if process.get(_CLEANUP_RETRY_KEY):
                        return
                    process[_CLEANUP_RETRY_KEY] = True
                try:
                    try:
                        self.client.cell.fire_and_forget(
                            targets=self._job_fqcn(job_id),
                            channel=CellChannel.CLIENT_COMMAND,
                            topic=AdminCommandNames.ABORT,
                            message=new_cell_message({}, {}),
                            optional=True,
                        )
                    except Exception as e:
                        self.logger.warning(f"job {job_id}: cannot send abort: {secure_format_exception(e)}")
                    if self._cleanup_failed_launch(job_id, job_handle, retry=True):
                        self.logger.info(
                            f"job {job_id}: backend completion observed after admin abort; "
                            "registration, admission resources and credentials remain retained; "
                            "operator reconciliation required"
                        )
                finally:
                    with self.lock:
                        process.pop(_CLEANUP_RETRY_KEY, None)
                return
            if process_status in (ClientStatus.STARTING, ClientStatus.STARTED, ClientStatus.STOPPED):
                try:
                    if process_status == ClientStatus.STARTING:
                        if heartbeat_cleanup:
                            job_handle.terminate(heartbeat_cleanup=True)
                        else:
                            job_handle.terminate()
                        break
                    if process_status == ClientStatus.STOPPED:
                        # STOPPED means the runner returned, not that the OS
                        # process exited. Give archival and process-local teardown
                        # the normal bounded grace period before reclaiming the
                        # still-registered owned process.
                        t = threading.Thread(target=self._terminate_job, args=[job_handle, job_id, heartbeat_cleanup])
                        t.start()
                        t.join()
                        break
                    request = new_cell_message({}, {})
                    self.client.cell.fire_and_forget(
                        targets=self._job_fqcn(job_id),
                        channel=CellChannel.CLIENT_COMMAND,
                        topic=AdminCommandNames.ABORT,
                        message=request,
                        optional=True,
                    )
                    self.logger.debug("abort sent to worker")
                    t = threading.Thread(target=self._terminate_job, args=[job_handle, job_id, heartbeat_cleanup])
                    t.start()
                    t.join()
                    break
                except Exception as e:
                    if retry == 0:
                        self.logger.error(
                            f"abort_worker_process execution exception: {secure_format_exception(e)} for run: {job_id}."
                        )
                        secure_log_traceback()
                    retry -= 1
                    time.sleep(5.0)
            else:
                self.logger.info(f"Client worker process for run: {job_id} was already terminated.")
                break

        self.logger.info("Client worker process is terminated.")

    def send_to_job(
        self,
        job_id,
        channel: str,
        topic: str,
        msg: CellMessage,
        timeout: float,
        optional=False,
    ) -> CellMessage:
        """Send a message to CJ

        Args:
            job_id: id of the job
            channel: message channel
            topic: message topic
            msg: the message to be sent
            timeout: how long to wait for reply
            optional: whether the message is optional

        Returns: reply from CJ

        """
        # send any serializable data to the job cell
        return self.client.cell.send_request(
            target=self._job_fqcn(job_id),
            channel=channel,
            topic=topic,
            request=msg,
            timeout=timeout,
            optional=optional,
        )

    def _terminate_job(self, job_handle, job_id, heartbeat_cleanup=False):
        max_wait = 10.0
        start = time.time()
        while True:
            with self.lock:
                process = self.run_processes.get(job_id)
            if not process:
                # already finished gracefully
                return

            if time.time() - start > max_wait:
                # waited enough
                break

            time.sleep(0.05)  # we want to quickly check

        if heartbeat_cleanup:
            job_handle.terminate(heartbeat_cleanup=True)
        else:
            job_handle.terminate()
        self.logger.info(f"run ({job_id}): child worker process terminated")

    def abort_task(self, job_id):
        """Aborts the client executing task.

        Args:
            job_id: the job_id
        """
        with self.lock:
            process = self.run_processes.get(job_id, {})
            process_status = process.get(RunProcessKey.STATUS, ClientStatus.NOT_STARTED)
            restart = process.get(_RESTART_KEY)
            job_handle = process.get(RunProcessKey.JOB_HANDLE)
        if restart and (restart.attempts or job_handle.poll() not in (None, JobReturnCode.UNKNOWN)):
            # There is no live task to notify while its CJ is being replaced.
            self.abort_app(job_id)
            return
        if process_status == ClientStatus.STARTED:
            data = {"command": AdminCommandNames.ABORT_TASK, "data": {}}
            request = new_cell_message({}, data)
            self.client.cell.fire_and_forget(
                targets=self._job_fqcn(job_id),
                channel=CellChannel.CLIENT_COMMAND,
                topic=AdminCommandNames.ABORT_TASK,
                message=request,
                optional=True,
            )
            self.logger.debug("abort_task sent")

    def _restart_job(self, job_id, job_handle, return_code, fl_ctx):
        # A replacement cannot reconstruct an intermediate CJ's child-result receipts.
        if not fl_ctx.get_prop(ReservedKey.IS_LEAF, True):
            return False
        if not self.max_cj_restart_failures or return_code not in (
            JobReturnCode.EXECUTION_ERROR,
            ProcessExitCode.EXCEPTION,
            ProcessExitCode.INFRASTRUCTURE_ERROR,
        ):
            return False
        if job_handle.can_restart() is not True:
            return False
        with self.lock:
            process = self.run_processes.get(job_id, {})
            restart = process.get(_RESTART_KEY)
            if (
                not restart
                or process.get(_ABORT_REQUESTED_KEY)
                or process.get(RunProcessKey.STATUS) == ClientStatus.STOPPED
            ):
                return False
            if restart.attempts >= self.max_cj_restart_failures:
                self.logger.error(f"job {job_id}: exhausted {restart.attempts} consecutive failed CJ restarts")
                return False
            restart.attempts += 1
            restart.launch_id = uuid.uuid4().hex
            pending_handle = _PendingJobHandle()
            process[RunProcessKey.JOB_HANDLE] = pending_handle
            process[RunProcessKey.STATUS] = ClientStatus.STARTING
        self.logger.info(f"job {job_id}: restarting CJ (attempt {restart.attempts}/{self.max_cj_restart_failures})")
        previous = job_handle._job_handle if isinstance(job_handle, _PendingJobHandle) else job_handle
        fl_ctx.set_prop(FLContextKey.JOB_RESTART_HANDLE, previous, private=True, sticky=False)
        try:
            self._launch_job(job_id, restart, pending_handle, fl_ctx)
        finally:
            fl_ctx.remove_prop(FLContextKey.JOB_RESTART_HANDLE)
        return True

    def _wait_child_process_finish(
        self, client, job_id, allocated_resource, token, resource_manager, workspace, fl_ctx
    ):
        self.logger.info(f"run ({job_id}): waiting for child worker process to finish.")
        job_handle = self.run_processes.get(job_id, {}).get(RunProcessKey.JOB_HANDLE)
        run_dir = Workspace.run_dir_path(workspace, job_id)
        try:
            while job_handle:
                job_handle.wait()
                return_code = get_return_code(job_handle, job_id, workspace, self.logger)
                try:
                    if not self._restart_job(job_id, job_handle, return_code, fl_ctx):
                        break
                except JobCleanupPending:
                    return  # The replacement handle and resources remain registered.
                except Exception as e:
                    # A submission exception may mean the backend accepted the job.
                    # Without a handle proving otherwise, never submit again.
                    self.logger.error(f"job {job_id}: cannot restart CJ: {secure_format_exception(e)}")
                    return_code = ProcessExitCode.INFRASTRUCTURE_ERROR
                    break
                job_handle = self.run_processes[job_id][RunProcessKey.JOB_HANDLE]
        except Exception as e:
            self.logger.error(f"job {job_id}: cannot observe CJ completion: {secure_format_exception(e)}")
            self._retain_cleanup_owner(job_id)
            return
        # Keep the same job credential throughout replacement. Observation errors
        # above leave it with the registered cleanup owner, never in a finally block.
        remove_job_cert(run_dir)
        if job_handle:
            with self.lock:
                process = self.run_processes.get(job_id, {})
                process_status = process.get(RunProcessKey.STATUS)
                abort_requested = process.get(_ABORT_REQUESTED_KEY, False)
            # A generic RC 1 is actionable only while a checked-in worker is still active.
            # STARTING remains an infrastructure failure, while STOPPED teardown noise and
            # launcher UNKNOWN retain their existing non-reportable behavior.
            if return_code == JobReturnCode.EXECUTION_ERROR and not abort_requested:
                if process_status == ClientStatus.STARTING:
                    return_code = ProcessExitCode.INFRASTRUCTURE_ERROR
                elif process_status == ClientStatus.STARTED:
                    return_code = ProcessExitCode.EXCEPTION

            self.logger.info(f"run ({job_id}): child worker process finished with RC {return_code}")

            # Every terminal result resolves the server's pending client outcome, including success.
            failure_reason = REPORTABLE_JOB_FAILURES.get(return_code)
            try:
                request = new_cell_message(
                    headers={},
                    payload={
                        JobOutcomeMsgKey.JOB_ID: job_id,
                        JobOutcomeMsgKey.CODE: return_code,
                        JobOutcomeMsgKey.REASON: failure_reason,
                    },
                )
                reply = self.client.send_request_before_shutdown(
                    target=FQCN.ROOT_SERVER,
                    channel=CellChannel.SERVER_MAIN,
                    topic=CellChannelTopic.REPORT_JOB_OUTCOME,
                    request=request,
                    timeout=self.job_query_timeout,
                    optional=True,
                )
                if reply is None:
                    # Shutdown invalidates the site token. The server's client-quit/dead-client
                    # path resolves any outcome still pending after communication stops.
                    self.logger.info(
                        f"not reporting terminal outcome of job {job_id}: client communication has stopped"
                    )
                elif reply.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK:
                    self.logger.error(f"could not report terminal outcome of job {job_id}")
            except Exception as e:
                self.logger.error(f"could not report terminal outcome of job {job_id}: {secure_format_exception(e)}")

        if allocated_resource:
            resource_manager.free_resources(
                resources=allocated_resource, token=token, fl_ctx=client.engine.new_context()
            )
        with self.lock:
            self.run_processes.pop(job_id, None)
        self.logger.debug(f"run ({job_id}): child worker resources freed.")

        engine = fl_ctx.get_engine()
        fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, job_id, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.CLIENT_NAME, client.client_name, private=True, sticky=False)
        engine.fire_event(EventType.JOB_COMPLETED, fl_ctx)
        self.logger.debug(f"Fired event JOB_COMPLETED {EventType.JOB_COMPLETED}")

    def check_start_app(self, job_id):
        """Reject a retained owner before the engine's already-started shortcut."""
        with self.lock:
            if job_id in self._deleting_runs:
                raise RuntimeError(f"job {job_id}: deletion in progress")
            process = self.run_processes.get(job_id, {})
            if process.get(_CLEANUP_PENDING_KEY):
                # This request owns a new admission allocation, not the retained
                # one. An ordinary error lets its caller release only that retry.
                raise RuntimeError(f"job {job_id}: cleanup remains unconfirmed; operator reconciliation required")
            return process.get(RunProcessKey.STATUS, ClientStatus.STOPPED)

    def get_status(self, job_id):
        with self.lock:
            process = self.run_processes.get(job_id, {})
            if process.get(_CLEANUP_PENDING_KEY):
                return ClientStatus.CLEANUP_PENDING
            return process.get(RunProcessKey.STATUS, ClientStatus.STOPPED)

    def delete_run(self, job_id, job_dir):
        """Remove files only after the client parent has released the run."""
        with self.lock:
            if job_id in self.run_processes:
                raise RuntimeError(f"job {job_id}: still registered; cannot delete an owned run")
            if job_id in self._deleting_runs:
                raise RuntimeError(f"job {job_id}: deletion in progress")
            self._deleting_runs.add(job_id)
        try:
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir)
        finally:
            with self.lock:
                self._deleting_runs.discard(job_id)

    def get_run_processes_keys(self):
        with self.lock:
            return [x for x in self.run_processes.keys()]
