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

import logging
import threading
import time
from abc import ABC, abstractmethod

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey, RunProcessKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobLauncherSpec
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.fuel.common.exit_codes import PROCESS_EXIT_REASON, ProcessExitCode
from nvflare.fuel.f3.cellnet.core_cell import FQCN
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.defs import CellChannel, CellChannelTopic, JobFailureMsgKey, new_cell_message
from nvflare.private.fed.utils.fed_utils import get_job_launcher, get_return_code
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .client_status import ClientStatus, get_status_message


class ClientExecutor(ABC):
    @abstractmethod
    def start_app(
        self,
        client,
        job_id,
        job_meta,
        args,
        app_custom_folder,
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
            app_custom_folder: FL application custom folder
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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.startup = startup
        self.run_processes = {}
        self.lock = threading.Lock()

        self.job_query_timeout = ConfigService.get_float_var(
            name="job_query_timeout", conf=SystemConfigs.APPLICATION_CONF, default=5.0
        )

    def start_app(
        self,
        client,
        job_id,
        job_meta,
        args,
        app_custom_folder,
        allocated_resource,
        token,
        resource_manager: ResourceManagerSpec,
        fl_ctx: FLContext,
    ):
        """Starts the app.

        Args:
            client: the FL client object
            job_id: the job_id
            job_meta: job meta data
            args: admin command arguments for starting the worker process
            app_custom_folder: FL application custom folder
            allocated_resource: allocated resources
            token: token from resource manager
            resource_manager: resource manager
            fl_ctx: FLContext
        """

        job_launcher: JobLauncherSpec = get_job_launcher(job_meta, fl_ctx)
        job_handle = job_launcher.launch_job(job_meta, fl_ctx)
        self.logger.info(f"Launch job_id: {job_id}  with job launcher: {type(job_launcher)} ")

        client.multi_gpu = False

        with self.lock:
            self.run_processes[job_id] = {
                RunProcessKey.JOB_HANDLE: job_handle,
                RunProcessKey.STATUS: ClientStatus.STARTING,
            }

        thread = threading.Thread(
            target=self._wait_child_process_finish,
            args=(client, job_id, allocated_resource, token, resource_manager, args.workspace, fl_ctx),
        )
        thread.start()

    def _get_job_launcher(self, job_meta: dict, fl_ctx: FLContext) -> JobLauncherSpec:
        engine = fl_ctx.get_engine()
        fl_ctx.set_prop(FLContextKey.JOB_META, job_meta, private=True, sticky=False)
        engine.fire_event(EventType.GET_JOB_LAUNCHER, fl_ctx)

        job_launcher = fl_ctx.get_prop(FLContextKey.JOB_LAUNCHER)
        if not (job_launcher and isinstance(job_launcher, list)):
            raise RuntimeError(f"There's no job launcher can handle this job: {job_meta}.")

        return job_launcher[0]

    def notify_job_status(self, job_id, job_status):
        run_process = self.run_processes.get(job_id)
        if run_process:
            run_process[RunProcessKey.STATUS] = job_status

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
            process_status = self.run_processes.get(job_id, {}).get(RunProcessKey.STATUS, ClientStatus.NOT_STARTED)
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

    def abort_app(self, job_id):
        """Aborts the running app.

        Args:
            job_id: the job_id
        """
        # When the HeartBeat cleanup process try to abort the worker process, the job maybe already terminated,
        # Use retry to avoid print out the error stack trace.
        retry = 1
        while retry >= 0:
            process_status = self.run_processes.get(job_id, {}).get(RunProcessKey.STATUS, ClientStatus.NOT_STARTED)
            if process_status == ClientStatus.STARTED:
                try:
                    with self.lock:
                        job_handle = self.run_processes[job_id][RunProcessKey.JOB_HANDLE]
                    data = {}
                    request = new_cell_message({}, data)
                    self.client.cell.fire_and_forget(
                        targets=self._job_fqcn(job_id),
                        channel=CellChannel.CLIENT_COMMAND,
                        topic=AdminCommandNames.ABORT,
                        message=request,
                        optional=True,
                    )
                    self.logger.debug("abort sent to worker")
                    t = threading.Thread(target=self._terminate_job, args=[job_handle, job_id])
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

    def _terminate_job(self, job_handle, job_id):
        max_wait = 10.0
        done = False
        start = time.time()
        while True:
            process = self.run_processes.get(job_id)
            if not process:
                # already finished gracefully
                done = True
                break

            if time.time() - start > max_wait:
                # waited enough
                break

            time.sleep(0.05)  # we want to quickly check

        job_handle.terminate()
        self.logger.info(f"run ({job_id}): child worker process terminated")

    def abort_task(self, job_id):
        """Aborts the client executing task.

        Args:
            job_id: the job_id
        """
        process_status = self.run_processes.get(job_id, {}).get(RunProcessKey.STATUS, ClientStatus.NOT_STARTED)
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

    def _wait_child_process_finish(
        self, client, job_id, allocated_resource, token, resource_manager, workspace, fl_ctx
    ):
        self.logger.info(f"run ({job_id}): waiting for child worker process to finish.")
        job_handle = self.run_processes.get(job_id, {}).get(RunProcessKey.JOB_HANDLE)
        if job_handle:
            job_handle.wait()

            return_code = get_return_code(job_handle, job_id, workspace, self.logger)

            self.logger.info(f"run ({job_id}): child worker process finished with RC {return_code}")

            if return_code in [ProcessExitCode.UNSAFE_COMPONENT, ProcessExitCode.CONFIG_ERROR]:
                request = new_cell_message(
                    headers={},
                    payload={
                        JobFailureMsgKey.JOB_ID: job_id,
                        JobFailureMsgKey.CODE: return_code,
                        JobFailureMsgKey.REASON: PROCESS_EXIT_REASON[return_code],
                    },
                )
                self.client.cell.fire_and_forget(
                    targets=[FQCN.ROOT_SERVER],
                    channel=CellChannel.SERVER_MAIN,
                    topic=CellChannelTopic.REPORT_JOB_FAILURE,
                    message=request,
                    optional=True,
                )
                self.logger.info(f"reported failure of job {job_id} to server!")

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

    def get_status(self, job_id):
        process_status = self.run_processes.get(job_id, {}).get(RunProcessKey.STATUS, ClientStatus.STOPPED)
        return process_status

    def get_run_processes_keys(self):
        with self.lock:
            return [x for x in self.run_processes.keys()]
