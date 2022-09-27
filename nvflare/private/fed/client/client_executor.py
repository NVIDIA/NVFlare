# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import shlex
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from multiprocessing.connection import Client

from nvflare.apis.fl_constant import AdminCommandNames, ReturnCode, RunProcessKey
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .client_status import ClientStatus, get_status_message


class ClientExecutor(ABC):
    @abstractmethod
    def start_app(
        self,
        client,
        job_id,
        args,
        app_custom_folder,
        listen_port,
        allocated_resource,
        token,
        resource_manager,
        target: str,
    ):
        """Starts the client app.

        Args:
            client: the FL client object
            job_id: the job_id
            args: admin command arguments for starting the FL client training
            app_custom_folder: FL application custom folder
            listen_port: port to listen the command.
            allocated_resource: allocated resources
            token: token from resource manager
            resource_manager: resource manager
            target: SP target location
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

    @abstractmethod
    def process_aux_command(self, shareable: Shareable, job_id):
        """Processes the aux command.

        Args:
            shareable: aux message Shareable
            job_id: the job_id
        """
        pass


class ProcessExecutor(ClientExecutor):
    """Run the Client executor in a child process."""

    def __init__(self, startup):
        """To init the ProcessExecutor.

        Args:
            startup: startup folder
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.startup = startup
        self.run_processes = {}
        self.lock = threading.Lock()

    def get_conn_client(self, job_id):
        # should be call within self.lock
        listen_port = self.run_processes.get(job_id, {}).get(RunProcessKey.LISTEN_PORT)
        conn_client = self.run_processes.get(job_id, {}).get(RunProcessKey.CONNECTION, None)

        if not conn_client:
            try:
                address = ("localhost", listen_port)
                conn_client = Client(address, authkey="client process secret password".encode())
                self.run_processes[job_id][RunProcessKey.CONNECTION] = conn_client
            except Exception:
                pass

        return conn_client

    def start_app(
        self,
        client,
        job_id,
        args,
        app_custom_folder,
        listen_port,
        allocated_resource,
        token,
        resource_manager: ResourceManagerSpec,
        target: str,
    ):
        """Starts the app.

        Args:
            client: the FL client object
            job_id: the job_id
            args: admin command arguments for starting the worker process
            app_custom_folder: FL application custom folder
            listen_port: port to listen the command.
            allocated_resource: allocated resources
            token: token from resource manager
            resource_manager: resource manager
            target: SP target location
        """
        new_env = os.environ.copy()
        if app_custom_folder != "":
            new_env["PYTHONPATH"] = new_env.get("PYTHONPATH", "") + os.pathsep + app_custom_folder

        command_options = ""
        for t in args.set:
            command_options += " " + t
        command = (
            f"{sys.executable} -m nvflare.private.fed.app.client.worker_process -m "
            + args.workspace
            + " -w "
            + self.startup
            + " -t "
            + client.token
            + " -d "
            + client.ssid
            + " -n "
            + job_id
            + " -c "
            + client.client_name
            + " -p "
            + str(listen_port)
            + " -g "
            + target
            + " -s fed_client.json "
            " --set" + command_options + " print_conf=True"
        )
        # use os.setsid to create new process group ID
        process = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, env=new_env)

        self.logger.info("Worker child process ID: {}".format(process.pid))

        client.multi_gpu = False

        with self.lock:
            self.run_processes[job_id] = {
                RunProcessKey.LISTEN_PORT: listen_port,
                RunProcessKey.CONNECTION: None,
                RunProcessKey.CHILD_PROCESS: process,
                RunProcessKey.STATUS: ClientStatus.STARTED,
            }

        thread = threading.Thread(
            target=self._wait_child_process_finish,
            args=(client, job_id, allocated_resource, token, resource_manager),
        )
        thread.start()

    def check_status(self, job_id):
        """Checks the status of the running client.

        Args:
            job_id: the job_id

        Returns:
            A client status message
        """
        try:
            with self.lock:
                conn_client = self.get_conn_client(job_id)

                if conn_client:
                    data = {"command": AdminCommandNames.CHECK_STATUS, "data": {}}
                    conn_client.send(data)
                    status_message = conn_client.recv()
                    self.logger.debug("check status from process listener......")
                    return status_message
                else:
                    process_status = ClientStatus.NOT_STARTED
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
            with self.lock:
                conn_client = self.get_conn_client(job_id)

                if conn_client:
                    data = {"command": AdminCommandNames.SHOW_STATS, "data": {}}
                    conn_client.send(data)
                    run_info = conn_client.recv()
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
            with self.lock:
                conn_client = self.get_conn_client(job_id)

                if conn_client:
                    data = {"command": AdminCommandNames.SHOW_ERRORS, "data": {}}
                    conn_client.send(data)
                    errors_info = conn_client.recv()
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
            with self.lock:
                conn_client = self.get_conn_client(job_id)

                if conn_client:
                    data = {"command": AdminCommandNames.RESET_ERRORS, "data": {}}
                    conn_client.send(data)
        except Exception as e:
            self.logger.error(f"reset_errors execution exception: {secure_format_exception(e)}.")
            secure_log_traceback()

    def process_aux_command(self, shareable: Shareable, job_id):
        """Processes the aux command.

        Args:
            shareable: aux message Shareable
            job_id: the job_id
        """
        try:
            with self.lock:
                conn_client = self.get_conn_client(job_id)
                if conn_client:
                    data = {"command": AdminCommandNames.AUX_COMMAND, "data": shareable}
                    conn_client.send(data)
                    reply = conn_client.recv()
                    return reply
                else:
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        except Exception:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def abort_app(self, job_id):
        """Aborts the running app.

        Args:
            job_id: the job_id
        """
        with self.lock:
            # When the HeartBeat cleanup process try to abort the worker process, the job maybe already terminated,
            # Use retry to avoid print out the error stack trace.
            retry = 1
            while retry >= 0:
                process_status = self.run_processes.get(job_id, {}).get(RunProcessKey.STATUS, ClientStatus.NOT_STARTED)
                if process_status == ClientStatus.STARTED:
                    try:
                        child_process = self.run_processes[job_id][RunProcessKey.CHILD_PROCESS]
                        conn_client = self.get_conn_client(job_id)
                        if conn_client:
                            data = {"command": AdminCommandNames.ABORT, "data": {}}
                            conn_client.send(data)
                            self.logger.debug("abort sent")

                        threading.Thread(target=self._terminate_process, args=[child_process, job_id]).start()
                        self.run_processes.pop(job_id)
                        break
                    except Exception as e:
                        if retry == 0:
                            self.logger.error(
                                f"abort_worker_process execution exception: {secure_format_exception(e)} for run: {job_id}."
                            )
                            secure_log_traceback()
                        retry -= 1
                        time.sleep(5.0)
                    finally:
                        if conn_client:
                            conn_client.close()
                else:
                    self.logger.info(f"Client worker process for run: {job_id} was already terminated.")
                    break

        self.logger.info("Client worker process is terminated.")

    def _terminate_process(self, child_process, job_id):
        # wait for client to handle abort
        time.sleep(10.0)
        # kill the sub-process group directly
        try:
            os.killpg(os.getpgid(child_process.pid), 9)
            self.logger.debug("kill signal sent")
        except Exception:
            pass
        child_process.terminate()
        self.logger.info(f"run ({job_id}): child worker process terminated")

    def abort_task(self, job_id):
        """Aborts the client executing task.

        Args:
            job_id: the job_id
        """
        with self.lock:
            process_status = self.run_processes.get(job_id, {}).get(RunProcessKey.STATUS, ClientStatus.NOT_STARTED)
            if process_status == ClientStatus.STARTED:
                conn_client = self.get_conn_client(job_id)
                if conn_client:
                    data = {"command": AdminCommandNames.ABORT_TASK, "data": {}}
                    conn_client.send(data)
                    self.logger.debug("abort_task sent")

    def _wait_child_process_finish(self, client, job_id, allocated_resource, token, resource_manager):
        # wait for the listen_command thread to start, and send "start" message to wake up the connection.
        start = time.time()
        while True:
            with self.lock:
                conn_client = self.get_conn_client(job_id)
                if conn_client:
                    data = {"command": AdminCommandNames.START_APP, "data": {}}
                    conn_client.send(data)
                    break
            time.sleep(1.0)
            if time.time() - start > 15:
                break

        self.logger.info(f"run ({job_id}): waiting for child worker process to finish.")
        with self.lock:
            child_process = self.run_processes.get(job_id, {}).get(RunProcessKey.CHILD_PROCESS)
        if child_process:
            child_process.wait()
            return_code = child_process.returncode
            self.logger.info(f"run ({job_id}): child worker process finished with execution code: {return_code}")

        if allocated_resource:
            resource_manager.free_resources(
                resources=allocated_resource, token=token, fl_ctx=client.engine.new_context()
            )

        with self.lock:
            conn_client = self.get_conn_client(job_id)
            if conn_client:
                conn_client.close()
            if job_id in self.run_processes.keys():
                self.run_processes.pop(job_id)

    def get_status(self, job_id):
        with self.lock:
            process_status = self.run_processes.get(job_id, {}).get(RunProcessKey.STATUS, ClientStatus.STOPPED)
            return process_status

    def get_run_processes_keys(self):
        with self.lock:
            return [x for x in self.run_processes.keys()]
