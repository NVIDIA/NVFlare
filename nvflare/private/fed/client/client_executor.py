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
from multiprocessing.connection import Client

from nvflare.apis.fl_constant import AdminCommandNames, ReturnCode, RunProcessKey
from nvflare.apis.resource_manager_spec import ResourceConsumerSpec, ResourceManagerSpec
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.fuel.utils.pipe.file_pipe import FilePipe

from .client_status import ClientStatus, get_status_message


class ClientExecutor(object):
    def __init__(self, uid, startup) -> None:
        """To init the ClientExecutor.

        Args:
            uid: client name
            startup: startup folder
        """
        pipe_path = startup + "/comm"
        if not os.path.exists(pipe_path):
            os.makedirs(pipe_path)

        self.pipe = FilePipe(root_path=pipe_path, name="training")
        self.logger = logging.getLogger(self.__class__.__name__)

    def start_train(
        self,
        client,
        job_id,
        args,
        app_root,
        app_custom_folder,
        listen_port,
        allocated_resource,
        token,
        resource_consumer,
        resource_manager,
        target: str,
    ):
        """start_train method to start the FL client training.

        Args:
            client: the FL client object
            job_id: the job_id
            args: admin command arguments for starting the FL client training
            app_root: the root folder of the running APP
            app_custom_folder: FL application custom folder
            listen_port: port to listen the command.
            allocated_resource: allocated resources
            token: token from resource manager
            resource_consumer: resource consumer
            resource_manager: resource manager
            target: SP target location

        """
        pass

    def check_status(self, client, job_id) -> str:
        """To check the status of the running client.

        Args:
            client: the FL client object
            job_id: the job_id

        Returns:
            A client status message
        """
        pass

    def abort_train(self, client, job_id):
        """To abort the client training.

        Args:
            client: the FL client object
            job_id: the job_id
        """
        pass

    def abort_task(self, client, job_id):
        """To abort the client executing task.

        Args:
            client: the FL client object
            job_id: the job_id
        """
        pass

    def get_run_info(self, job_id) -> dict:
        """Get the run information.

        Args:
            job_id: the job_id

        Returns:
            A dict of run information.
        """
        pass

    def get_errors(self, job_id):
        """Get the error information.

        Returns:
            A dict of error information.

        """
        pass

    def reset_errors(self, job_id):
        """Reset the error information."""
        pass

    def send_aux_command(self, shareable: Shareable, job_id):
        """To send the aux command to child process.

        Args:
            shareable: aux message Shareable
            job_id: the job_id
        """
        pass

    def cleanup(self):
        """Cleanup."""
        self.pipe.clear()


class ProcessExecutor(ClientExecutor):
    """Run the Client executor in a child process."""

    def __init__(self, uid, startup):
        """To init the ProcessExecutor.

        Args:
            uid: client name
            startup: startup folder
        """
        ClientExecutor.__init__(self, uid, startup)

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

    def start_train(
        self,
        client,
        job_id,
        args,
        app_root,
        app_custom_folder,
        listen_port,
        allocated_resource,
        token,
        resource_consumer: ResourceConsumerSpec,
        resource_manager: ResourceManagerSpec,
        target: str,
    ):
        if allocated_resource:
            resource_consumer.consume(allocated_resource)

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

        print("training child process ID: {}".format(process.pid))

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

    def check_status(self, client, job_id):
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
            self.logger.error(f"check_status execution exception: {e}.", exc_info=True)
            return "execution exception. Please try again."

    def get_run_info(self, job_id):
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
            self.logger.error(f"get_run_info execution exception: {e}.", exc_info=True)
            return {"error": "no info collector. Please try again."}

    def get_errors(self, job_id):
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
            self.logger.error(f"get_errors execution exception: {e}.", exc_info=True)
            return None

    def reset_errors(self, job_id):
        try:
            with self.lock:
                conn_client = self.get_conn_client(job_id)

                if conn_client:
                    data = {"command": AdminCommandNames.RESET_ERRORS, "data": {}}
                    conn_client.send(data)
        except Exception as e:
            self.logger.error(f"reset_errors execution exception: {e}.", exc_info=True)

    def send_aux_command(self, shareable: Shareable, job_id):
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

    def abort_train(self, client, job_id):
        with self.lock:
            # When the HeartBeat cleanup process try to abort the train, the job maybe already terminated,
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
                            self.logger.error(f"abort_train execution exception: {e} for run: {job_id}.", exc_info=True)
                        retry -= 1
                        time.sleep(5.0)
                    finally:
                        if conn_client:
                            conn_client.close()
                        self.cleanup()
                else:
                    self.logger.info(f"run: {job_id} already terminated.")
                    break

        self.logger.info("Client training was terminated.")

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

    def abort_task(self, client, job_id):
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

    def close(self):
        self.cleanup()
