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
from typing import Optional

from nvflare.apis.fl_constant import AdminCommandNames, ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.utils.common_utils import get_open_ports

from .client_status import ClientStatus, get_status_message


class ClientExecutor(ABC):
    def __init__(self) -> None:
        """Init ClientExecutor."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def start_train(self, client, args, app_root, app_custom_folder, listen_port):
        """Start training.

        Args:
            client: the FL client object
            args: admin command arguments for starting the FL client training
            app_root: the root folder of the running APP
            app_custom_folder: FL application custom folder
            listen_port: port to listen the command.
        """
        pass

    @abstractmethod
    def check_status(self, client) -> str:
        """Check status.

        Args:
            client: the FL client object

        Returns:
            A client status message
        """
        pass

    @abstractmethod
    def abort_train(self, client):
        """Abort training.

        Args:
            client: the FL client object
        """
        pass

    @abstractmethod
    def abort_task(self, client):
        """Abort the executing task.

        Args:
            client: the FL client object
        """
        pass

    @abstractmethod
    def get_run_info(self) -> dict:
        """Get the run information.

        Returns:
            A dict of run information.
        """
        pass

    @abstractmethod
    def get_errors(self) -> Optional[dict]:
        """Get the error information.

        Returns:
            None if no error, otherwise a dict of error information.
        """
        pass

    @abstractmethod
    def reset_errors(self):
        """Reset the error information."""
        pass

    @abstractmethod
    def send_aux_command(self, aux_message: Shareable):
        """Send aux message.

        Args:
            aux_message (Shareable): message to sent
        """
        pass


class ProcessExecutor(ClientExecutor):
    """Run the client worker process in a child process."""

    def __init__(self, uid, startup):
        """Init a ProcessExecutor.

        Args:
            uid: client name
            startup: startup folder
        """
        super().__init__()

        self.startup = startup
        self.conn_client = None
        self.listen_port = get_open_ports(1)[0]
        self.lock = threading.Lock()
        self.child_process = None

    def get_conn_client(self):
        if not self.conn_client:
            try:
                address = ("localhost", self.listen_port)
                self.conn_client = Client(address, authkey="client process secret password".encode())
            except Exception as e:
                pass

    def start_train(self, client, args, app_root, app_custom_folder, listen_port):
        self.listen_port = listen_port

        new_env = os.environ.copy()
        if app_custom_folder != "":
            new_env["PYTHONPATH"] = new_env["PYTHONPATH"] + ":" + app_custom_folder

        command_options = ""
        for t in args.set:
            command_options += " " + t
        command = (
            f"{sys.executable} -m nvflare.private.fed.app.client.worker_process"
            f" --workspace {args.workspace}"
            f" --startup {self.startup}"
            f" --fed_client fed_client.json"
            f" --parent_pid {os.getpid()}"
            f" --set" + command_options + " print_conf=True"
        )
        # TODO:: this is only supported in UNIX
        # use os.setsid to create new process group ID
        process = subprocess.Popen(shlex.split(command, " "), preexec_fn=os.setsid, env=new_env)

        print("training child process ID: {}".format(process.pid))

        self.child_process = process
        client.multi_gpu = False

        client.status = ClientStatus.STARTED
        thread = threading.Thread(target=self.wait_training_process_finish, args=(client,))
        thread.start()

    def check_status(self, client):
        try:
            self.get_conn_client()

            if self.conn_client:
                data = {"command": AdminCommandNames.CHECK_STATUS, "data": {}}
                self.conn_client.send(data)
                status_message = self.conn_client.recv()
                self.logger.debug("check status from process listener......")
                return status_message
            else:
                return get_status_message(client.status)
        except Exception as e:
            self.logger.error(f"check_status() execution exception: {e}")
            return "execution exception. Please try again."

    def get_run_info(self) -> dict:
        try:
            self.get_conn_client()

            if self.conn_client:
                data = {"command": AdminCommandNames.SHOW_STATS, "data": {}}
                self.conn_client.send(data)
                run_info = self.conn_client.recv()
                return run_info
            else:
                return {}
        except Exception as e:
            self.logger.error(f"get_run_info() execution exception: {e}")
            return {"error": "no info collector. Please try again."}

    def get_errors(self) -> Optional[dict]:
        try:
            self.get_conn_client()

            if self.conn_client:
                data = {"command": AdminCommandNames.SHOW_ERRORS, "data": {}}
                self.conn_client.send(data)
                errors_info = self.conn_client.recv()
                return errors_info
            else:
                return None
        except Exception as e:
            self.logger.error(f"get_errors() execution exception: {e}")
            return None

    def reset_errors(self):
        try:
            self.get_conn_client()

            if self.conn_client:
                data = {"command": AdminCommandNames.RESET_ERRORS, "data": {}}
                self.conn_client.send(data)
        except Exception as e:
            self.logger.error(f"reset_errors() execution exception: {e}")

    def send_aux_command(self, aux_message: Shareable) -> Shareable:
        try:
            self.get_conn_client()
            if self.conn_client:
                data = {"command": AdminCommandNames.AUX_COMMAND, "data": aux_message}
                self.conn_client.send(data)
                reply = self.conn_client.recv()
                return reply
            else:
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        except:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def abort_train(self, client):
        if client.status == ClientStatus.STARTED:
            with self.lock:
                if self.child_process:
                    # wait for client to handle abort
                    if self.conn_client:
                        data = {"command": AdminCommandNames.ABORT, "data": {}}
                        self.conn_client.send(data)
                        self.logger.debug("abort sent")
                        time.sleep(2.0)

                    # kill the sub-process group directly
                    self._kill_child_process()
                    self.logger.debug("terminated")

                if self.conn_client:
                    self.conn_client.close()
                self.conn_client = None

        self.logger.info("Client training was terminated.")

    def abort_task(self, client):
        if client.status == ClientStatus.STARTED:
            if self.conn_client:
                data = {"command": AdminCommandNames.ABORT_TASK, "data": {}}
                self.conn_client.send(data)
                self.logger.debug("abort_task sent")

    def wait_training_process_finish(self, client):
        # wait for the listen_command thread to start, and send "start" message to wake up the connection.
        start = time.time()
        while True:
            self.get_conn_client()
            if self.conn_client:
                data = {"command": AdminCommandNames.START_APP, "data": {}}
                self.conn_client.send(data)
                break
            time.sleep(1.0)
            if time.time() - start > 15:
                break

        self.logger.info("waiting for process to finish.")
        self.child_process.wait()
        returncode = self.child_process.returncode
        self.logger.info(f"process finished with execution code: {returncode}")

        with self.lock:
            self.child_process = None

            if self.conn_client:
                self.conn_client.close()
            self.conn_client = None

        client.status = ClientStatus.STOPPED

    def _kill_child_process(self):
        try:
            os.killpg(os.getpgid(self.child_process.pid), 9)
            self.logger.debug("kill child signal sent")
        except Exception:
            pass
        self.child_process.terminate()
        self.child_process = None

    def close(self):
        if self.conn_client:
            data = {"command": AdminCommandNames.SHUTDOWN, "data": {}}
            self.conn_client.send(data)
            self.conn_client = None
        with self.lock:
            if self.child_process:
                self._kill_child_process()
