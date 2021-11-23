# Copyright (c) 2021, NVIDIA CORPORATION.
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
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from nvflare.apis.fl_constant import MachineStatus
from nvflare.apis.shareable import Shareable
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes
from nvflare.private.admin_defs import Message
from nvflare.private.defs import ClientStatusKey

from .client_engine_internal_spec import ClientEngineInternalSpec
from .client_executor import ProcessExecutor
from .client_run_manager import ClientRunInfo
from .client_status import ClientStatus


class ClientEngine(ClientEngineInternalSpec):
    """
    ClientEngine runs in the client parent process.
    """
    def __init__(self, client, client_name, sender, args, rank, workers=5):
        self.client = client
        self.client_name = client_name
        self.sender = sender
        self.args = args
        self.rank = rank
        self.client.process = None
        self.client_executor = ProcessExecutor(client.client_name)

        self.run_number = -1
        self.status = MachineStatus.STOPPED

        assert workers >= 1, "workers must >= 1"
        self.executor = ThreadPoolExecutor(max_workers=workers)

        self.logger = logging.getLogger(self.__class__.__name__)

    def set_agent(self, admin_agent):
        self.admin_agent = admin_agent

    def _get_open_port(self):
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        return port

    def do_validate(self, req: Message):
        self.logger.info("starting cross site validation.")
        future = self.executor.submit(lambda p: _do_validate(*p), [self.sender, req])
        # thread = threading.Thread(target=_do_validate, args=(self.sender, req))
        # thread.start()

        return "validate process started."

    # def client_status(self):
    #     if self.rank == 0:
    #         self.logger.info("check client status.")
    #         client_name = self.client.uid
    #         token = self.client.token
    #         message = "client name: {}".format(client_name)
    #         message += "\ttoken: {}".format(token)
    #
    #         message += "\tstatus: {}".format(self.client_executor.check_status(self.client))
    #         # if self.client.status == ClientStatus.TRAINING_STOPPED:
    #         #     message += '\tlocal epochs: {}'.format(self.client.model_manager.fitter.num_epochs)
    #
    #         return message
    #     else:
    #         return ""

    def get_engine_status(self):
        app_name = "?"
        if self.run_number == -1:
            run_number = "?"
        else:
            run_number = str(self.run_number)
            run_folder = os.path.join(self.args.workspace, "run_" + str(run_number))
            app_file = os.path.join(run_folder, "fl_app.txt")
            if os.path.exists(app_file):
                with open(app_file, "r") as f:
                    app_name = f.readline().strip()

        result = {
            ClientStatusKey.APP_NAME: app_name,
            ClientStatusKey.RUN_NUM: run_number,
            ClientStatusKey.STATUS: self.client_executor.check_status(self.client),
        }
        return result

    def start_app(self, run_number: int) -> str:
        status = self.client.status
        if status == ClientStatus.STARTING or status == ClientStatus.STARTED:
            return "Client app already started."

        app_root = os.path.join(self.args.workspace, "run_" + str(run_number), "app_" + self.client.client_name)
        if not os.path.exists(app_root):
            return "Client app does not exist. Please deploy it before starting client."

        if self.client.enable_byoc:
            app_custom_folder = os.path.join(app_root, "custom")
            try:
                sys.path.index(app_custom_folder)
            except ValueError:
                self.remove_custom_path()
                sys.path.append(app_custom_folder)
        else:
            app_custom_folder = ""

        self.logger.info("Starting client app. rank: {}".format(self.rank))

        open_port = self._get_open_port()
        self._write_token_file(run_number, open_port)
        self.run_number = run_number

        self.client_executor.start_train(self.client, self.args, app_root, app_custom_folder, open_port)

        return "Start the client app..."

    def set_run_number(self, run_number: int) -> str:
        self.run_number = run_number
        return ""

    def get_client_name(self):
        return self.client.client_name

    # def wait_training_process_finish(self):
    #     self.client.process.join()
    #
    #     # _cross_validation(self.client, self.args)
    #     self.client.status = ClientStatus.STOPPED

    # def start_mgpu_client(self, run_number, gpu_number):
    #     status = self.client.status
    #     if status == ClientStatus.STARTING or status == ClientStatus.STARTED:
    #         return "Client already in training."
    #
    #     app_root = os.path.join(self.args.app, "run_" + str(run_number), "app_" + self.client.uid)
    #     if not os.path.exists(app_root):
    #         return "Client app does not exist. Please deploy it before start client."
    #
    #     app_custom_folder = os.path.join(app_root, "custom")
    #     try:
    #         sys.path.index(app_custom_folder)
    #     except ValueError:
    #         self.remove_custom_path()
    #         sys.path.append(app_custom_folder)
    #
    #     self.logger.info("Starting client training. rank: {}".format(self.rank))
    #
    #     open_port = self._get_open_port()
    #     self._write_token_file(run_number, open_port)
    #
    #     self.client_executor.start_mgpu_train(
    #         self.client, self.args, app_root, gpu_number, app_custom_folder, open_port
    #     )
    #
    #     return "Start the client..."

    def _write_token_file(self, run_number, open_port):
        token_file = os.path.join(self.args.workspace, "client_token.txt")
        if os.path.exists(token_file):
            os.remove(token_file)
        with open(token_file, "wt") as f:
            f.write("%s\n%s\n%s\n%s\n" % (self.client.token, run_number, self.client.client_name, open_port))

    def wait_process_complete(self):
        self.client.process.wait()

        # self.client.cross_validation()
        self.client.status = ClientStatus.STOPPED

    def remove_custom_path(self):
        regex = re.compile(".*/run_.*/custom")
        custom_paths = list(filter(regex.search, sys.path))
        for path in custom_paths:
            sys.path.remove(path)

    def abort_app(self, run_number: int) -> str:
        status = self.client.status
        if status == ClientStatus.STOPPED:
            return "Client app already stopped."

        if status == ClientStatus.NOT_STARTED:
            return "Client app has not started."

        if status == ClientStatus.STARTING:
            return "Client app is starting, please wait for client to have started before abort."

        self.client_executor.abort_train(self.client)
        # self.run_number = -1

        return "Abort signal has been sent to the client App."

    def abort_task(self, run_number: int) -> str:
        status = self.client.status
        if status == ClientStatus.NOT_STARTED:
            return "Client app has not started."

        if status == ClientStatus.STARTING:
            return "Client app is starting, please wait for started before abort_task."

        self.client_executor.abort_task(self.client)
        # self.run_number = -1

        return "Abort signal has been sent to the current task. "

    def shutdown(self) -> str:
        self.logger.info("Client shutdown...")
        touch_file = os.path.join(self.args.workspace, "shutdown.fl")
        self.client_executor.close()
        future = self.executor.submit(lambda p: _shutdown_client(*p), [self.client, self.admin_agent, touch_file])

        return "Shutdown the client..."

    def restart(self) -> str:
        self.logger.info("Client shutdown...")
        touch_file = os.path.join(self.args.workspace, "restart.fl")
        self.client_executor.close()
        future = self.executor.submit(lambda p: _shutdown_client(*p), [self.client, self.admin_agent, touch_file])

        return "Restart the client..."

    def deploy_app(self, app_name: str, run_num: int, client_name: str, app_data) -> str:
        # if not os.path.exists('/tmp/tmp'):
        #     os.makedirs('/tmp/tmp')
        dest = os.path.join(self.args.workspace, "run_" + str(run_num), "app_" + client_name)
        # Remove the previous deployed app.
        if os.path.exists(dest):
            shutil.rmtree(dest)

        if not os.path.exists(dest):
            os.makedirs(dest)
        unzip_all_from_bytes(app_data, dest)

        app_file = os.path.join(self.args.workspace, "run_" + str(run_num), "fl_app.txt")
        if os.path.exists(app_file):
            os.remove(app_file)
        with open(app_file, "wt") as f:
            f.write(f"{app_name}")

        return ""

    def delete_run(self, run_num: int) -> str:
        run_number_folder = os.path.join(self.args.workspace, "run_" + str(run_num))
        if os.path.exists(run_number_folder):
            shutil.rmtree(run_number_folder)
        return "Delete run folder: {}".format(run_number_folder)

    def get_current_run_info(self) -> ClientRunInfo:
        return self.client_executor.get_run_info()

    def get_errors(self):
        return self.client_executor.get_errors()

    def reset_errors(self):
        self.client_executor.reset_errors()

    def send_aux_command(self, shareable: Shareable):
        return self.client_executor.send_aux_command(shareable)


def _do_validate(sender, message):
    print("starting the validate process .....")
    time.sleep(60)
    print("Generating processing result ......")
    reply = Message(topic=message.topic, body="")
    sender.send_result(reply)
    pass


def _shutdown_client(client, admin_agent, touch_file):
    with open(touch_file, "a"):
        os.utime(touch_file, None)

    try:
        print("About to shutdown the client...")
        client.communicator.heartbeat_done = True
        time.sleep(3)
        client.close()

        if client.process:
            client.process.terminate()

        admin_agent.shutdown()
    except BaseException as e:
        traceback.print_exc()
        print("FL client execution exception: " + str(e))
        # client.status = ClientStatus.TRAINING_EXCEPTION
