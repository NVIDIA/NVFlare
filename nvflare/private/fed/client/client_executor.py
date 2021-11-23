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
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client

from nvflare.apis.fl_constant import AdminCommandNames, ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from .client_status import ClientStatus, get_status_message


class ClientExecutor(object):
    def __init__(self, uid) -> None:
        pipe_path = "/tmp/fl/" + uid + "/comm"
        if not os.path.exists(pipe_path):
            os.makedirs(pipe_path)

        self.pipe = FilePipe(root_path=pipe_path, name="training")
        self.logger = logging.getLogger(self.__class__.__name__)

    def start_train(self, client, args, app_root, app_custom_folder, listen_port):
        """
        start_train method to start the FL client training.
        :param client: the FL client object.
        :param args: admin command arguments for starting the FL client training.
        :param app_root: the root folder of the running APP.
        :return:
        """
        pass

    def start_mgpu_train(self, client, args, app_root, gpu_number, app_custom_folder, listen_port):
        """
        start the FL client training using multi-GPU.
        :param client: the FL client object.
        :param args: admin command arguments for starting the FL client training.
        :param app_root: the root folder of the running APP.
        :param gpu_number: number of GPUs to run FL training
        :return:
        """
        pass

    def check_status(self, client):
        """
        check the status of the running client.
        :param client: the FL client object.
        :return: running FL client status message.
        """
        pass

    def abort_train(self, client):
        """
        To abort the running client.
        :param client: the FL client object.
        :return: N/A
        """
        pass

    def abort_task(self, client):
        """
        To abort the client executing task.
        :param client: the FL client object.
        :return: N/A
        """
        pass

    def get_run_info(self):
        """
        To get the run_info from the InfoCollector.
        Returns:

        """
        pass

    def get_errors(self):
        """
        To get the error_info from the InfoCollector.
        Returns:

        """
        pass

    def reset_errors(self):
        """
        To reset the error_info for the InfoCollector.
        Returns:

        """
        pass

    def send_aux_command(self, shareable: Shareable):
        """
        To send the aux command to child process.
        Returns:

        """
        pass

    def cleanup(self):
        self.pipe.clear()


class ProcessExecutor(ClientExecutor):
    """
    Run the Client executor in a child process.
    """
    def __init__(self, uid):
        ClientExecutor.__init__(self, uid)
        # self.client = client

        self.conn_client = None
        # self.pool = None

        self.listen_port = 6000

        self.lock = threading.Lock()

    def get_conn_client(self):
        if not self.conn_client:
            try:
                address = ("localhost", self.listen_port)
                self.conn_client = Client(address, authkey="client process secret password".encode())
            except Exception as e:
                pass

    def create_pipe(self):
        """Create pipe to communicate between child (training) and main (logic) thread."""
        pipe = FilePipe(root_path="/fl/server", name="training")

        return pipe

    def start_train(self, client, args, app_root, app_custom_folder, listen_port):
        # self.pool = multiprocessing.Pool(processes=1)
        # result = self.pool.apply_async(_start_client, (client, args, app_root))

        # self.conn_client, child_conn = mp.Pipe()
        # process = multiprocessing.Process(target=_start_client, args=(client, args, app_root, child_conn, self.pipe))
        # # process = multiprocessing.Process(target=_start_new)
        # process.start()

        self.listen_port = listen_port

        new_env = os.environ.copy()
        if app_custom_folder != "":
            new_env["PYTHONPATH"] = new_env["PYTHONPATH"] + ":" + app_custom_folder

        # self.retrieve_cross_validate_setting(client, app_root)

        command_options = ""
        for t in args.set:
            command_options += " " + t
        command = (
            f"{sys.executable} -m nvflare.private.fed.app.client.worker_process -m "
            + args.workspace
            + " -s fed_client.json "
            " --set" + command_options + " print_conf=True"
        )
        # use os.setsid to create new process group ID
        process = subprocess.Popen(shlex.split(command, " "), preexec_fn=os.setsid, env=new_env)

        print("training child process ID: {}".format(process.pid))

        client.process = process
        client.multi_gpu = False

        client.status = ClientStatus.STARTED
        thread = threading.Thread(
            target=self.wait_training_process_finish, args=(client, args, app_root, app_custom_folder)
        )
        thread.start()

    # def retrieve_cross_validate_setting(self, client, app_root):
    #     if client.config_folder == "":
    #         client_config = "config_fed_client.json"
    #     else:
    #         client_config = client.config_folder + "/config_fed_client.json"
    #     client_config = os.path.join(app_root, client_config)
    #     conf = Configurator(
    #         app_root=app_root,
    #         cmd_vars={},
    #         env_config={},
    #         wf_config_file_name=client_config,
    #         base_pkgs=[],
    #         module_names=[],
    #     )
    #     conf.configure()
    #     client.cross_site_validate = conf.wf_config_data.get("cross_validate", False)

    # def start_mgpu_train(self, client, args, app_root, gpu_number, app_custom_folder, listen_port):
    #     self.listen_port = listen_port
    #
    #     new_env = os.environ.copy()
    #     new_env["PYTHONPATH"] = new_env["PYTHONPATH"] + ":" + app_custom_folder
    #
    #     # self.retrieve_cross_validate_setting(client, app_root)
    #
    #     if client.platform == "PT":
    #         command = (
    #             f"{sys.executable} -m torch.distributed.launch --nproc_per_node="
    #             + str(gpu_number)
    #             + " --nnodes=1 --node_rank=0 "
    #             + '--master_addr="localhost" --master_port=1234 '
    #             + "-m nvflare.private.fed.app.client.worker_process -m "
    #             + args.workspace
    #             + " -s fed_client.json "
    #             " --set secure_train="
    #             + str(client.secure_train)
    #             + " print_conf=True use_gpu=True multi_gpu=True uid="
    #             + client.client_name
    #             + " config_folder="
    #             + client.config_folder
    #         )
    #         # use os.setsid to create new process group ID
    #         process = subprocess.Popen(shlex.split(command, " "), preexec_fn=os.setsid, env=new_env)
    #     else:
    #         command = (
    #             "mpirun -np "
    #             + str(gpu_number)
    #             + " -H localhost:"
    #             + str(gpu_number)
    #             + " -bind-to none -map-by slot -x NCCL_DEBUG=DEBUG -x LD_LIBRARY_PATH -x PATH "
    #             "-mca pml ob1 -mca btl ^openib --allow-run-as-root "
    #             f"{sys.executable} -u -m nvmidl.apps.fed_learn.client.worker_process -m "
    #             + args.workspace
    #             + " -s fed_client.json --set secure_train="
    #             + str(client.secure_train)
    #             + " multi_gpu=true uid="
    #             + client.client_name
    #             + " config_folder="
    #             + client.config_folder
    #         )
    #         process = subprocess.Popen(shlex.split(command, " "), env=new_env)
    #     client.process = process
    #     client.multi_gpu = True
    #     # self.pool = multiprocessing.Pool(processes=1)
    #     # result = self.pool.apply_async(self.call_mpirun, (client, args, app_root))
    #
    #     client.status = ClientStatus.STARTED
    #     thread = threading.Thread(
    #         target=self.wait_training_process_finish, args=(client, args, app_root, app_custom_folder)
    #     )
    #     thread.start()

    def check_status(self, client):
        try:
            self.get_conn_client()

            if self.conn_client:
                data = {"command": AdminCommandNames.CHECK_STATUS, "data": {}}
                self.conn_client.send(data)
                status_message = self.conn_client.recv()
                print("check status from process listener......")
                return status_message
            else:
                return get_status_message(client.status)
        except:
            self.logger.error("Check_status execution exception.")
            return "execution exception. Please try again."

    def get_run_info(self):
        try:
            self.get_conn_client()

            if self.conn_client:
                data = {"command": AdminCommandNames.SHOW_STATS, "data": {}}
                self.conn_client.send(data)
                run_info = self.conn_client.recv()
                return run_info
            else:
                return {}
        except:
            self.logger.error("get_run_info() execution exception.")
            return {"error": "no info collector. Please try again."}

    def get_errors(self):
        try:
            self.get_conn_client()

            if self.conn_client:
                data = {"command": AdminCommandNames.SHOW_ERRORS, "data": {}}
                self.conn_client.send(data)
                errors_info = self.conn_client.recv()
                return errors_info
            else:
                return None
        except:
            self.logger.error("get_errors() execution exception.")
            return None

    def reset_errors(self):
        try:
            self.get_conn_client()

            if self.conn_client:
                data = {"command": AdminCommandNames.RESET_ERRORS, "data": {}}
                self.conn_client.send(data)
        except:
            self.logger.error("reset_errors() execution exception.")

    def send_aux_command(self, shareable: Shareable):
        try:
            self.get_conn_client()
            if self.conn_client:
                data = {"command": AdminCommandNames.AUX_COMMAND, "data": shareable}
                self.conn_client.send(data)
                reply = self.conn_client.recv()
                return reply
            else:
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        except:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def abort_train(self, client):
        # if client.status == ClientStatus.CROSS_SITE_VALIDATION:
        #     # Only aborts cross site validation.
        #     client.abort()
        # elif client.status == ClientStatus.TRAINING_STARTED:
        if client.status == ClientStatus.STARTED:
            with self.lock:
                if client.process:
                    # if client.platform == 'PT' and client.multi_gpu:
                    #     # kill the sub-process group directly
                    #     os.killpg(os.getpgid(client.process.pid), 9)
                    # else:
                    #     client.process.terminate()

                    # kill the sub-process group directly
                    if self.conn_client:
                        data = {"command": AdminCommandNames.ABORT, "data": {}}
                        self.conn_client.send(data)
                        self.logger.debug("abort sent")
                        # wait for client to handle abort
                        time.sleep(2.0)

                    # kill the sub-process group directly
                    try:
                        os.killpg(os.getpgid(client.process.pid), 9)
                        self.logger.debug("kill signal sent")
                    except Exception as e:
                        pass
                    client.process.terminate()
                    self.logger.debug("terminated")

                # if self.pool:
                #     self.pool.terminate()
                if self.conn_client:
                    self.conn_client.close()
                self.conn_client = None
                self.cleanup()

        self.logger.info("Client training was terminated.")

    def abort_task(self, client):
        if client.status == ClientStatus.STARTED:
            if self.conn_client:
                data = {"command": AdminCommandNames.ABORT_TASK, "data": {}}
                self.conn_client.send(data)
                self.logger.debug("abort_task sent")

    def wait_training_process_finish(self, client, args, app_root, app_custom_folder):
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

        self.logger.info("waiting for process to finish")
        client.process.wait()
        returncode = client.process.returncode
        self.logger.info(f"process finished with execution code: {returncode}")

        with self.lock:
            client.process = None

            if self.conn_client:
                self.conn_client.close()
            self.conn_client = None

        # # result.get()
        # self.pool.close()
        # self.pool.join()
        # self.pool.terminate()

        # Not to run cross_validation in a new process any more.
        client.cross_site_validate = False

        client.status = ClientStatus.STOPPED

    def close(self):
        if self.conn_client:
            data = {"command": AdminCommandNames.SHUTDOWN, "data": {}}
            self.conn_client.send(data)
            self.conn_client = None
        self.cleanup()


# class ThreadExecutor(ClientExecutor):
#     def __init__(self, client, executor):
#         self.client = client
#         self.executor = executor

#     def start_train(self, client, args, app_root, app_custom_folder, listen_port):
#         future = self.executor.submit(lambda p: _start_client(*p), [client, args, app_root])

#     def start_mgpu_train(self, client, args, app_root, gpu_number, app_custom_folder, listen_port):
#         self.start_train(client, args, app_root)

#     def check_status(self, client):
#         return get_status_message(self.client.status)

#     def abort_train(self, client):
#         self.client.train_end = True
#         self.client.fitter.train_ctx.ask_to_stop_immediately()
#         self.client.fitter.train_ctx.set_prop("early_end", True)
#         # self.client.model_manager.close()
#         # self.client.status = ClientStatus.TRAINING_STOPPED
#         return "Aborting the client..."


def update_client_properties(client, trainer):
    # servers = [{t['name']: t['service']} for t in trainer.server_config]
    retry_timeout = 30
    # if trainer.client_config['retry_timeout']:
    #     retry_timeout = trainer.client_config['retry_timeout']
    client.client_args = trainer.client_config
    # client.servers = sorted(servers)[0]
    # client.model_manager.federated_meta = {task_name: list() for task_name in tuple(client.servers)}
    exclude_vars = trainer.client_config.get("exclude_vars", "dummy")
    # client.model_manager.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
    # client.model_manager.privacy_policy = trainer.privacy
    # client.model_manager.model_reader_writer = trainer.model_reader_writer
    # client.model_manager.model_validator = trainer.model_validator
    # client.pool = ThreadPool(len(client.servers))
    # client.communicator.ssl_args = trainer.client_config
    # client.communicator.secure_train = trainer.secure_train
    # client.communicator.model_manager = client.model_manager
    client.communicator.should_stop = False
    client.communicator.retry = int(math.ceil(float(retry_timeout) / 5))
    # client.communicator.outbound_filters = trainer.outbound_filters
    # client.communicator.inbound_filters = trainer.inbound_filters
    client.handlers = trainer.handlers
    # client.inbound_filters = trainer.inbound_filters
    client.executors = trainer.executors
    # client.task_inbound_filters = trainer.task_inbound_filters
    # client.task_outbound_filters = trainer.task_outbound_filters
    # client.secure_train = trainer.secure_train
    client.heartbeat_done = False
    # client.fl_ctx = FLContext()
