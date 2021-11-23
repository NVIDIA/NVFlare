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

import gc
import logging
import os
import pickle
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Tuple

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import MachineStatus, ReservedTopic, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.workspace import Workspace
from nvflare.fuel.hci.zip_utils import zip_directory_to_bytes
from nvflare.private.admin_defs import Message
from nvflare.private.fed.server.server_json_config import ServerJsonConfigurator
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import Widget, WidgetID

from .run_manager import RunManager
from .server_engine_internal_spec import EngineInfo, RunInfo, ServerEngineInternalSpec
from .server_status import ServerStatus


class ServerEngine(ServerEngineInternalSpec):
    def __init__(self, server, args, client_manager, workers=3):
        self.server = server
        self.args = args
        self.run_number = -1
        self.run_manager = None
        self.conf = None
        self.client_manager = client_manager

        self.widgets = {
            WidgetID.INFO_COLLECTOR: InfoCollector(),
            # WidgetID.FED_EVENT_RUNNER: ServerFedEventRunner()
        }

        self.engine_info = EngineInfo()

        assert workers >= 1, "workers must >= 1"
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.lock = Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.asked_to_stop = False

    def _get_server_app_folder(self):
        return "app_server"

    def _get_client_app_folder(self, client_name):
        return "app_" + client_name

    def _get_run_folder(self):
        return os.path.join(self.args.workspace, "run_" + str(self.run_number))

    def get_engine_info(self) -> EngineInfo:
        if self.run_number > 0:
            run_folder = os.path.join(self.args.workspace, "run_" + str(self.run_number))
            app_file = os.path.join(run_folder, "fl_app.txt")
            if os.path.exists(app_file):
                with open(app_file, "r") as f:
                    self.engine_info.app_name = f.readline().strip()
            else:
                self.engine_info.app_name = "?"

        return self.engine_info

    def get_run_info(self) -> RunInfo:
        if self.run_manager:
            return self.run_manager.get_run_info()
        else:
            return None

    def set_run_number(self, num):
        # status = self.server.status
        status = self.engine_info.status
        if status == MachineStatus.STARTING or status == MachineStatus.STARTED:
            return "run_number can not be changed during the FL app running."

        run_folder = os.path.join(self.args.workspace, "run_" + str(num))
        if os.path.exists(run_folder):
            self.run_number = num
            return "run number already exists. Set the FL run number to {}.".format(num)
        else:
            self.run_number = num
            os.makedirs(run_folder)
            return "Created a new run folder: run_{}".format(num)

    def delete_run_number(self, num):
        run_number_folder = os.path.join(self.args.workspace, "run_" + str(num))
        if os.path.exists(run_number_folder):
            shutil.rmtree(run_number_folder)
        return ""

    def get_run_number(self):
        return self.run_number

    def get_clients(self) -> [Client]:
        return list(self.client_manager.clients.values())

    def validate_clients(self, client_names: List[str]) -> Tuple[List[Client], List[str]]:
        return self._get_all_clients_from_inputs(client_names)

    def start_app_on_server(self) -> str:
        if self.run_number == -1:
            return "Please set a run number."

        status = self.engine_info.status
        if status == MachineStatus.STARTING or status == MachineStatus.STARTED:
            return "Server already starting or started."
        else:
            app_root = os.path.join(self._get_run_folder(), self._get_server_app_folder())
            if not os.path.exists(app_root):
                return "Server app does not exist. Please deploy the server app before starting."

            self.engine_info.status = MachineStatus.STARTING
            if self.server.enable_byoc:
                app_custom_folder = os.path.join(app_root, "custom")
                try:
                    sys.path.index(app_custom_folder)
                except ValueError:
                    self.remove_custom_path()
                    sys.path.append(app_custom_folder)

            # future = self.executor.submit(start_server_training, (self.server))
            future = self.executor.submit(
                lambda p: start_server_training(*p), [self.server, self.args, app_root, self.run_number]
            )

            start = time.time()
            # Wait for the server App to start properly
            while self.engine_info.status == MachineStatus.STARTING:
                time.sleep(0.3)
                if time.time() - start > 300.0:
                    return "Could not start the server app."

            if self.engine_info.status != MachineStatus.STARTED:
                return f"Failed to start server app: {self.engine_info.status}"

            return ""

    def remove_custom_path(self):
        regex = re.compile(".*/run_.*/custom")
        custom_paths = list(filter(regex.search, sys.path))
        for path in custom_paths:
            sys.path.remove(path)

    def abort_app_on_clients(self, clients: List[str]) -> str:
        status = self.engine_info.status
        if status == MachineStatus.STOPPED:
            return "Server app has not started."
        if status == MachineStatus.STARTING:
            return "Server app is starting, please wait for started before abort."
        return ""

    def abort_app_on_server(self) -> str:
        status = self.engine_info.status
        if status == MachineStatus.STOPPED:
            return "Server app has not started."
        if status == MachineStatus.STARTING:
            return "Server app is starting, please wait for started before abort."

        self.logger.info("Abort the server app run.")
        # self.server.stop_training()
        self.server.shutdown = True
        self.server.abort_run()
        return ""

    def check_app_start_readiness(self) -> str:
        status = self.engine_info.status
        if status != MachineStatus.STARTING and status != MachineStatus.STARTED:
            return "Server app has not started."
        return ""

    def shutdown_server(self) -> str:
        status = self.server.status
        if status == ServerStatus.STARTING:
            return "Server app is starting, please wait for started before shutdown."

        self.logger.info("FL server shutdown.")
        # self.server.fl_shutdown()
        # future = self.executor.submit(server_shutdown, (self.server))

        touch_file = os.path.join(self.args.workspace, "shutdown.fl")
        future = self.executor.submit(lambda p: server_shutdown(*p), [self.server, touch_file])
        return ""

    def restart_server(self) -> str:
        status = self.server.status
        if status == ServerStatus.STARTING:
            return "Server is starting, please wait for started before restart."

        self.logger.info("FL server restart.")

        touch_file = os.path.join(self.args.workspace, "restart.fl")
        future = self.executor.submit(lambda p: server_shutdown(*p), [self.server, touch_file])
        return ""

    def get_widget(self, widget_id: str) -> Widget:
        return self.widgets.get(widget_id)

    def get_client_name_from_token(self, token: str) -> str:
        client = self.server.client_manager.clients.get(token)
        if client:
            return client.name
        else:
            return ""

    def get_all_clients(self):
        return list(self.server.client_manager.clients.keys())

    def _get_client_from_name(self, client_name):
        for c in self.get_clients():
            if client_name == c.name:
                return c
        return None

    def _get_all_clients_from_inputs(self, inputs):
        clients = []
        invalid_inputs = []
        all_clients = self.get_clients()
        for item in inputs:
            client = self.client_manager.clients.get(item)
            # if item in self.get_all_clients():
            if client:
                clients.append(client)
            else:
                client = self._get_client_from_name(item)
                if client:
                    clients.append(client)
                else:
                    invalid_inputs.append(item)
        return clients, invalid_inputs

    def get_all_taskname(self) -> str:
        return self.server.server_name

    def get_app_data(self, app_name: str) -> Tuple[str, object]:
        if self.run_number == -1:
            return "Please set a FL run number.", None

        # folder = self._get_run_folder()
        # data = zip_directory_to_bytes(folder, self._get_client_app_folder(client_name))
        fullpath_src = os.path.join(self.server.admin_server.file_upload_dir, app_name)
        if not os.path.exists(fullpath_src):
            return f"App folder '{app_name}' does not exist in staging area.", None

        # folder = self.args.workspace
        # data = zip_directory_to_bytes(
        #     folder, os.path.join("run_" + str(self.run_number), self._get_client_app_folder(client_name))
        # )
        data = zip_directory_to_bytes(fullpath_src, "")
        return "", data

    def set_run_manager(self, run_manager: RunManager):
        self.run_manager = run_manager
        for _, widget in self.widgets.items():
            self.run_manager.add_handler(widget)

    def set_configurator(self, conf: ServerJsonConfigurator):
        if not isinstance(conf, ServerJsonConfigurator):
            raise TypeError("Must be a type of ServerJsonConfigurator.")
        self.conf = conf

    def build_component(self, config_dict):
        return self.conf.build_component(config_dict)

    def new_context(self) -> FLContext:
        if self.run_manager:
            return self.run_manager.new_context()
        else:
            return FLContext()

    def get_component(self, component_id: str) -> object:
        return self.run_manager.get_component(component_id)

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        self.run_manager.fire_event(event_type, fl_ctx)

    def get_staging_path_of_app(self, app_name: str) -> str:
        return os.path.join(self.server.admin_server.file_upload_dir, app_name)

    def deploy_app_to_server(self, app_name: str, app_staging_path: str) -> str:
        return self.deploy_app(app_name, "app_server")

    def prepare_deploy_app_to_client(self, app_name: str, app_staging_path: str, client_name: str) -> str:
        return self.deploy_app(app_name, "app_" + client_name)

    def get_workspace(self) -> Workspace:
        return self.run_manager.get_workspace()

    def ask_to_stop(self):
        self.asked_to_stop = True

    def deploy_app(self, src, dest):
        if self.run_number == -1:
            return "Please set a run number."

        fullpath_src = os.path.join(self.server.admin_server.file_upload_dir, src)
        fullpath_dest = os.path.join(self._get_run_folder(), dest)
        if not os.path.exists(fullpath_src):
            return f"App folder '{src}' does not exist in staging area."
        if os.path.exists(fullpath_dest):
            shutil.rmtree(fullpath_dest)
        shutil.copytree(fullpath_src, fullpath_dest)

        app_file = os.path.join(self._get_run_folder(), "fl_app.txt")
        if os.path.exists(app_file):
            os.remove(app_file)
        with open(app_file, "wt") as f:
            f.write(f"{src}")

        return ""

    def remove_clients(self, clients: List[str]) -> str:
        for client in clients:
            self._remove_dead_client(client)
        return ""

    def _remove_dead_client(self, token):
        client = self.server.client_manager.remove_client(token)
        # self.tokens.pop(token, None)
        self.server.remove_client_data(token)
        if self.server.admin_server:
            self.server.admin_server.client_dead(client.name)

    def register_aux_message_handler(self, topic: str, message_handle_func):
        self.run_manager.aux_runner.register_aux_message_handler(topic, message_handle_func)

    def send_aux_request(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        if not targets:
            targets = []
            for t in self.get_clients():
                targets.append(t.name)
        if targets:
            return self.run_manager.aux_runner.send_aux_request(
                targets=targets, topic=topic, request=request, timeout=timeout, fl_ctx=fl_ctx
            )
        else:
            return {}

    def aux_send(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        # Send the aux messages through admin_server
        request.set_peer_props(fl_ctx.get_all_public_props())

        message = Message(topic=ReservedTopic.AUX_COMMAND, body=pickle.dumps(request))
        requests = {}
        for n in targets:
            requests.update({n: message})

        replies = self.server.admin_server.send_requests(requests, timeout_secs=timeout)
        results = {}
        for r in replies:
            client_name = self.get_client_name_from_token(r.client_token)
            if r.reply:
                try:
                    results[client_name] = pickle.loads(r.reply.body)
                except BaseException:
                    results[client_name] = make_reply(ReturnCode.COMMUNICATION_ERROR)
                    self.logger.error(
                        f"Received unexpected reply from client: {client_name}, "
                        f"message body:{r.reply.body} processing topic:{topic}"
                    )
            else:
                results[client_name] = None

        return results

    def dispatch(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        return self.run_manager.aux_runner.dispatch(topic=topic, request=request, fl_ctx=fl_ctx)

    def close(self):
        self.executor.shutdown()


def start_server_training(server, args, app_root, run_number):

    restart_file = os.path.join(args.workspace, "restart.fl")
    if os.path.exists(restart_file):
        os.remove(restart_file)

    if os.path.exists(os.path.join(app_root, args.env)):
        env_config = args.env
    else:
        env_config = "/tmp/fl_server/environment.json"
    try:
        server_config_file_name = os.path.join(app_root, args.server_config)

        conf = ServerJsonConfigurator(
            config_file_name=server_config_file_name,
        )
        conf.configure()

        set_up_run_config(server, conf)

        server.start_run(run_number, app_root, conf, args)
    except BaseException as e:
        traceback.print_exc()
        logging.getLogger().warning("FL server execution exception: " + str(e))
    finally:
        server.status = ServerStatus.STOPPED
        server.engine.engine_info.status = MachineStatus.STOPPED
        server.stop_training()
        # if trainer:
        #     trainer.close()

        # Force garbage collection
        gc.collect()

    # return server.start()


def server_shutdown(server, touch_file):
    with open(touch_file, "a"):
        os.utime(touch_file, None)

    try:
        server.fl_shutdown()
        server.admin_server.stop()
    finally:
        sys.exit(2)


def copy_new_server_properties(server, new_server):
    # server.model_manager = new_server.model_manager
    # server.model_saver = new_server.model_saver
    server.builder = new_server.builder

    server.wait_after_min_clients = new_server.wait_after_min_clients

    server.outbound_filters = new_server.outbound_filters
    server.inbound_filters = new_server.inbound_filters
    server.cmd_modules = new_server.cmd_modules
    server.processors = new_server.processors

    # server.task_name = new_server.task_name
    server.min_num_clients = new_server.min_num_clients
    server.max_num_clients = new_server.max_num_clients
    server.current_round = new_server.current_round
    server.num_rounds = new_server.num_rounds
    server.start_round = new_server.start_round

    # server.heart_beat_timeout = new_server.heart_beat_timeout
    # server.handlers = new_server.handlers

    # clients = server.client_manager.clients
    # server.client_manager = new_server.client_manager
    # server.client_manager.clients = clients
    server.client_manager.min_num_clients = new_server.client_manager.min_num_clients
    server.client_manager.max_num_clients = new_server.client_manager.max_num_clients
    server.client_manager.logger = new_server.client_manager.logger
    server.client_manager.logger.disabled = False

    server.reset_tokens()
    server.contributed_clients.clear()
    # server.accumulator.clear()

    server.fl_ctx = new_server.fl_ctx

    server.controller = new_server.controller
    # server.model_aggregator = new_server.model_aggregator
    # server.model_saver = new_server.model_saver
    # server.shareable_generator = new_server.shareable_generator


def set_up_run_config(server, conf):
    server.heart_beat_timeout = conf.heartbeat_timeout
    server.runner_config = conf.runner_config
    server.handlers = conf.handlers
