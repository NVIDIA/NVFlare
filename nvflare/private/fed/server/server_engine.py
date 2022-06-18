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

import copy
import logging
import multiprocessing
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client as CommandClient
from multiprocessing.connection import Listener
from threading import Lock
from typing import Dict, List, Tuple

from nvflare.apis.client import Client
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import (
    AdminCommandNames,
    FLContextKey,
    MachineStatus,
    ReservedTopic,
    ReturnCode,
    RunProcessKey,
    ServerCommandKey,
    ServerCommandNames,
    SnapshotKey,
    WorkspaceConstants,
)
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.fl_snapshot import RunSnapshot
from nvflare.apis.impl.job_def_manager import JobDefManagerSpec
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.utils.common_utils import get_open_ports
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.apis.workspace import Workspace
from nvflare.fuel.hci.zip_utils import zip_directory_to_bytes
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.server.server_json_config import ServerJsonConfigurator
from nvflare.private.scheduler_constants import ShareableHeader
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import Widget, WidgetID

from .admin import ClientReply
from .client_manager import ClientManager
from .job_runner import JobRunner
from .run_manager import RunManager
from .server_engine_internal_spec import EngineInfo, RunInfo, ServerEngineInternalSpec
from .server_status import ServerStatus


class ClientConnection:
    def __init__(self, client):
        self.client = client

    def send(self, data):
        data = pickle.dumps(data)
        self.client.send(data)

    def recv(self):
        return self.client.recv()


class ServerEngine(ServerEngineInternalSpec):
    def __init__(self, server, args, client_manager: ClientManager, snapshot_persistor, workers=3):
        """Server engine.

        Args:
            server: server
            args: arguments
            client_manager (ClientManager): client manager.
            workers: number of worker threads.
        """
        # TODO:: clean up the server function / requirement here should be BaseServer
        self.server = server
        self.args = args
        self.run_processes = {}
        self.execution_exception_run_processes = {}
        self.run_manager = None
        self.conf = None
        # TODO:: does this class need client manager?
        self.client_manager = client_manager

        self.widgets = {
            WidgetID.INFO_COLLECTOR: InfoCollector(),
            # WidgetID.FED_EVENT_RUNNER: ServerFedEventRunner()
        }

        self.engine_info = EngineInfo()

        if not workers >= 1:
            raise ValueError("workers must >= 1 but got {}".format(workers))

        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.lock = Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.asked_to_stop = False
        self.snapshot_persistor = snapshot_persistor
        self.parent_conn = None
        self.parent_conn_lock = Lock()
        self.job_runner = None
        self.job_def_manager = None
        self.snapshot_lock = multiprocessing.Lock()

    def _get_server_app_folder(self):
        return WorkspaceConstants.APP_PREFIX + "server"

    def _get_client_app_folder(self, client_name):
        return WorkspaceConstants.APP_PREFIX + client_name

    def _get_run_folder(self, job_id):
        return os.path.join(self.args.workspace, WorkspaceConstants.WORKSPACE_PREFIX + str(job_id))

    def get_engine_info(self) -> EngineInfo:
        self.engine_info.app_names = {}
        if bool(self.run_processes):
            self.engine_info.status = MachineStatus.STARTED
        else:
            self.engine_info.status = MachineStatus.STOPPED

        for job_id, _ in self.run_processes.items():
            run_folder = os.path.join(self.args.workspace, WorkspaceConstants.WORKSPACE_PREFIX + str(job_id))
            app_file = os.path.join(run_folder, "fl_app.txt")
            if os.path.exists(app_file):
                with open(app_file, "r") as f:
                    self.engine_info.app_names[job_id] = f.readline().strip()
            else:
                self.engine_info.app_names[job_id] = "?"

        return self.engine_info

    def get_run_info(self) -> RunInfo:
        if self.run_manager:
            return self.run_manager.get_run_info()
        else:
            return None

    def create_parent_connection(self, port):
        while not self.parent_conn:
            try:
                address = ("localhost", port)
                self.parent_conn = CommandClient(address, authkey="parent process secret password".encode())
            except BaseException:
                time.sleep(1.0)
                pass

        threading.Thread(target=self.heartbeat_to_parent, args=[]).start()

    def heartbeat_to_parent(self):
        while True:
            try:
                with self.parent_conn_lock:
                    data = {ServerCommandKey.COMMAND: ServerCommandNames.HEARTBEAT, ServerCommandKey.DATA: {}}
                    self.parent_conn.send(data)
                time.sleep(1.0)
            except BaseException:
                # The parent process can not be reached. Terminate the child process.
                break
        # delay some time for the wrap up process before the child process self terminate.
        time.sleep(30)
        os.killpg(os.getpgid(os.getpid()), 9)

    def delete_job_id(self, num):
        job_id_folder = os.path.join(self.args.workspace, WorkspaceConstants.WORKSPACE_PREFIX + str(num))
        if os.path.exists(job_id_folder):
            shutil.rmtree(job_id_folder)
        return ""

    def get_clients(self) -> [Client]:
        return list(self.client_manager.get_clients().values())

    def validate_clients(self, client_names: List[str]) -> Tuple[List[Client], List[str]]:
        return self._get_all_clients_from_inputs(client_names)

    def start_app_on_server(self, run_number: str, job_id: str = None, job_clients=None, snapshot=None) -> str:
        if run_number in self.run_processes.keys():
            return f"Server run_{run_number} already started."
        else:
            app_root = os.path.join(self._get_run_folder(run_number), self._get_server_app_folder())
            if not os.path.exists(app_root):
                return "Server app does not exist. Please deploy the server app before starting."

            self.engine_info.status = MachineStatus.STARTING

            app_custom_folder = ""
            if self.server.enable_byoc:
                app_custom_folder = os.path.join(app_root, "custom")

            open_ports = get_open_ports(2)
            self._start_runner_process(
                self.args, app_root, run_number, app_custom_folder, open_ports, job_id, job_clients, snapshot
            )

            threading.Thread(target=self._listen_command, args=(open_ports[0], run_number)).start()

            self.engine_info.status = MachineStatus.STARTED
            return ""

    def _listen_command(self, listen_port, job_id):
        address = ("localhost", int(listen_port))
        listener = Listener(address, authkey="parent process secret password".encode())
        conn = listener.accept()

        while job_id in self.run_processes.keys():
            clients = self.run_processes.get(job_id).get(RunProcessKey.PARTICIPANTS)
            job_id = self.run_processes.get(job_id).get(RunProcessKey.JOB_ID)
            try:
                if conn.poll(0.1):
                    received_data = conn.recv()
                    command = received_data.get(ServerCommandKey.COMMAND)
                    data = received_data.get(ServerCommandKey.DATA)

                    if command == ServerCommandNames.GET_CLIENTS:
                        return_data = {ServerCommandKey.CLIENTS: clients, ServerCommandKey.JOB_ID: job_id}
                        conn.send(return_data)
                    elif command == ServerCommandNames.AUX_SEND:
                        targets = data.get("targets")
                        topic = data.get("topic")
                        request = data.get("request")
                        timeout = data.get("timeout")
                        fl_ctx = data.get("fl_ctx")
                        replies = self.aux_send(
                            targets=targets, topic=topic, request=request, timeout=timeout, fl_ctx=fl_ctx
                        )
                        conn.send(replies)
            except BaseException as e:
                self.logger.warning(f"Failed to process the child process command: {e}", exc_info=True)

    def wait_for_complete(self, job_id):
        while True:
            try:
                with self.lock:
                    command_conn = self.get_command_conn(job_id)
                    if command_conn:
                        data = {ServerCommandKey.COMMAND: ServerCommandNames.HEARTBEAT, ServerCommandKey.DATA: {}}
                        command_conn.send(data)
                time.sleep(1.0)
            except BaseException:
                with self.lock:
                    run_process_info = self.run_processes.pop(job_id)
                    return_code = run_process_info[RunProcessKey.CHILD_PROCESS].poll()
                    # if process exit but with Execution exception
                    if return_code and return_code != 0:
                        self.execution_exception_run_processes[job_id] = run_process_info
                self.engine_info.status = MachineStatus.STOPPED
                break

    def _start_runner_process(
        self, args, app_root, run_number, app_custom_folder, open_ports, job_id, job_clients, snapshot
    ):
        new_env = os.environ.copy()
        if app_custom_folder != "":
            new_env["PYTHONPATH"] = new_env.get("PYTHONPATH", "") + os.pathsep + app_custom_folder

        listen_port = open_ports[1]
        if snapshot:
            restore_snapshot = True
        else:
            restore_snapshot = False
        command_options = ""
        for t in args.set:
            command_options += " " + t
        command = (
            sys.executable
            + " -m nvflare.private.fed.app.server.runner_process -m "
            + args.workspace
            + " -s fed_server.json -r "
            + app_root
            + " -n "
            + str(run_number)
            + " -p "
            + str(listen_port)
            + " -c "
            + str(open_ports[0])
            + " --set"
            + command_options
            + " print_conf=True restore_snapshot="
            + str(restore_snapshot)
        )
        # use os.setsid to create new process group ID

        process = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, env=new_env)

        if not job_id:
            job_id = ""
        if not job_clients:
            job_clients = self.client_manager.clients

        with self.lock:
            self.run_processes[run_number] = {
                RunProcessKey.LISTEN_PORT: listen_port,
                RunProcessKey.CONNECTION: None,
                RunProcessKey.CHILD_PROCESS: process,
                RunProcessKey.JOB_ID: job_id,
                RunProcessKey.PARTICIPANTS: job_clients,
            }

        threading.Thread(target=self.wait_for_complete, args=[run_number]).start()
        return process

    def get_job_clients(self, client_sites):
        job_clients = {}
        if client_sites:
            for site, dispatch_info in client_sites.items():
                client = self.get_client_from_name(site)
                if client:
                    job_clients[client.token] = client
        return job_clients

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

    def abort_app_on_server(self, job_id: str) -> str:
        if job_id not in self.run_processes.keys():
            return "Server app has not started."

        self.logger.info("Abort the server app run.")

        try:
            with self.lock:
                command_conn = self.get_command_conn(job_id)
                if command_conn:
                    data = {ServerCommandKey.COMMAND: AdminCommandNames.ABORT, ServerCommandKey.DATA: {}}
                    command_conn.send(data)
                    status_message = command_conn.recv()
                    self.logger.info(f"Abort server: {status_message}")
        except BaseException:
            with self.lock:
                child_process = self.run_processes.get(job_id, {}).get(RunProcessKey.CHILD_PROCESS, None)
                if child_process:
                    child_process.terminate()
        finally:
            with self.lock:
                self.run_processes.pop(job_id)

        self.engine_info.status = MachineStatus.STOPPED
        return ""

    def check_app_start_readiness(self, job_id: str) -> str:
        if job_id not in self.run_processes.keys():
            return f"Server app run_{job_id} has not started."
        return ""

    def shutdown_server(self) -> str:
        status = self.server.status
        if status == ServerStatus.STARTING:
            return "Server app is starting, please wait for started before shutdown."

        self.logger.info("FL server shutdown.")

        touch_file = os.path.join(self.args.workspace, "shutdown.fl")
        _ = self.executor.submit(lambda p: server_shutdown(*p), [self.server, touch_file])
        while self.server.status != ServerStatus.SHUTDOWN:
            time.sleep(1.0)
        return ""

    def restart_server(self) -> str:
        status = self.server.status
        if status == ServerStatus.STARTING:
            return "Server is starting, please wait for started before restart."

        self.logger.info("FL server restart.")

        touch_file = os.path.join(self.args.workspace, "restart.fl")
        _ = self.executor.submit(lambda p: server_shutdown(*p), [self.server, touch_file])
        while self.server.status != ServerStatus.SHUTDOWN:
            time.sleep(1.0)
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

    def get_client_from_name(self, client_name):
        for c in self.get_clients():
            if client_name == c.name:
                return c
        return None

    def _get_all_clients_from_inputs(self, inputs):
        clients = []
        invalid_inputs = []
        for item in inputs:
            client = self.client_manager.clients.get(item)
            # if item in self.get_all_clients():
            if client:
                clients.append(client)
            else:
                client = self.get_client_from_name(item)
                if client:
                    clients.append(client)
                else:
                    invalid_inputs.append(item)
        return clients, invalid_inputs

    def get_app_data(self, app_name: str) -> Tuple[str, object]:
        fullpath_src = os.path.join(self.server.admin_server.file_upload_dir, app_name)
        if not os.path.exists(fullpath_src):
            return f"App folder '{app_name}' does not exist in staging area.", None

        data = zip_directory_to_bytes(fullpath_src, "")
        return "", data

    def get_app_run_info(self, job_id) -> RunInfo:
        run_info = None
        try:
            with self.lock:
                command_conn = self.get_command_conn(job_id)
                if command_conn:
                    data = {ServerCommandKey.COMMAND: ServerCommandNames.GET_RUN_INFO, ServerCommandKey.DATA: {}}
                    command_conn.send(data)
                    run_info = command_conn.recv()
        except BaseException:
            self.logger.error(f"Failed to get_app_run_info from run_{job_id}")

        return run_info

    def set_run_manager(self, run_manager: RunManager):
        self.run_manager = run_manager
        for _, widget in self.widgets.items():
            self.run_manager.add_handler(widget)

    def set_job_runner(self, job_runner: JobRunner, job_manager: JobDefManagerSpec):
        self.job_runner = job_runner
        self.job_def_manager = job_manager

    def set_configurator(self, conf: ServerJsonConfigurator):
        if not isinstance(conf, ServerJsonConfigurator):
            raise TypeError("conf must be ServerJsonConfigurator but got {}".format(type(conf)))
        self.conf = conf

    def build_component(self, config_dict):
        return self.conf.build_component(config_dict)

    def new_context(self) -> FLContext:
        if self.run_manager:
            return self.run_manager.new_context()
        else:
            # return FLContext()
            return FLContextManager(
                engine=self, identity_name=self.server.project_name, job_id="", public_stickers={}, private_stickers={}
            ).new_context()

    def get_component(self, component_id: str) -> object:
        return self.run_manager.get_component(component_id)

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        self.run_manager.fire_event(event_type, fl_ctx)

    def get_staging_path_of_app(self, app_name: str) -> str:
        return os.path.join(self.server.admin_server.file_upload_dir, app_name)

    def deploy_app_to_server(self, run_destination: str, app_name: str, app_staging_path: str) -> str:
        return self.deploy_app(run_destination, app_name, WorkspaceConstants.APP_PREFIX + "server")

    def get_workspace(self) -> Workspace:
        return self.run_manager.get_workspace()

    def ask_to_stop(self):
        self.asked_to_stop = True

    def deploy_app(self, job_id, src, dst):
        fullpath_src = os.path.join(self.server.admin_server.file_upload_dir, src)
        fullpath_dst = os.path.join(self._get_run_folder(job_id), dst)
        if not os.path.exists(fullpath_src):
            return f"App folder '{src}' does not exist in staging area."
        if os.path.exists(fullpath_dst):
            shutil.rmtree(fullpath_dst)
        shutil.copytree(fullpath_src, fullpath_dst)

        app_file = os.path.join(self._get_run_folder(job_id), "fl_app.txt")
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
        _ = self.server.client_manager.remove_client(token)
        self.server.remove_client_data(token)
        if self.server.admin_server:
            self.server.admin_server.client_dead(token)

    def register_aux_message_handler(self, topic: str, message_handle_func):
        self.run_manager.aux_runner.register_aux_message_handler(topic, message_handle_func)

    def send_aux_request(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        try:
            if not targets:
                self.sync_clients_from_main_process()
                targets = []
                for t in self.get_clients():
                    targets.append(t.name)
            if targets:
                return self.run_manager.aux_runner.send_aux_request(
                    targets=targets, topic=topic, request=request, timeout=timeout, fl_ctx=fl_ctx
                )
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Failed to send the aux_message: {topic} with exception: {e}.")

    def sync_clients_from_main_process(self):
        with self.parent_conn_lock:
            data = {ServerCommandKey.COMMAND: ServerCommandNames.GET_CLIENTS, ServerCommandKey.DATA: {}}
            self.parent_conn.send(data)
            return_data = self.parent_conn.recv()
            clients = return_data.get(ServerCommandKey.CLIENTS)
            self.client_manager.clients = clients

    def parent_aux_send(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        with self.parent_conn_lock:
            data = {
                ServerCommandKey.COMMAND: ServerCommandNames.AUX_SEND,
                ServerCommandKey.DATA: {
                    "targets": targets,
                    "topic": topic,
                    "request": request,
                    "timeout": timeout,
                    "fl_ctx": get_serializable_data(fl_ctx),
                },
            }
            self.parent_conn.send(data)
            return_data = self.parent_conn.recv()
            return return_data

    def aux_send(self, targets: [], topic: str, request: Shareable, timeout: float, fl_ctx: FLContext) -> dict:
        # Send the aux messages through admin_server
        request.set_peer_props(fl_ctx.get_all_public_props())

        message = Message(topic=ReservedTopic.AUX_COMMAND, body=pickle.dumps(request))
        message.set_header(RequestHeader.JOB_ID, str(fl_ctx.get_prop(FLContextKey.CURRENT_RUN)))
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

    def get_command_conn(self, job_id):
        # this function need to be called with self.lock
        port = self.run_processes.get(job_id, {}).get(RunProcessKey.LISTEN_PORT)
        command_conn = self.run_processes.get(job_id, {}).get(RunProcessKey.CONNECTION, None)

        if not command_conn:
            try:
                address = ("localhost", port)
                command_conn = CommandClient(address, authkey="client process secret password".encode())
                command_conn = ClientConnection(command_conn)
                self.run_processes[job_id][RunProcessKey.CONNECTION] = command_conn
            except Exception:
                pass
        return command_conn

    def persist_components(self, fl_ctx: FLContext, completed: bool):

        # Call the State Persistor to persist all the component states
        # 1. call every component to generate the component states data
        #    Make sure to include the current round number
        # 2. call persistence API to save the component states

        try:
            job_id = fl_ctx.get_job_id()
            snapshot = RunSnapshot(job_id)
            for component_id, component in self.run_manager.components.items():
                if isinstance(component, FLComponent):
                    snapshot.set_component_snapshot(
                        component_id=component_id, component_state=component.get_persist_state(fl_ctx)
                    )

            snapshot.set_component_snapshot(
                component_id=SnapshotKey.FL_CONTEXT, component_state=copy.deepcopy(get_serializable_data(fl_ctx).props)
            )

            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
            data = zip_directory_to_bytes(workspace.get_run_dir(fl_ctx.get_prop(FLContextKey.CURRENT_RUN)), "")
            snapshot.set_component_snapshot(component_id=SnapshotKey.WORKSPACE, component_state={"content": data})

            job_info = fl_ctx.get_prop(FLContextKey.JOB_INFO)
            if not job_info:
                with self.parent_conn_lock:
                    data = {ServerCommandKey.COMMAND: ServerCommandNames.GET_CLIENTS, ServerCommandKey.DATA: {}}
                    self.parent_conn.send(data)
                    return_data = self.parent_conn.recv()
                    job_id = return_data.get(ServerCommandKey.JOB_ID)
                    job_clients = return_data.get(ServerCommandKey.CLIENTS)
                    fl_ctx.set_prop(FLContextKey.JOB_INFO, (job_id, job_clients))
            else:
                (job_id, job_clients) = job_info
            snapshot.set_component_snapshot(
                component_id=SnapshotKey.JOB_INFO,
                component_state={SnapshotKey.JOB_CLIENTS: job_clients, SnapshotKey.JOB_ID: job_id},
            )

            snapshot.completed = completed

            self.server.snapshot_location = self.snapshot_persistor.save(snapshot=snapshot)
            if not completed:
                self.logger.info(f"persist the snapshot to: {self.server.snapshot_location}")
            else:
                self.logger.info(f"The snapshot: {self.server.snapshot_location} has been removed.")
        except BaseException as e:
            self.logger.error(f"Failed to persist the components. {str(e)}")

    def restore_components(self, snapshot: RunSnapshot, fl_ctx: FLContext):
        for component_id, component in self.run_manager.components.items():
            component.restore(snapshot.get_component_snapshot(component_id=component_id), fl_ctx)

        fl_ctx.props.update(snapshot.get_component_snapshot(component_id=SnapshotKey.FL_CONTEXT))

    def dispatch(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        return self.run_manager.aux_runner.dispatch(topic=topic, request=request, fl_ctx=fl_ctx)

    def show_stats(self, job_id):
        stats = None
        try:
            with self.lock:
                command_conn = self.get_command_conn(job_id)
                if command_conn:
                    data = {ServerCommandKey.COMMAND: ServerCommandNames.SHOW_STATS, ServerCommandKey.DATA: {}}
                    command_conn.send(data)
                    stats = command_conn.recv()
        except BaseException:
            self.logger.error(f"Failed to get_stats from run_{job_id}")

        return stats

    def get_errors(self, job_id):
        stats = None
        try:
            with self.lock:
                command_conn = self.get_command_conn(job_id)
                if command_conn:
                    data = {ServerCommandKey.COMMAND: ServerCommandNames.GET_ERRORS, ServerCommandKey.DATA: {}}
                    command_conn.send(data)
                    stats = command_conn.recv()
        except BaseException:
            self.logger.error(f"Failed to get_stats from run_{job_id}")

        return stats

    def _send_admin_requests(self, requests, timeout_secs=10) -> List[ClientReply]:
        return self.server.admin_server.send_requests(requests, timeout_secs=timeout_secs)

    def check_client_resources(self, resource_reqs) -> Dict[str, Tuple[bool, str]]:
        requests = {}
        for site_name, resource_requirements in resource_reqs.items():
            # assume server resource is unlimited
            if site_name == "server":
                continue
            request = Message(topic=TrainingTopic.CHECK_RESOURCE, body=pickle.dumps(resource_requirements))
            client = self.get_client_from_name(site_name)
            if client:
                requests.update({client.token: request})
        replies = []
        if requests:
            replies = self._send_admin_requests(requests, 15)
        result = {}
        for r in replies:
            site_name = self.get_client_name_from_token(r.client_token)
            if r.reply:
                resp = pickle.loads(r.reply.body)
                result[site_name] = (
                    resp.get_header(ShareableHeader.CHECK_RESOURCE_RESULT, False),
                    resp.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, ""),
                )
            else:
                result[site_name] = (False, "")
        return result

    def cancel_client_resources(
        self, resource_check_results: Dict[str, Tuple[bool, str]], resource_reqs: Dict[str, dict]
    ):
        requests = {}
        for site_name, result in resource_check_results.items():
            check_result, token = result
            if check_result and token:
                resource_requirements = resource_reqs[site_name]
                request = Message(topic=TrainingTopic.CANCEL_RESOURCE, body=pickle.dumps(resource_requirements))
                request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, token)
                client = self.get_client_from_name(site_name)
                if client:
                    requests.update({client.token: request})
        if requests:
            _ = self._send_admin_requests(requests)

    def start_client_job(self, job_id, client_sites):
        requests = {}
        for site, dispatch_info in client_sites.items():
            resource_requirement = dispatch_info.resource_requirements
            token = dispatch_info.token
            request = Message(topic=TrainingTopic.START_JOB, body=pickle.dumps(resource_requirement))
            request.set_header(RequestHeader.JOB_ID, job_id)
            request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, token)
            client = self.get_client_from_name(site)
            if client:
                requests.update({client.token: request})
        replies = []
        if requests:
            replies = self._send_admin_requests(requests, timeout_secs=20)
        return replies

    def stop_all_jobs(self):
        fl_ctx = self.new_context()
        self.job_runner.stop_all_runs(fl_ctx)

    def close(self):
        self.executor.shutdown()


def server_shutdown(server, touch_file):
    with open(touch_file, "a"):
        os.utime(touch_file, None)

    try:
        server.fl_shutdown()
        server.admin_server.stop()
        time.sleep(3.0)
    finally:
        server.status = ServerStatus.SHUTDOWN
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
