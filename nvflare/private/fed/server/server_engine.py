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
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Tuple

from nvflare.apis.client import Client
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import (
    AdminCommandNames,
    FLContextKey,
    MachineStatus,
    RunProcessKey,
    ServerCommandKey,
    ServerCommandNames,
    SiteType,
    SnapshotKey,
    WorkspaceConstants,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_snapshot import RunSnapshot
from nvflare.apis.impl.job_def_manager import JobDefManagerSpec
from nvflare.apis.job_def import Job
from nvflare.apis.job_launcher_spec import JobLauncherSpec, JobProcessArgs
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import ConsumerFactory, ObjectProducer, StreamableEngine, StreamContext
from nvflare.apis.utils.fl_context_utils import gen_new_peer_ctx, get_serializable_data
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellMsgReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.zip_utils import zip_directory_to_bytes
from nvflare.private.admin_defs import Message, MsgHeader
from nvflare.private.aux_runner import AuxMsgTarget
from nvflare.private.defs import (
    AUTH_CLIENT_NAME_FOR_SJ,
    CellChannel,
    CellMessageHeaderKeys,
    RequestHeader,
    TrainingTopic,
    new_cell_message,
)
from nvflare.private.fed.server.server_json_config import ServerJsonConfigurator
from nvflare.private.fed.utils.fed_utils import (
    get_job_launcher,
    get_return_code,
    security_close,
    set_message_security_data,
)
from nvflare.private.scheduler_constants import ShareableHeader
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import Widget, WidgetID

from .client_manager import ClientManager
from .job_runner import JobRunner
from .message_send import ClientReply
from .run_info import RunInfo
from .run_manager import RunManager
from .server_commands import ServerCommands
from .server_engine_internal_spec import EngineInfo, ServerEngineInternalSpec
from .server_status import ServerStatus


class ServerEngine(ServerEngineInternalSpec, StreamableEngine):
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
        self.exception_run_processes = {}
        self.run_manager = None
        self.conf = None
        self.cell = None
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
        self.logger = get_obj_logger(self)

        self.asked_to_stop = False
        self.snapshot_persistor = snapshot_persistor
        self.job_runner = None
        self.job_def_manager = None

        self.kv_list = parse_vars(args.set)

    def has_relays(self):
        if not self.client_manager:
            return False
        else:
            return self.client_manager.has_relays()

    def _get_run_folder(self, job_id):
        workspace = Workspace(self.args.workspace)
        return workspace.get_run_dir(job_id)

    def get_engine_info(self) -> EngineInfo:
        self.engine_info.app_names = {}
        if bool(self.run_processes):
            self.engine_info.status = MachineStatus.STARTED
        else:
            self.engine_info.status = MachineStatus.STOPPED

        keys = list(self.run_processes.keys())
        for job_id in keys:
            run_folder = self._get_run_folder(job_id)
            app_file = os.path.join(run_folder, "fl_app.txt")
            if os.path.exists(app_file):
                with open(app_file, "r") as f:
                    self.engine_info.app_names[job_id] = f.readline().strip()
            else:
                self.engine_info.app_names[job_id] = "?"

        return self.engine_info

    def get_run_info(self) -> Optional[RunInfo]:
        if self.run_manager:
            run_info: RunInfo = self.run_manager.get_run_info()
            return run_info
        return None

    def delete_job_id(self, num):
        job_id_folder = self._get_run_folder(str(num))
        if os.path.exists(job_id_folder):
            shutil.rmtree(job_id_folder)
        return ""

    def get_clients(self) -> [Client]:
        return list(self.client_manager.get_clients().values())

    def validate_targets(self, client_names: List[str]) -> Tuple[List[Client], List[str]]:
        return self.client_manager.get_all_clients_from_inputs(client_names)

    def start_app_on_server(self, fl_ctx: FLContext, job: Job = None, job_clients=None, snapshot=None) -> str:
        if not isinstance(job, Job):
            return "Must provide a job object to start the server app."

        if job.job_id in self.run_processes.keys():
            return f"Server run: {job.job_id} already started."
        else:
            workspace = Workspace(root_dir=self.args.workspace, site_name=SiteType.SERVER)
            app_root = workspace.get_app_dir(job.job_id)
            if not os.path.exists(app_root):
                return "Server app does not exist. Please deploy the server app before starting."

            self.engine_info.status = MachineStatus.STARTING

            self._start_runner_process(job, job_clients, snapshot, fl_ctx)

            self.engine_info.status = MachineStatus.STARTED
            return ""

    def remove_exception_process(self, job_id):
        with self.lock:
            if job_id in self.exception_run_processes:
                self.exception_run_processes.pop(job_id)

    def wait_for_complete(self, workspace, job_id, process):
        process.wait()
        run_process_info = self.run_processes.get(job_id)
        if run_process_info:
            # Wait for the job process to finish UPDATE_RUN_STATUS process
            start_time = time.time()
            max_wait = 2.0
            while True:
                process_finished = run_process_info.get(RunProcessKey.PROCESS_FINISHED, False)
                if process_finished:
                    break
                if time.time() - start_time >= max_wait:
                    self.logger.debug(f"Job:{job_id} UPDATE_RUN_STATUS didn't finish fast enough.")
                    break
                time.sleep(0.1)
            with self.lock:
                return_code = get_return_code(process, job_id, workspace, self.logger)
                # if process exit but with Execution exception
                if return_code and return_code != 0:
                    self.logger.info(f"Job: {job_id} child process exit with return code {return_code}")
                    run_process_info[RunProcessKey.PROCESS_RETURN_CODE] = return_code
                    if job_id not in self.exception_run_processes:
                        self.exception_run_processes[job_id] = run_process_info
                self.run_processes.pop(job_id, None)
        self.engine_info.status = MachineStatus.STOPPED

    def _start_runner_process(self, job, job_clients, snapshot, fl_ctx: FLContext):
        job_launcher: JobLauncherSpec = get_job_launcher(job.meta, fl_ctx)
        if snapshot:
            restore_snapshot = True
        else:
            restore_snapshot = False

        # Job process args are the same for all job launchers! Letting each job launcher compute the job
        # args would be error-prone and would require access to internal server components (
        # e.g. cell, server_state, self.server, etc.), which violates component layering.
        #
        # We prepare job process args here and save the prepared result in the fl_ctx.
        # This way, the job launcher won't need to compute these args again.
        # The job launcher will only need to use the args properly to launch the job process!
        #
        # Each arg is a tuple of (arg_option, arg_value).
        # Note that the arg_option is fixed for each arg, and is not launcher specific!
        workspace_obj: Workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        args = fl_ctx.get_prop(FLContextKey.ARGS)
        server = fl_ctx.get_prop(FLContextKey.SITE_OBJ)
        job_id = job.job_id
        app_root = workspace_obj.get_app_dir(job_id)
        cell = server.cell
        server_state = server.server_state
        command_options = ""
        for t in args.set:
            command_options += " " + t
        command_options += f" restore_snapshot={restore_snapshot} print_conf=True"
        args.set.append("print_conf=True")
        args.set.append(f"restore_snapshot={restore_snapshot}")

        # create token and signature for SJ
        token = job_id  # use the run_number as the auth token
        client_name = AUTH_CLIENT_NAME_FOR_SJ
        signature = self.server.sign_auth_token(client_name, token)

        job_args = {
            JobProcessArgs.JOB_ID: ("-n", job_id),
            JobProcessArgs.EXE_MODULE: ("-m", "nvflare.private.fed.app.server.runner_process"),
            JobProcessArgs.WORKSPACE: ("-m", args.workspace),
            JobProcessArgs.STARTUP_CONFIG_FILE: ("-s", "fed_server.json"),
            JobProcessArgs.APP_ROOT: ("-r", app_root),
            JobProcessArgs.HA_MODE: ("--ha_mode", server.ha_mode),
            JobProcessArgs.AUTH_TOKEN: ("-t", token),
            JobProcessArgs.TOKEN_SIGNATURE: ("-ts", signature),
            JobProcessArgs.PARENT_URL: ("-p", str(cell.get_internal_listener_url())),
            JobProcessArgs.ROOT_URL: ("-u", str(cell.get_root_url_for_child())),
            JobProcessArgs.SERVICE_HOST: ("--host", str(server_state.host)),
            JobProcessArgs.SERVICE_PORT: ("--port", str(server_state.service_port)),
            JobProcessArgs.SSID: ("--ssid", str(server_state.ssid)),
            JobProcessArgs.OPTIONS: ("--set", command_options),
        }
        fl_ctx.set_prop(key=FLContextKey.JOB_PROCESS_ARGS, value=job_args, private=True, sticky=False)
        job_handle = job_launcher.launch_job(job.meta, fl_ctx)
        self.logger.info(f"Launch job_id: {job.job_id}  with job launcher: {type(job_launcher)} ")

        if not job_clients:
            job_clients = self.client_manager.clients

        with self.lock:
            self.run_processes[job.job_id] = {
                RunProcessKey.JOB_HANDLE: job_handle,
                RunProcessKey.JOB_ID: job.job_id,
                RunProcessKey.PARTICIPANTS: job_clients,
            }

        threading.Thread(target=self.wait_for_complete, args=[args.workspace, job.job_id, job_handle]).start()
        return job_handle

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

    def abort_app_on_server(self, job_id: str, turn_to_cold: bool = False) -> str:

        self.logger.info("Abort the server app run.")
        command_data = Shareable()
        command_data.set_header(ServerCommandKey.TURN_TO_COLD, turn_to_cold)

        try:
            status_message = self.send_command_to_child_runner_process(
                job_id=job_id,
                command_name=AdminCommandNames.ABORT,
                command_data=command_data,
                timeout=1.0,
                optional=True,
            )
            self.logger.info(f"Abort server status: {status_message}")
        except Exception:
            with self.lock:
                child_process = self.run_processes.get(job_id, {}).get(RunProcessKey.JOB_HANDLE, None)
                if child_process:
                    child_process.terminate()
        finally:
            threading.Thread(target=self._remove_run_processes, args=[job_id]).start()

        self.engine_info.status = MachineStatus.STOPPED
        return ""

    def _remove_run_processes(self, job_id):
        # wait for the run process to gracefully terminated, and ensure to remove from run_processes.
        max_wait = 5.0
        start = time.time()
        while True:
            if job_id not in self.run_processes:
                # job already gone
                return
            if time.time() - start >= max_wait:
                break
            time.sleep(0.1)
        self.run_processes.pop(job_id, None)

    def check_app_start_readiness(self, job_id: str) -> str:
        if job_id not in self.run_processes.keys():
            return f"Server app run: {job_id} has not started."
        return ""

    def shutdown_server(self) -> str:
        status = self.server.status
        if status == ServerStatus.STARTING:
            return "Server app is starting, please wait for started before shutdown."

        self.logger.info("FL server shutdown.")

        touch_file = os.path.join(self.args.workspace, "shutdown.fl")
        _ = self.executor.submit(lambda p: server_shutdown(*p), [self.server, touch_file])
        # while self.server.status != ServerStatus.SHUTDOWN:
        #     time.sleep(1.0)
        return ""

    def restart_server(self) -> str:
        status = self.server.status
        if status == ServerStatus.STARTING:
            return "Server is starting, please wait for started before restart."

        self.logger.info("FL server restart.")

        touch_file = os.path.join(self.args.workspace, "restart.fl")
        _ = self.executor.submit(lambda p: server_shutdown(*p), [self.server, touch_file])
        # while self.server.status != ServerStatus.SHUTDOWN:
        #     time.sleep(1.0)
        return ""

    def get_widget(self, widget_id: str) -> Widget:
        return self.widgets.get(widget_id)

    def get_client_name_from_token(self, token: str) -> str:
        client = self.server.client_manager.clients.get(token)
        if client:
            return client.name
        else:
            return ""

    def get_client_from_name(self, client_name):
        return self.client_manager.get_client_from_name(client_name)

    def get_app_data(self, app_name: str) -> Tuple[str, object]:
        fullpath_src = os.path.join(self.server.admin_server.file_upload_dir, app_name)
        if not os.path.exists(fullpath_src):
            return f"App folder '{app_name}' does not exist in staging area.", None

        data = zip_directory_to_bytes(fullpath_src, "")
        return "", data

    def get_app_run_info(self, job_id) -> Optional[RunInfo]:
        run_info = None
        try:
            run_info = self.send_command_to_child_runner_process(
                job_id=job_id,
                command_name=ServerCommandNames.GET_RUN_INFO,
                command_data={},
            )
        except Exception:
            self.logger.error(f"Failed to get_app_run_info for run: {job_id}")
        return run_info

    def send_app_command(self, job_id: str, topic: str, cmd_data, timeout: float) -> Shareable:
        cmd = Shareable()
        cmd[ServerCommandKey.TOPIC] = topic
        cmd[ServerCommandKey.DATA] = cmd_data
        try:
            result = self.send_command_to_child_runner_process(
                job_id=job_id,
                command_name=ServerCommandNames.APP_COMMAND,
                command_data=cmd,
                timeout=timeout,
            )
        except Exception as ex:
            self.logger.error(f"Exception sending app command to SJ {job_id}: {secure_format_exception(ex)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        return result

    def set_run_manager(self, run_manager: RunManager):
        self.logger.debug("set_run_manager is called")
        self.run_manager = run_manager

        # we set the run_manager's cell if we have the cell.
        if self.cell:
            self.run_manager.cell = self.cell

        for _, widget in self.widgets.items():
            self.run_manager.add_handler(widget)

    def get_cell(self):
        return self.cell

    def initialize_comm(self, cell: Cell):
        """This is called when the communication cell has been created.
        We will set up aux message handler here.

        Args:
            cell:

        Returns:

        """
        self.logger.debug("initialize_comm called!")
        self.cell = cell
        if self.run_manager:
            # Note that the aux_runner is created with the self.run_manager as the "engine".
            # We must set the cell in it; otherwise it won't be able to send messages.
            # The timing of the creation of the run_manager and the cell is not deterministic, we set the cell here
            # only if the run_manager has been created.
            self.run_manager.cell = cell

        cell.register_request_cb(
            channel=CellChannel.AUX_COMMUNICATION,
            topic="*",
            cb=self._handle_aux_message,
        )

    def _handle_aux_message(self, request: CellMessage) -> CellMessage:
        assert isinstance(request, CellMessage), "request must be CellMessage but got {}".format(type(request))
        data = request.payload

        topic = request.get_header(MessageHeaderKey.TOPIC)
        with self.new_context() as fl_ctx:
            reply = self.run_manager.aux_runner.dispatch(topic=topic, request=data, fl_ctx=fl_ctx)
            shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
            reply.set_peer_context(shared_fl_ctx)

            if reply is not None:
                return_message = new_cell_message({}, reply)
                return_message.set_header(MessageHeaderKey.RETURN_CODE, CellMsgReturnCode.OK)
            else:
                return_message = new_cell_message({}, None)
            return return_message

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
            # this call should never be made before the run_manager is created!
            self.logger.warning("no run_manager in Server Engine.")
            return FLContext()

    def add_component(self, component_id: str, component):
        self.server.runner_config.add_component(component_id, component)

    def get_component(self, component_id: str) -> object:
        return self.run_manager.get_component(component_id)

    def fire_event(self, event_type: str, fl_ctx: FLContext):
        self.run_manager.fire_event(event_type, fl_ctx)

    def get_staging_path_of_app(self, app_name: str) -> str:
        return os.path.join(self.server.admin_server.file_upload_dir, app_name)

    def deploy_app_to_server(self, run_destination: str, app_name: str, app_staging_path: str) -> str:
        return self.deploy_app(run_destination, app_name, WorkspaceConstants.APP_PREFIX + SiteType.SERVER)

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

    def send_aux_request(
        self,
        targets: [],
        topic: str,
        request: Shareable,
        timeout: float,
        fl_ctx: FLContext,
        optional=False,
        secure=False,
    ) -> dict:
        try:
            return self.send_aux_to_targets(targets, topic, request, timeout, fl_ctx, optional, secure)
        except Exception as e:
            self.logger.error(f"Failed to send the aux_message: {topic} with exception: {secure_format_exception(e)}.")

    def multicast_aux_requests(
        self,
        topic: str,
        target_requests: Dict[str, Shareable],
        timeout: float,
        fl_ctx: FLContext,
        optional: bool = False,
        secure: bool = False,
    ) -> dict:
        if not target_requests:
            return {}

        aux_target_reqs = []
        for name, req in target_requests.items():
            amt = self._get_aux_msg_target(name)
            if not amt:
                self.logger.error(f"unknown AuxMessage target {name}")
            else:
                aux_target_reqs.append((amt, req))

        if not aux_target_reqs:
            return {}

        return self.run_manager.aux_runner.multicast_aux_requests(
            topic=topic,
            target_requests=aux_target_reqs,
            timeout=timeout,
            fl_ctx=fl_ctx,
            optional=optional,
            secure=secure,
        )

    def _get_aux_msg_target(self, name: str):
        if name.lower() == SiteType.SERVER:
            return AuxMsgTarget.server_target()

        c = self.get_client_from_name(name)
        if c:
            return AuxMsgTarget.client_target(c)
        else:
            return None

    def _to_aux_msg_targets(self, target_names: List[str]):
        msg_targets = []
        if not target_names:
            # all clients
            for c in self.get_clients():
                msg_targets.append(AuxMsgTarget.client_target(c))
        elif not isinstance(target_names, list):
            raise TypeError(f"invalid target_names {type(target_names)}")
        else:
            # this is a list of targets: check targets
            for t in target_names:
                if not isinstance(t, str):
                    raise TypeError(f"target name must be str but got {type(t)}")

                amt = self._get_aux_msg_target(t)
                if not amt:
                    self.logger.error(f"invalid target {t}")
                    return {}
                else:
                    msg_targets.append(amt)
        return msg_targets

    def send_aux_to_targets(self, targets, topic, request, timeout, fl_ctx, optional, secure):
        msg_targets = self._to_aux_msg_targets(targets)
        if msg_targets:
            return self.run_manager.aux_runner.send_aux_request(
                targets=msg_targets,
                topic=topic,
                request=request,
                timeout=timeout,
                fl_ctx=fl_ctx,
                optional=optional,
                secure=secure,
            )
        else:
            return {}

    def stream_objects(
        self,
        channel: str,
        topic: str,
        stream_ctx: StreamContext,
        targets: List[str],
        producer: ObjectProducer,
        fl_ctx: FLContext,
        optional=False,
        secure=False,
    ):
        if not self.run_manager:
            raise RuntimeError("run_manager has not been created")

        if not self.run_manager.object_streamer:
            raise RuntimeError("object_streamer has not been created")

        return self.run_manager.object_streamer.stream(
            channel=channel,
            topic=topic,
            stream_ctx=stream_ctx,
            targets=self._to_aux_msg_targets(targets),
            producer=producer,
            fl_ctx=fl_ctx,
            secure=secure,
            optional=optional,
        )

    def register_stream_processing(
        self,
        channel: str,
        topic: str,
        factory: ConsumerFactory,
        stream_done_cb=None,
        consumed_cb=None,
        **cb_kwargs,
    ):
        if not self.run_manager:
            raise RuntimeError("run_manager has not been created")

        if not self.run_manager.object_streamer:
            raise RuntimeError("object_streamer has not been created")

        self.run_manager.object_streamer.register_stream_processing(
            channel=channel,
            topic=topic,
            factory=factory,
            stream_done_cb=stream_done_cb,
            consumed_cb=consumed_cb,
            **cb_kwargs,
        )

    def shutdown_streamer(self):
        if self.run_manager and self.run_manager.object_streamer:
            self.run_manager.object_streamer.shutdown()

    def sync_clients_from_main_process(self):
        # repeatedly ask the parent process to get participating clients until we receive the result
        # or timed out after 30 secs (already tried 30 times).
        start = time.time()
        max_wait = 30.0
        job_id = self.args.job_id
        while True:
            clients = self._retrieve_clients_data(job_id)
            if clients:
                self.client_manager.set_clients(clients)
                self.logger.debug(f"received participating clients {clients}")
                return

            if time.time() - start >= max_wait:
                self.logger.critical(f"Cannot get participating clients for job {job_id} after {max_wait} seconds")
                raise RuntimeError(f"Exiting job process: Cannot get participating clients for job {job_id}")

            self.logger.debug("didn't receive clients info - retry in 1 second")
            time.sleep(1.0)

    def get_participating_clients(self):
        # called from server's job cell
        return self.client_manager.clients

    def _retrieve_clients_data(self, job_id):
        request = new_cell_message({CellMessageHeaderKeys.JOB_ID: job_id}, {})
        return_data = self.server.cell.send_request(
            target=FQCN.ROOT_SERVER,
            channel=CellChannel.SERVER_PARENT_LISTENER,
            topic=ServerCommandNames.GET_CLIENTS,
            request=request,
            timeout=5.0,
            optional=True,
        )
        rc = return_data.get_header(MessageHeaderKey.RETURN_CODE, CellMsgReturnCode.OK)
        if rc != CellMsgReturnCode.OK:
            self.logger.debug(f"cannot retrieve clients from parent: {rc}")
            return None

        data = return_data.payload
        clients = data.get(ServerCommandKey.CLIENTS, None)
        if clients is None:
            self.logger.error(f"parent failed to return clients info for job {job_id}")
        return clients

    def update_job_run_status(self):
        with self.new_context() as fl_ctx:
            execution_error = fl_ctx.get_prop(FLContextKey.FATAL_SYSTEM_ERROR, False)
            data = {"execution_error": execution_error}
            job_id = fl_ctx.get_job_id()
            request = new_cell_message({CellMessageHeaderKeys.JOB_ID: job_id}, data)
            self.server.cell.fire_and_forget(
                targets=FQCN.ROOT_SERVER,
                channel=CellChannel.SERVER_PARENT_LISTENER,
                topic=ServerCommandNames.UPDATE_RUN_STATUS,
                message=request,
            )

    def notify_dead_job(self, job_id: str, client_name: str, reason: str):
        shareable = Shareable()
        shareable.set_header(ServerCommandKey.FL_CLIENT, client_name)
        shareable.set_header(ServerCommandKey.REASON, reason)
        self.send_command_to_child_runner_process(
            job_id=job_id,
            command_name=ServerCommandNames.HANDLE_DEAD_JOB,
            command_data=shareable,
            timeout=0.0,
            optional=True,
        )
        self.logger.warning(f"notified SJ of dead-job: {job_id=}; {client_name=}; {reason=}")

    def send_command_to_child_runner_process(
        self, job_id: str, command_name: str, command_data, timeout=5.0, optional=False
    ):
        with self.lock:
            fqcn = FQCN.join([FQCN.ROOT_SERVER, job_id])
            request = new_cell_message({}, command_data)
            if timeout <= 0.0:
                self.server.cell.fire_and_forget(
                    targets=fqcn,
                    channel=CellChannel.SERVER_COMMAND,
                    topic=command_name,
                    message=request,
                    optional=optional,
                )
                return None

            return_data = self.server.cell.send_request(
                target=fqcn,
                channel=CellChannel.SERVER_COMMAND,
                topic=command_name,
                request=request,
                timeout=timeout,
                optional=optional,
            )
            rc = return_data.get_header(MessageHeaderKey.RETURN_CODE, CellMsgReturnCode.OK)
            if rc == CellMsgReturnCode.OK:
                result = return_data.payload
            else:
                result = None
        return result

    def persist_components(self, fl_ctx: FLContext, completed: bool):
        if not self.server.ha_mode:
            return

        self.logger.info("Start saving snapshot on server.")

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
                job_clients = self.get_participating_clients()
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
        except Exception as e:
            self.logger.error(f"Failed to persist the components. {secure_format_exception(e)}")

    def restore_components(self, snapshot: RunSnapshot, fl_ctx: FLContext):
        for component_id, component in self.run_manager.components.items():
            if isinstance(component, FLComponent):
                component.restore(snapshot.get_component_snapshot(component_id=component_id), fl_ctx)

        fl_ctx.props.update(snapshot.get_component_snapshot(component_id=SnapshotKey.FL_CONTEXT))

    def dispatch(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        if self.run_manager and self.run_manager.aux_runner:
            return self.run_manager.aux_runner.dispatch(topic=topic, request=request, fl_ctx=fl_ctx)
        else:
            self.logger.warning("Server is not ready")
            return make_reply(ReturnCode.SERVER_NOT_READY)

    def show_stats(self, job_id) -> dict:
        stats = None
        try:
            stats = self.send_command_to_child_runner_process(
                job_id=job_id,
                command_name=ServerCommandNames.SHOW_STATS,
                command_data={},
            )
        except Exception as ex:
            self.logger.error(f"Failed to show_stats for JOB: {job_id}: {secure_format_exception(ex)}")

        if stats is None:
            stats = {}
        return stats

    def get_errors(self, job_id) -> dict:
        errors = None
        try:
            errors = self.send_command_to_child_runner_process(
                job_id=job_id,
                command_name=ServerCommandNames.GET_ERRORS,
                command_data={},
            )
        except Exception as ex:
            self.logger.error(f"Failed to get_errors for JOB: {job_id}: {secure_format_exception(ex)}")

        if errors is None:
            errors = {}
        return errors

    def reset_errors(self, job_id) -> str:
        errors = None
        try:
            self.send_command_to_child_runner_process(
                job_id=job_id,
                command_name=ServerCommandNames.RESET_ERRORS,
                command_data={},
            )
        except Exception as ex:
            self.logger.error(f"Failed to reset_errors for JOB: {job_id}: {secure_format_exception(ex)}")

        return f"reset the server error stats for job: {job_id}"

    def configure_job_log(self, job_id, data) -> str:
        error = None
        try:
            error = self.send_command_to_child_runner_process(
                job_id=job_id,
                command_name=AdminCommandNames.CONFIGURE_JOB_LOG,
                command_data=data,
            )
        except Exception as ex:
            err = f"Failed to configure_job_log for JOB: {job_id}: {secure_format_exception(ex)}"
            self.logger.error(err)
            return err

        return error

    def _send_admin_requests(self, requests, fl_ctx: FLContext, timeout_secs=10) -> List[ClientReply]:
        return self.server.admin_server.send_requests(requests, fl_ctx, timeout_secs=timeout_secs)

    def check_client_resources(self, job: Job, resource_reqs, fl_ctx: FLContext) -> Dict[str, Tuple[bool, str]]:
        requests = {}
        for site_name, resource_requirements in resource_reqs.items():
            # assume server resource is unlimited
            if site_name == SiteType.SERVER:
                continue
            request = self._make_message_for_check_resource(job, resource_requirements, fl_ctx)

            client = self.get_client_from_name(site_name)
            if client:
                requests.update({client.token: request})
        replies = []
        if requests:
            replies = self._send_admin_requests(requests, fl_ctx, 15)
        result = {}
        for r in replies:
            site_name = r.client_name
            if r.reply:
                error_code = r.reply.get_header(MsgHeader.RETURN_CODE, ReturnCode.OK)
                message = r.reply.body
                if error_code != ReturnCode.OK:
                    self.logger.error(f"Client reply error: {message}")
                    result[site_name] = (False, message)
                else:
                    resp = message
                    result[site_name] = (
                        resp.get_header(ShareableHeader.IS_RESOURCE_ENOUGH, False),
                        resp.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, ""),
                    )
            else:
                result[site_name] = (False, "")
        return result

    def _make_message_for_check_resource(self, job, resource_requirements, fl_ctx):
        request = Message(topic=TrainingTopic.CHECK_RESOURCE, body=resource_requirements)
        request.set_header(RequestHeader.JOB_ID, job.job_id)
        request.set_header(RequestHeader.REQUIRE_AUTHZ, "false")
        request.set_header(RequestHeader.ADMIN_COMMAND, AdminCommandNames.CHECK_RESOURCES)

        set_message_security_data(request, job, fl_ctx)
        return request

    def cancel_client_resources(
        self, resource_check_results: Dict[str, Tuple[bool, str]], resource_reqs: Dict[str, dict], fl_ctx: FLContext
    ):
        requests = {}
        for site_name, result in resource_check_results.items():
            is_resource_enough, token = result
            if is_resource_enough and token:
                resource_requirements = resource_reqs.get(site_name, {})
                request = Message(topic=TrainingTopic.CANCEL_RESOURCE, body=resource_requirements)
                request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, token)
                client = self.get_client_from_name(site_name)
                if client:
                    requests.update({client.token: request})
        if requests:
            _ = self._send_admin_requests(requests, fl_ctx)

    def start_client_job(self, job, client_sites, fl_ctx: FLContext):
        requests = {}
        for site, dispatch_info in client_sites.items():
            resource_requirement = dispatch_info.resource_requirements
            token = dispatch_info.token
            request = Message(topic=TrainingTopic.START_JOB, body=resource_requirement)
            request.set_header(RequestHeader.JOB_ID, job.job_id)
            request.set_header(RequestHeader.JOB_META, job.meta)
            request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, token)
            client = self.get_client_from_name(site)
            if client:
                requests.update({client.token: request})
        replies = []
        if requests:
            replies = self._send_admin_requests(requests, fl_ctx, timeout_secs=20)
        return replies

    def register_app_command(self, topic: str, cmd_func, *args, **kwargs):
        self.logger.debug(f"registering app command {topic}")
        ServerCommands.register_app_command(topic, cmd_func, *args, *kwargs)

    def stop_all_jobs(self):
        fl_ctx = self.new_context()
        self.job_runner.stop_all_runs(fl_ctx)

    def pause_server_jobs(self):
        running_jobs = list(self.run_processes.keys())
        for job_id in running_jobs:
            self.job_runner.remove_running_job(job_id)
            self.abort_app_on_server(job_id, turn_to_cold=True)

    def close(self):
        self.executor.shutdown()
        self.shutdown_streamer()


def server_shutdown(server, touch_file):
    with open(touch_file, "a"):
        os.utime(touch_file, None)

    try:
        # server.admin_server.stop()
        server.fl_shutdown()
        # time.sleep(3.0)
    finally:
        security_close()
        server.status = ServerStatus.SHUTDOWN
        # sys.exit(2)
