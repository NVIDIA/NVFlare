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
import shutil
import threading
import time
from abc import ABC, abstractmethod
from concurrent import futures
from threading import Lock
from typing import List, Optional

from .root.root_commander import RootCommander

from nvflare.private.defs import CellChannel, SessionTopic, TaskTopic, MessagePayloadKey
from nvflare.private.fed.cmi import CellMessageInterface
import nvflare.private.fed.protos.admin_pb2_grpc as admin_service
import nvflare.private.fed.protos.federated_pb2 as fed_msg
import nvflare.private.fed.protos.federated_pb2_grpc as fed_service
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import (
    FLContextKey,
    MachineStatus,
    ReservedKey,
    RunProcessKey,
    ServerCommandKey,
    ServerCommandNames,
    SnapshotKey,
    WorkspaceConstants,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, ReturnCode, Shareable, make_reply
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.cellnet.cell import (
    Cell, Message, MessageHeaderKey, new_message, ServiceUnavailable,
    AbortRun, InvalidSession, InvalidRequest
)
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.zip_utils import unzip_all_from_bytes
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.server.job.server_runner import ServerRunner
from nvflare.private.fed.utils.fed_utils import shareable_to_modeldata
from nvflare.private.fed.utils.messageproto import message_to_proto, proto_to_message
from nvflare.private.fed.utils.numproto import proto_to_bytes
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.fed_event import ServerFedEventRunner

from .client_manager import ClientManager
from .run_manager import RunManager
from .server_engine import ServerEngine
from nvflare.private.fed.server.server_state import (
    ABORT_RUN,
    ACTION,
    MESSAGE,
    NIS,
    Cold2HotState,
    ColdState,
    Hot2ColdState,
    HotState,
    ServerState,
)
from .server_status import ServerStatus

GRPC_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
]


class BaseServer(ABC):
    def __init__(
        self,
        project_name=None,
        min_num_clients=2,
        max_num_clients=10,
        heart_beat_timeout=600,
        handlers: Optional[List[FLComponent]] = None,
    ):
        """Base server that provides the clients management and server deployment."""
        self.project_name = project_name
        self.min_num_clients = max(min_num_clients, 1)
        self.max_num_clients = max(max_num_clients, 1)

        self.heart_beat_timeout = heart_beat_timeout
        self.handlers = handlers
        # self.cmd_modules = cmd_modules

        self.client_manager = ClientManager(
            project_name=self.project_name, min_num_clients=self.min_num_clients, max_num_clients=self.max_num_clients
        )

        self.grpc_server = None
        self.admin_server = None
        self.lock = Lock()
        self.snapshot_lock = Lock()
        self.fl_ctx = FLContext()
        self.platform = None

        self.shutdown = False
        self.status = ServerStatus.NOT_STARTED

        self.abort_signal = None
        self.executor = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def get_all_clients(self):
        return self.client_manager.get_clients()

    @abstractmethod
    def remove_client_data(self, token):
        pass

    def close(self):
        """Shutdown the server."""
        try:
            if self.lock:
                self.lock.release()
        except RuntimeError:
            self.logger.info("canceling sync locks")
        try:
            if self.grpc_server:
                self.grpc_server.stop(0)
        finally:
            self.logger.info("server off")
            return 0

    def deploy(self, args, grpc_args=None, secure_train=False):
        """Start a grpc server and listening the designated port."""
        num_server_workers = grpc_args.get("num_server_workers", 1)
        num_server_workers = max(self.client_manager.get_min_clients(), num_server_workers)
        target = grpc_args["service"].get("target", "0.0.0.0:6007")
        grpc_options = grpc_args["service"].get("options", GRPC_DEFAULT_OPTIONS)

        compression = grpc.Compression.NoCompression
        if "Deflate" == grpc_args.get("compression"):
            compression = grpc.Compression.Deflate
        elif "Gzip" == grpc_args.get("compression"):
            compression = grpc.Compression.Gzip

        if not self.grpc_server:
            self.executor = futures.ThreadPoolExecutor(max_workers=num_server_workers)
            self.grpc_server = grpc.server(
                self.executor,
                options=grpc_options,
                compression=compression,
            )
            fed_service.add_FederatedTrainingServicer_to_server(self, self.grpc_server)
            admin_service.add_AdminCommunicatingServicer_to_server(self, self.grpc_server)

        if secure_train:
            with open(grpc_args["ssl_private_key"], "rb") as f:
                private_key = f.read()
            with open(grpc_args["ssl_cert"], "rb") as f:
                certificate_chain = f.read()
            with open(grpc_args["ssl_root_cert"], "rb") as f:
                root_ca = f.read()

            server_credentials = grpc.ssl_server_credentials(
                (
                    (
                        private_key,
                        certificate_chain,
                    ),
                ),
                root_certificates=root_ca,
                require_client_auth=True,
            )
            port = target.split(":")[1]
            tmp_target = f"0.0.0.0:{port}"
            self.grpc_server.add_secure_port(tmp_target, server_credentials)
            self.logger.info("starting secure server at %s", target)
        else:
            self.grpc_server.add_insecure_port(target)
            self.logger.info("starting insecure server at %s", target)
        self.grpc_server.start()

        # return self.start()
        cleanup_thread = threading.Thread(target=self.client_cleanup)
        # heartbeat_thread.daemon = True
        cleanup_thread.start()

    def client_cleanup(self):
        while not self.shutdown:
            self.remove_dead_clients()
            time.sleep(15)

    def set_admin_server(self, admin_server):
        self.admin_server = admin_server

    def remove_dead_clients(self):
        # Clean and remove the dead client without heartbeat.
        self.logger.debug("trying to remove dead clients .......")
        delete = []
        for token, client in self.client_manager.get_clients().items():
            if client.last_connect_time < time.time() - self.heart_beat_timeout:
                delete.append(token)
        for token in delete:
            client = self.logout_client(token)
            self.logger.info(
                "Remove the dead Client. Name: {}\t Token: {}.  Total clients: {}".format(
                    client.name, token, len(self.client_manager.get_clients())
                )
            )

    def logout_client(self, token):
        client = self.client_manager.remove_client(token)
        self.remove_client_data(token)
        if self.admin_server:
            self.admin_server.client_dead(token)
        self.notify_dead_client(client)
        return client

    def notify_dead_client(self, client):
        """Called to do further processing of the dead client

        Args:
            client: the dead client

        Returns:

        """
        pass

    def fl_shutdown(self):
        self.shutdown = True
        self.close()
        if self.executor:
            self.executor.shutdown()


class FederatedServer(BaseServer, fed_service.FederatedTrainingServicer, admin_service.AdminCommunicatingServicer):
    def __init__(
        self,
        cell: Cell,
        project_name=None,
        min_num_clients=2,
        max_num_clients=10,
        cmd_modules=None,
        heart_beat_timeout=600,
        handlers: Optional[List[FLComponent]] = None,
        args=None,
        secure_train=False,
        snapshot_persistor=None,
        overseer_agent=None,
    ):
        """Federated server services.

        Args:
            project_name: server project name.
            min_num_clients: minimum number of contributors at each round.
            max_num_clients: maximum number of contributors at each round.
            cmd_modules: command modules.
            heart_beat_timeout: heartbeat timeout
            handlers: A list of handler
            args: arguments
            secure_train: whether to use secure communication
        """
        BaseServer.__init__(
            self,
            project_name=project_name,
            min_num_clients=min_num_clients,
            max_num_clients=max_num_clients,
            heart_beat_timeout=heart_beat_timeout,
            handlers=handlers,
        )

        self.cell = cell
        self.contributed_clients = {}
        self.tokens = None
        self.round_started = Timestamp()

        with self.lock:
            self.reset_tokens()

        self.cmd_modules = cmd_modules

        self.builder = None

        # Additional fields for CurrentTask meta_data in GetModel API.
        self.current_model_meta_data = {}

        self.engine = self._create_server_engine(args, snapshot_persistor)
        self.run_manager = None
        self.server_runner = None

        self.processors = {}
        self.runner_config = None
        self.secure_train = secure_train

        self.workspace = args.workspace
        self.snapshot_location = None
        self.overseer_agent = overseer_agent
        self.server_state: ServerState = ColdState()
        self.snapshot_persistor = snapshot_persistor

        cell.set_message_interceptor(self._inspect_message)
        cell.register_request_cb(
            channel=CellChannel.SESSION,
            topic=SessionTopic.REGISTER,
            cb=self._do_register
        )
        cell.register_request_cb(
            channel=CellChannel.SESSION,
            topic=SessionTopic.HEARTBEAT,
            cb=self._do_heartbeat
        )
        cell.register_request_cb(
            channel=CellChannel.SESSION,
            topic=SessionTopic.LOGOUT,
            cb=self._do_quit
        )

    def _create_server_engine(self, args, snapshot_persistor):
        return ServerEngine(
            server=self, args=args, client_manager=self.client_manager, snapshot_persistor=snapshot_persistor
        )

    def get_current_model_meta_data(self):
        """Get the model metadata, which usually contains additional fields."""
        s = Struct()
        for k, v in self.current_model_meta_data.items():
            s.update({k: v})
        return s

    def remove_client_data(self, token):
        self.tokens.pop(token, None)

    def reset_tokens(self):
        """Reset the token set.

        After resetting, each client can take a token
        and start fetching the current global model.
        This function is not thread-safe.
        """
        self.tokens = dict()
        for client in self.get_all_clients().keys():
            self.tokens[client] = self.task_meta_info

    def _inspect_message(self, message: Message):
        channel = message.get_header(MessageHeaderKey.CHANNEL)
        topic = message.get_header(MessageHeaderKey.TOPIC)
        with self.engine.new_context() as fl_ctx:
            if channel == CellChannel.SESSION:
                if topic == SessionTopic.REGISTER:
                    state_check = self.server_state.register(fl_ctx)
                elif topic == SessionTopic.HEARTBEAT:
                    state_check = self.server_state.heartbeat(fl_ctx)
                else:
                    return None
            else:
                self._ssid_check(message)
                if channel == CellChannel.TASK:
                    if topic == TaskTopic.GET_TASK:
                        state_check = self.server_state.get_task(fl_ctx)
                    elif topic == TaskTopic.SUBMIT_RESULT:
                        state_check = self.server_state.submit_result(fl_ctx)
                    else:
                        return None
                elif channel == CellChannel.AUX:
                    state_check = self.server_state.aux_communicate(fl_ctx)
                else:
                    return None
            return self._handle_state_check(state_check)

    def _do_register(self, request: Message):
        """Register new clients on the fly.

        Each client must get registered before getting the global model.
        The server will expect updates from the registered clients
        for multiple federated rounds.

        This function does not change min_num_clients and max_num_clients.
        """

        token = self.client_manager.authenticate(request)
        self.tokens[token] = self.task_meta_info
        if self.admin_server:
            self.admin_server.client_heartbeat(token)

        result = Shareable({
                CellMessageInterface.HEADER_CLIENT_TOKEN: token,
                CellMessageInterface.HEADER_SSID: self.server_state.ssid
            })
        return new_message(payload=result)

    @staticmethod
    def _handle_state_check(state_check):
        action = state_check.get(ACTION)
        msg = state_check.get(MESSAGE)
        if action == NIS:
            raise ServiceUnavailable(msg)
        if action == ABORT_RUN:
            raise AbortRun(msg)

    def _ssid_check(self, request: Message):
        ssid = request.get_header(CellMessageInterface.HEADER_SSID)
        if ssid != self.server_state.ssid:
            raise InvalidSession("Invalid Service session ID")

    def _do_quit(self, request: Message):
        """Existing client quits the federated training process.
        This function does not change min_num_clients and max_num_clients.
        """
        client = self.client_manager.validate_client(request)
        if client:
            token = client.get_token()
            self.logout_client(token)
        return new_message()

    def _do_heartbeat(self, request: Message):
        token = request.get_header(CellMessageInterface.HEADER_CLIENT_TOKEN)
        client_name = request.get_prop(CellMessageInterface.HEADER_CLIENT_NAME)

        if self.client_manager.heartbeat(token, client_name):
            self.tokens[token] = self.task_meta_info
        if self.admin_server:
            self.admin_server.client_heartbeat(token)

        abort_runs = self._sync_client_jobs(request, token)
        summary_info = Shareable()
        if abort_runs:
            summary_info[MessagePayloadKey.ABORT_JOBS] = abort_runs
            display_runs = ",".join(abort_runs)
            self.logger.info(
                f"These jobs: {display_runs} are not running on the server. "
                f"Ask client: {client_name} to abort these runs."
            )
        return new_message(payload=summary_info)

    def _sync_client_jobs(self, request: Message, client_token):
        # jobs that are running on client but not on server need to be aborted!
        payload = request.payload
        if not isinstance(payload, Shareable):
            raise InvalidRequest(f"payload must be Shareable but got {type(payload)}")

        client_jobs = payload.get(MessagePayloadKey.JOBS)
        server_jobs = self.engine.run_processes.keys()
        jobs_need_abort = list(set(client_jobs).difference(server_jobs))

        # also check jobs that are running on server but not on the client
        jobs_on_server_but_not_on_client = list(set(server_jobs).difference(client_jobs))
        if jobs_on_server_but_not_on_client:
            # should this job be running on the client?
            for job_id in jobs_on_server_but_not_on_client:
                job_info = self.engine.run_processes[job_id]
                participating_clients = job_info.get(RunProcessKey.PARTICIPANTS, None)
                if participating_clients:
                    # this is a dict: token => nvflare.apis.client.Client
                    client = participating_clients.get(client_token, None)
                    if client:
                        self._notify_dead_job(client, job_id)

        return jobs_need_abort

    def _notify_dead_job(self, client, job_id: str):
        commander = self.engine.get_commander()
        assert isinstance(commander, RootCommander)
        commander.notify_dead_job(client.name, job_id)

    def notify_dead_client(self, client):
        """Called to do further processing of the dead client

        Args:
            client: the dead client

        Returns:

        """
        # find all RUNs that this client is participating
        if not self.engine.run_processes:
            return

        for job_id, process_info in self.engine.run_processes.items():
            assert isinstance(process_info, dict)
            participating_clients = process_info.get(RunProcessKey.PARTICIPANTS, None)
            if participating_clients and client.token in participating_clients:
                self._notify_dead_job(client, job_id)

    def start_run(self, job_id, run_root, conf, args, snapshot):
        # Create the FL Engine
        workspace = Workspace(args.workspace, "server", args.config_folder)
        self.run_manager = self.create_run_manager(workspace, job_id)
        self.engine.set_run_manager(self.run_manager)
        self.engine.set_configurator(conf)
        self.engine.asked_to_stop = False

        fed_event_runner = ServerFedEventRunner()
        self.run_manager.add_handler(fed_event_runner)

        try:
            self.server_runner = ServerRunner(config=self.runner_config, job_id=job_id, engine=self.engine)
            self.run_manager.add_handler(self.server_runner)
            self.run_manager.add_component("_Server_Runner", self.server_runner)

            with self.engine.new_context() as fl_ctx:

                if snapshot:
                    self.engine.restore_components(snapshot=snapshot, fl_ctx=FLContext())

                fl_ctx.set_prop(FLContextKey.APP_ROOT, run_root, sticky=True)
                fl_ctx.set_prop(FLContextKey.CURRENT_RUN, job_id, private=False, sticky=True)
                fl_ctx.set_prop(FLContextKey.WORKSPACE_ROOT, args.workspace, private=True, sticky=True)
                fl_ctx.set_prop(FLContextKey.ARGS, args, private=True, sticky=True)
                fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True)
                fl_ctx.set_prop(FLContextKey.SECURE_MODE, self.secure_train, private=True, sticky=True)
                fl_ctx.set_prop(FLContextKey.RUNNER, self.server_runner, private=True, sticky=True)

            engine_thread = threading.Thread(target=self.run_engine)
            engine_thread.start()

            self.engine.engine_info.status = MachineStatus.STARTED
            while self.engine.engine_info.status != MachineStatus.STOPPED:
                if self.engine.asked_to_stop:
                    self.engine.engine_info.status = MachineStatus.STOPPED

                time.sleep(3)

            if engine_thread.is_alive():
                engine_thread.join()

        finally:
            self.engine.engine_info.status = MachineStatus.STOPPED
            self.run_manager = None

    def create_run_manager(self, workspace, job_id):
        return RunManager(
            server_name=self.project_name,
            engine=self.engine,
            job_id=job_id,
            workspace=workspace,
            components=self.runner_config.components,
            client_manager=self.client_manager,
            handlers=self.runner_config.handlers,
        )

    def abort_run(self):
        with self.engine.new_context() as fl_ctx:
            if self.server_runner:
                self.server_runner.abort(fl_ctx)

    def run_engine(self):
        self.engine.engine_info.status = MachineStatus.STARTED
        self.server_runner.run()
        self.engine.engine_info.status = MachineStatus.STOPPED

    def deploy(self, args, grpc_args=None, secure_train=False):
        super().deploy(args, grpc_args, secure_train)

        target = grpc_args["service"].get("target", "0.0.0.0:6007")
        self.server_state.host = target.split(":")[0]
        self.server_state.service_port = target.split(":")[1]

        self.overseer_agent = self._init_agent(args)

        if secure_train:
            if self.overseer_agent:
                self.overseer_agent.set_secure_context(
                    ca_path=grpc_args["ssl_root_cert"],
                    cert_path=grpc_args["ssl_cert"],
                    prv_key_path=grpc_args["ssl_private_key"],
                )

        self.overseer_agent.start(self.overseer_callback)

    def _init_agent(self, args=None):
        kv_list = parse_vars(args.set)
        sp = kv_list.get("sp")

        if sp:
            with self.engine.new_context() as fl_ctx:
                fl_ctx.set_prop(FLContextKey.SP_END_POINT, sp)
                self.overseer_agent.initialize(fl_ctx)

        return self.overseer_agent

    def overseer_callback(self, overseer_agent):
        if overseer_agent.is_shutdown():
            self.engine.shutdown_server()
            return

        sp = overseer_agent.get_primary_sp()
        # print(sp)
        with self.engine.new_context() as fl_ctx:
            self.server_state = self.server_state.handle_sd_callback(sp, fl_ctx)

        if isinstance(self.server_state, Cold2HotState):
            server_thread = threading.Thread(target=self._turn_to_hot)
            server_thread.start()

        if isinstance(self.server_state, Hot2ColdState):
            server_thread = threading.Thread(target=self._turn_to_cold)
            server_thread.start()

    def _turn_to_hot(self):
        # Restore Snapshot
        with self.snapshot_lock:
            fl_snapshot = self.snapshot_persistor.retrieve()
            if fl_snapshot:
                for run_number, snapshot in fl_snapshot.run_snapshots.items():
                    if snapshot and not snapshot.completed:
                        # Restore the workspace
                        workspace_data = snapshot.get_component_snapshot(SnapshotKey.WORKSPACE).get("content")
                        dst = os.path.join(self.workspace, WorkspaceConstants.WORKSPACE_PREFIX + str(run_number))
                        if os.path.exists(dst):
                            shutil.rmtree(dst, ignore_errors=True)

                        os.makedirs(dst, exist_ok=True)
                        unzip_all_from_bytes(workspace_data, dst)

                        job_id = snapshot.get_component_snapshot(SnapshotKey.JOB_INFO).get(SnapshotKey.JOB_ID)
                        job_clients = snapshot.get_component_snapshot(SnapshotKey.JOB_INFO).get(SnapshotKey.JOB_CLIENTS)
                        self.logger.info(f"Restore the previous snapshot. Run_number: {run_number}")
                        with self.engine.new_context() as fl_ctx:
                            job_runner = self.engine.job_runner
                            job_runner.restore_running_job(
                                run_number=run_number,
                                job_id=job_id,
                                job_clients=job_clients,
                                snapshot=snapshot,
                                fl_ctx=fl_ctx,
                            )

            self.server_state = HotState(
                host=self.server_state.host, port=self.server_state.service_port, ssid=self.server_state.ssid
            )

    def _turn_to_cold(self):
        # Wrap-up server operations
        self.server_state = ColdState(host=self.server_state.host, port=self.server_state.service_port)

    def stop_training(self):
        self.status = ServerStatus.STOPPED
        self.logger.info("Server app stopped.\n\n")

    def fl_shutdown(self):
        self.engine.stop_all_jobs()
        self.engine.fire_event(EventType.SYSTEM_END, self.engine.new_context())

        super().fl_shutdown()

    def close(self):
        """Shutdown the server."""
        self.logger.info("shutting down server")
        self.shutdown = True
        if self.overseer_agent:
            self.overseer_agent.end()
        return super().close()
