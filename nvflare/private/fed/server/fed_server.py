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

import grpc
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp

import nvflare.private.fed.protos.admin_pb2 as admin_msg
import nvflare.private.fed.protos.admin_pb2_grpc as admin_service
import nvflare.private.fed.protos.federated_pb2 as fed_msg
import nvflare.private.fed.protos.federated_pb2_grpc as fed_service
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import (
    FLContextKey,
    MachineStatus,
    ReservedKey,
    ServerCommandKey,
    ServerCommandNames,
    SnapshotKey,
    WorkspaceConstants,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, ReturnCode, Shareable, make_reply
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.apis.workspace import Workspace
from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.server.server_runner import ServerRunner
from nvflare.private.fed.utils.fed_utils import shareable_to_modeldata
from nvflare.private.fed.utils.messageproto import message_to_proto, proto_to_message
from nvflare.private.fed.utils.numproto import proto_to_bytes
from nvflare.widgets.fed_event import ServerFedEventRunner

from .client_manager import ClientManager
from .run_manager import RunManager
from .server_engine import ServerEngine
from .server_state import (
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
            if self.admin_server:
                self.admin_server.stop()
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
            client = self.client_manager.remove_client(token)
            self.remove_client_data(token)
            if self.admin_server:
                self.admin_server.client_dead(token)
            self.logger.info(
                "Remove the dead Client. Name: {}\t Token: {}.  Total clients: {}".format(
                    client.name, token, len(self.client_manager.get_clients())
                )
            )

    def fl_shutdown(self):
        self.shutdown = True
        self.close()
        if self.executor:
            self.executor.shutdown()


class FederatedServer(BaseServer, fed_service.FederatedTrainingServicer, admin_service.AdminCommunicatingServicer):
    def __init__(
        self,
        project_name=None,
        min_num_clients=2,
        max_num_clients=10,
        cmd_modules=None,
        heart_beat_timeout=600,
        handlers: Optional[List[FLComponent]] = None,
        args=None,
        secure_train=False,
        enable_byoc=False,
        snapshot_persistor=None,
        overseer_agent=None,
        collective_command_timeout=600.0,
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
            enable_byoc: whether to enable custom components
            collective_command_timeout: timeout for waiting all collective requests from clients
        """
        self.logger = logging.getLogger("FederatedServer")

        BaseServer.__init__(
            self,
            project_name=project_name,
            min_num_clients=min_num_clients,
            max_num_clients=max_num_clients,
            heart_beat_timeout=heart_beat_timeout,
            handlers=handlers,
        )

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
        self.enable_byoc = enable_byoc

        self.workspace = args.workspace
        self.snapshot_location = None
        self.overseer_agent = overseer_agent
        self.server_state: ServerState = ColdState()
        self.snapshot_persistor = snapshot_persistor

        flare_decomposers.register()
        common_decomposers.register()
        self._collective_comm_timeout = collective_command_timeout

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

    @property
    def task_meta_info(self):
        """Task meta information.

        The model_meta_info uniquely defines the current model,
        it is used to reject outdated client's update.
        """
        meta_info = fed_msg.MetaData()
        meta_info.created.CopyFrom(self.round_started)
        meta_info.project.name = self.project_name
        return meta_info

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

    def Register(self, request, context):
        """Register new clients on the fly.

        Each client must get registered before getting the global model.
        The server will expect updates from the registered clients
        for multiple federated rounds.

        This function does not change min_num_clients and max_num_clients.
        """

        with self.engine.new_context() as fl_ctx:
            state_check = self.server_state.register(fl_ctx)
            self._handle_state_check(context, state_check)

            token = self.client_manager.authenticate(request, context)
            if token:
                self.tokens[token] = self.task_meta_info
                if self.admin_server:
                    self.admin_server.client_heartbeat(token)

                return fed_msg.FederatedSummary(
                    comment="New client registered", token=token, ssid=self.server_state.ssid
                )

    def _handle_state_check(self, context, state_check):
        if state_check.get(ACTION) == NIS:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                state_check.get(MESSAGE),
            )
        elif state_check.get(ACTION) == ABORT_RUN:
            context.abort(
                grpc.StatusCode.ABORTED,
                state_check.get(MESSAGE),
            )

    def _ssid_check(self, client_state, context):
        if client_state.ssid != self.server_state.ssid:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid Service session ID")

    def Quit(self, request, context):
        """Existing client quits the federated training process.

        Server will stop sharing the global model with the client,
        further contribution will be rejected.

        This function does not change min_num_clients and max_num_clients.
        """
        # fire_event(EventType.CLIENT_QUIT, self.handlers, self.fl_ctx)

        client = self.client_manager.validate_client(request, context)
        if client:
            token = client.get_token()

            _ = self.client_manager.remove_client(token)
            self.tokens.pop(token, None)
            if self.admin_server:
                self.admin_server.client_dead(token)

        return fed_msg.FederatedSummary(comment="Removed client")

    def GetTask(self, request, context):
        """Process client's get task request."""

        with self.engine.new_context() as fl_ctx:
            state_check = self.server_state.get_task(fl_ctx)
            self._handle_state_check(context, state_check)
            self._ssid_check(request, context)

            client = self.client_manager.validate_client(request, context)
            if client is None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Client not valid.")

            self.logger.debug(f"Fetch task requested from client: {client.name} ({client.get_token()})")
            token = client.get_token()

            # engine = fl_ctx.get_engine()
            shared_fl_ctx = fobs.loads(proto_to_bytes(request.context["fl_context"]))
            job_id = str(shared_fl_ctx.get_prop(FLContextKey.CURRENT_RUN))
            # fl_ctx.set_peer_context(shared_fl_ctx)

            with self.lock:
                # if self.server_runner is None or engine is None or self.engine.run_manager is None:
                if job_id not in self.engine.run_processes.keys():
                    self.logger.info("server has no current run - asked client to end the run")
                    task_name = SpecialTaskName.END_RUN
                    task_id = ""
                    shareable = None
                else:
                    shareable, task_id, task_name = self._process_task_request(client, fl_ctx, shared_fl_ctx)

                    # task_name, task_id, shareable = self.server_runner.process_task_request(client, fl_ctx)

                if shareable is None:
                    shareable = Shareable()

                task = fed_msg.CurrentTask(task_name=task_name)
                task.meta.CopyFrom(self.task_meta_info)
                meta_data = self.get_current_model_meta_data()

                # we need TASK_ID back as a cookie
                shareable.add_cookie(name=FLContextKey.TASK_ID, data=task_id)

                # we also need to make TASK_ID available to the client
                shareable.set_header(key=FLContextKey.TASK_ID, value=task_id)

                task.meta_data.CopyFrom(meta_data)

                current_model = shareable_to_modeldata(shareable, fl_ctx)
                task.data.CopyFrom(current_model)
                if task_name == SpecialTaskName.TRY_AGAIN:
                    self.logger.debug(f"GetTask: Return task: {task_name} to client: {client.name} ({token}) ")
                else:
                    self.logger.info(f"GetTask: Return task: {task_name} to client: {client.name} ({token}) ")

                return task

    def _process_task_request(self, client, fl_ctx, shared_fl_ctx):
        task_name = SpecialTaskName.END_RUN
        task_id = ""
        shareable = None
        try:
            with self.engine.lock:
                job_id = shared_fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
                command_conn = self.engine.get_command_conn(str(job_id))
                if command_conn:
                    command_shareable = Shareable()
                    command_shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)
                    command_shareable.set_header(ServerCommandKey.FL_CLIENT, client)

                    data = {
                        ServerCommandKey.COMMAND: ServerCommandNames.GET_TASK,
                        ServerCommandKey.DATA: command_shareable,
                    }
                    command_conn.send(data)

                    return_data = fobs.loads(command_conn.recv())
                    task_name = return_data.get(ServerCommandKey.TASK_NAME)
                    task_id = return_data.get(ServerCommandKey.TASK_ID)
                    shareable = return_data.get(ServerCommandKey.SHAREABLE)
                    child_fl_ctx = return_data.get(ServerCommandKey.FL_CONTEXT)

                    fl_ctx.props.update(child_fl_ctx)
        except BaseException as e:
            self.logger.error(f"Could not connect to server runner process: {e} - asked client to end the run")
        return shareable, task_id, task_name

    def SubmitUpdate(self, request, context):
        """Handle client's submission of the federated updates."""
        # if self.server_runner is None or self.engine.run_manager is None:

        with self.engine.new_context() as fl_ctx:
            state_check = self.server_state.submit_result(fl_ctx)
            self._handle_state_check(context, state_check)
            self._ssid_check(request.client, context)

            contribution = request

            client = self.client_manager.validate_client(contribution.client, context)
            if client is None:
                response_comment = "Ignored the submit from invalid client. "
                self.logger.info(response_comment)
            else:
                with self.lock:
                    shareable = Shareable.from_bytes(proto_to_bytes(request.data.params["data"]))
                    shared_fl_context = fobs.loads(proto_to_bytes(request.data.params["fl_context"]))

                    job_id = str(shared_fl_context.get_prop(FLContextKey.CURRENT_RUN))
                    if job_id not in self.engine.run_processes.keys():
                        self.logger.info("ignored result submission since Server Engine isn't ready")
                        context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server has stopped")

                    shared_fl_context.set_prop(FLContextKey.SHAREABLE, shareable, private=True)

                    contribution_meta = contribution.client.meta
                    client_contrib_id = "{}_{}_{}".format(
                        contribution_meta.project.name, client.name, contribution_meta.current_round
                    )
                    contribution_task_name = contribution.task_name

                    timenow = Timestamp()
                    timenow.GetCurrentTime()
                    time_seconds = timenow.seconds - self.round_started.seconds
                    self.logger.info(
                        "received update from %s (%s Bytes, %s seconds)",
                        client_contrib_id,
                        contribution.ByteSize(),
                        time_seconds or "less than 1",
                    )

                    task_id = shareable.get_cookie(FLContextKey.TASK_ID)
                    shareable.set_header(ServerCommandKey.FL_CLIENT, client)
                    shareable.set_header(ServerCommandKey.TASK_NAME, contribution_task_name)
                    data = {ReservedKey.SHAREABLE: shareable, ReservedKey.SHARED_FL_CONTEXT: shared_fl_context}

                    self._submit_update(data, shared_fl_context)

                    # self.server_runner.process_submission(client, contribution_task_name, task_id, shareable, fl_ctx)

            response_comment = "Received from {} ({} Bytes, {} seconds)".format(
                contribution.client.client_name,
                contribution.ByteSize(),
                time_seconds or "less than 1",
            )
            summary_info = fed_msg.FederatedSummary(comment=response_comment)
            summary_info.meta.CopyFrom(self.task_meta_info)

            return summary_info

    def _submit_update(self, submit_update_data, shared_fl_context):
        try:
            with self.engine.lock:
                job_id = shared_fl_context.get_prop(FLContextKey.CURRENT_RUN)
                command_conn = self.engine.get_command_conn(str(job_id))
                if command_conn:
                    data = {
                        ServerCommandKey.COMMAND: ServerCommandNames.SUBMIT_UPDATE,
                        ServerCommandKey.DATA: submit_update_data,
                    }
                    command_conn.send(data)
        except BaseException as e:
            self.logger.error(f"Could not connect to server runner process: {str(e)}", exc_info=True)

    def AuxCommunicate(self, request, context):
        """Handle auxiliary channel communication."""
        with self.engine.new_context() as fl_ctx:
            state_check = self.server_state.aux_communicate(fl_ctx)
            self._handle_state_check(context, state_check)
            self._ssid_check(request.client, context)

            self.logger.info("getting AuxCommunicate request")

            contribution = request

            client = self.client_manager.validate_client(contribution.client, context)
            if client is None:
                response_comment = "Ignored the submit from invalid client. "
                self.logger.info(response_comment)

            shareable = Shareable()
            shareable = shareable.from_bytes(proto_to_bytes(request.data["data"]))
            shared_fl_context = fobs.loads(proto_to_bytes(request.data["fl_context"]))

            job_id = str(shared_fl_context.get_prop(FLContextKey.CURRENT_RUN))
            if job_id not in self.engine.run_processes.keys():
                self.logger.info("ignored AuxCommunicate request since Server Engine isn't ready")
                reply = make_reply(ReturnCode.SERVER_NOT_READY)
                aux_reply = fed_msg.AuxReply()
                aux_reply.data.CopyFrom(shareable_to_modeldata(reply, fl_ctx))

                return aux_reply

            fl_ctx.set_peer_context(shared_fl_context)
            shareable.set_peer_props(shared_fl_context.get_all_public_props())

            shared_fl_context.set_prop(FLContextKey.SHAREABLE, shareable, private=True)

            topic = shareable.get_header(ReservedHeaderKey.TOPIC)

            reply = self._aux_communicate(fl_ctx, shareable, shared_fl_context, topic)

            # reply = self.engine.dispatch(topic=topic, request=shareable, fl_ctx=fl_ctx)

            aux_reply = fed_msg.AuxReply()
            aux_reply.data.CopyFrom(shareable_to_modeldata(reply, fl_ctx))

            return aux_reply

    def _aux_communicate(self, fl_ctx, shareable, shared_fl_context, topic):
        try:
            with self.engine.lock:
                job_id = shared_fl_context.get_prop(FLContextKey.CURRENT_RUN)
                command_conn = self.engine.get_command_conn(str(job_id))
                if command_conn:
                    command_shareable = Shareable()
                    command_shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_context)
                    command_shareable.set_header(ServerCommandKey.TOPIC, topic)
                    command_shareable.set_header(ServerCommandKey.SHAREABLE, shareable)

                    data = {
                        ServerCommandKey.COMMAND: ServerCommandNames.AUX_COMMUNICATE,
                        ServerCommandKey.DATA: command_shareable,
                    }
                    command_conn.send(data)

                    return_data = command_conn.recv()
                    reply = return_data.get(ServerCommandKey.AUX_REPLY)
                    child_fl_ctx = return_data.get(ServerCommandKey.FL_CONTEXT)

                    fl_ctx.props.update(child_fl_ctx)
                else:
                    reply = make_reply(ReturnCode.ERROR)
        except BaseException as e:
            self.logger.error(f"Could not connect to server runner process: {str(e)} - asked client to end the run")
            reply = make_reply(ReturnCode.COMMUNICATION_ERROR)

        return reply

    def Heartbeat(self, request, context):

        with self.engine.new_context() as fl_ctx:
            state_check = self.server_state.heartbeat(fl_ctx)
            self._handle_state_check(context, state_check)

            token = request.token
            cn_names = context.auth_context().get("x509_common_name")
            if cn_names:
                client_name = cn_names[0].decode("utf-8")
            else:
                client_name = request.client_name

            if self.client_manager.heartbeat(token, client_name, context):
                self.tokens[token] = self.task_meta_info
            if self.admin_server:
                self.admin_server.client_heartbeat(token)

            abort_runs = self._sync_client_jobs(request)
            summary_info = fed_msg.FederatedSummary()
            if abort_runs:
                del summary_info.abort_jobs[:]
                summary_info.abort_jobs.extend(abort_runs)
                display_runs = ",".join(abort_runs)
                self.logger.info(
                    f"These jobs: {display_runs} are not running on the server. "
                    f"Ask client: {client_name} to abort these runs."
                )
            return summary_info

    def _sync_client_jobs(self, request):
        client_jobs = request.jobs
        server_jobs = self.engine.run_processes.keys()
        jobs_need_abort = list(set(client_jobs).difference(server_jobs))
        return jobs_need_abort

    def Retrieve(self, request, context):
        client_name = request.client_name
        messages = self.admin_server.get_outgoing_requests(client_token=client_name) if self.admin_server else []

        response = admin_msg.Messages()
        for m in messages:
            response.message.append(message_to_proto(m))
        return response

    def SendReply(self, request, context):
        client_name = request.client_name
        message = proto_to_message(request.message)
        if self.admin_server:
            self.admin_server.accept_reply(client_token=client_name, reply=message)

        response = admin_msg.Empty()
        return response

    def SendResult(self, request, context):
        client_name = request.client_name
        message = proto_to_message(request.message)

        processor = self.processors.get(message.topic)
        processor.process(client_name, message)

        response = admin_msg.Empty()
        return response

    def start_run(self, job_id, run_root, conf, args, snapshot):
        # Create the FL Engine
        workspace = Workspace(args.workspace, "server", args.config_folder)
        self.run_manager = RunManager(
            server_name=self.project_name,
            engine=self.engine,
            job_id=job_id,
            workspace=workspace,
            components=self.runner_config.components,
            client_manager=self.client_manager,
            handlers=self.runner_config.handlers,
        )
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
            self.engine.run_manager = None
            self.run_manager = None

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
        if self.engine:
            self.engine.close()
        if self.overseer_agent:
            self.overseer_agent.end()
        return super().close()
