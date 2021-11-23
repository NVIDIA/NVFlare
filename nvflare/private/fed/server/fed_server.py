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

"""federated server for aggregating and sharing federated model"""

import logging
import pickle
import threading
import time
from abc import abstractmethod
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
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import MachineStatus, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, ReservedHeaderKey
from nvflare.apis.workspace import Workspace
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.server.server_runner import ServerRunner
from nvflare.private.fed.utils.messageproto import message_to_proto, proto_to_message
from nvflare.private.fed.utils.numproto import proto_to_bytes
from nvflare.widgets.fed_event import ServerFedEventRunner
from .client_manager import ClientManager
from .run_manager import RunManager
from .server_engine import ServerEngine
from .server_status import ServerStatus
from ..utils.fed_utils import shareable_to_modeldata

GRPC_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
]


class BaseServer:
    """
    Base FL server, provides the clients management, server deployment.
    """
    def __init__(
        self,
        project_name=None,
        min_num_clients=2,
        max_num_clients=10,
        heart_beat_timeout=600,
        handlers: Optional[List[FLComponent]] = None,
    ):
        self.project_name = project_name  # should uniquely define a tlt workflow
        self.min_num_clients = max(min_num_clients, 1)
        self.max_num_clients = max(max_num_clients, 1)

        self.heart_beat_timeout = heart_beat_timeout
        self.handlers: [FLComponent] = handlers
        # self.cmd_modules = cmd_modules

        self.client_manager = ClientManager(
            project_name=self.project_name, min_num_clients=self.min_num_clients, max_num_clients=self.max_num_clients
        )

        self.grpc_server = None
        self.admin_server = None
        self.lock = Lock()
        self.fl_ctx = FLContext()
        self.platform = None

        self.shutdown = False
        self.status = ServerStatus.NOT_STARTED

        self.abort_signal = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def get_all_clients(self):
        return self.client_manager.get_clients()

    @abstractmethod
    def remove_client_data(self, token):
        pass

    def close(self):
        """
        shutdown the server.

        :return:
        """
        try:
            if self.lock:
                self.lock.release()
        except RuntimeError:
            self.logger.info("canceling sync locks")
        try:
            if self.grpc_server:
                self.grpc_server.stop(0)
            # if self.handlers:
            #     for h in self.handlers:
            #         h.shutdown(self.fl_ctx)
            # if self.admin_server:
            #     self.admin_server.stop()
        finally:
            self.logger.info("server off")
            return 0

    def deploy(self, grpc_args=None, secure_train=False):
        """
        start a grpc server and listening the designated port.
        :param grpc_args:
        """
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
            self.grpc_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=num_server_workers),
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
            self.grpc_server.add_secure_port(target, server_credentials)
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
            # self.tokens.pop(token, None)
            self.remove_client_data(token)
            if self.admin_server:
                self.admin_server.client_dead(client.name)
            self.logger.info(
                "Remove the dead Client. Name: {}\t Token: {}.  Total clients: {}".format(
                    client.name, token, len(self.client_manager.get_clients())
                )
            )

    def fl_shutdown(self):
        self.shutdown = True
        self.close()


class FederatedServer(BaseServer, fed_service.FederatedTrainingServicer, admin_service.AdminCommunicatingServicer):
    """
    Federated server services
    """

    def __init__(
        self,
        project_name=None,
        min_num_clients=2,
        max_num_clients=10,
        wait_after_min_clients=10,
        cmd_modules=None,
        heart_beat_timeout=600,
        handlers: Optional[List[FLComponent]] = None,
        args=None,
        secure_train=False,
        enable_byoc=False
    ):
        """

        :param project_name: server project name.
        :param min_num_clients: minimum number of contributors at each round.
        :param max_num_clients: maximum number of contributors at each round.
        """
        # assert model_log_dir is not None, "model_log_dir must be specified to save checkpoint"

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

        self.wait_after_min_clients = wait_after_min_clients

        self.cmd_modules = cmd_modules

        self.builder = None

        # Additional fields for CurrentTask meta_data in GetModel API.
        self.current_model_meta_data = {}

        self.engine = ServerEngine(self, args, self.client_manager)
        self.run_manager = None
        self.server_runner = None

        self.processors = {}
        self.runner_config = None
        self.secure_train = secure_train
        self.enable_byoc = enable_byoc

    def get_current_model_meta_data(self):
        """Get the model meta data, which usually contains additional fields"""
        s = Struct()
        for k, v in self.current_model_meta_data.items():
            s.update({k: v})
        return s

    @property
    def task_meta_info(self):
        """
        the model_meta_info uniquely defines the current model,
        it is used to reject outdated client's update.

        :return: model meta data object
        """
        meta_info = fed_msg.MetaData()
        meta_info.created.CopyFrom(self.round_started)
        meta_info.project.name = self.project_name
        return meta_info

    # def register_processor(self, processor: ResultProcessor):
    #     # assert isinstance(processor, ResultProcessor), 'processor must be ResultProcessor'
    #
    #     topics = processor.get_topics()
    #     for topic in topics:
    #         assert topic not in self.processors, "duplicate processors for topic {}".format(topic)
    #         self.processors[topic] = processor
    #
    def remove_client_data(self, token):
        self.tokens.pop(token, None)

    def reset_tokens(self):
        """
        restart the token set, so that each client can take a token
        and start fetching the current global model.
        This function is not thread-safe.
        """
        # last_time = self.round_started.seconds
        # self.round_started.GetCurrentTime()
        # now_time = self.round_started.seconds
        # reset_duration = now_time - last_time
        # self.logger.info("Round time: %s second(s).", reset_duration if reset_duration > 0 else "less than a")

        self.tokens = dict()
        for client in self.get_all_clients().keys():
            self.tokens[client] = self.task_meta_info

    def Register(self, request, context):
        """
        register new clients on the fly.
        Each client must get registered before getting the global model.
        The server will expect updates from the registered clients
        for multiple federated rounds.

        This function does not change min_num_clients and max_num_clients.
        """
        # fire_event(EventType.CLIENT_REGISTER, self.handlers, self.fl_ctx)

        token = self.client_manager.authenticate(request, context)
        if token:
            self.tokens[token] = self.task_meta_info
            if self.admin_server:
                self.admin_server.client_heartbeat(token)

            return fed_msg.FederatedSummary(comment="New client registered", token=token)

    def Quit(self, request, context):
        """
        existing client quits the federated training process.
        Server will stop sharing the global model with the client,
        further contribution will be rejected.

        This function does not change min_num_clients and max_num_clients.
        """
        # fire_event(EventType.CLIENT_QUIT, self.handlers, self.fl_ctx)

        client = self.client_manager.validate_client(request, context)
        if client:
            token = client.get_token()

            client = self.client_manager.remove_client(token)
            self.tokens.pop(token, None)
            if self.admin_server:
                self.admin_server.client_dead(token)

        return fed_msg.FederatedSummary(comment="Removed client")

    def GetTask(self, request, context):
        """
        process client's request of the current global model
        """
        # # fl_ctx = self.fl_ctx.clone_sticky()
        # if not self.run_manager:
        #     context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server training stopped")

        # if self.server_runner is None:
        #     context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server has stopped")

        with self.engine.new_context() as fl_ctx:

            client = self.client_manager.validate_client(request, context)
            if client is None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Client not valid.")

            self.logger.info(f"Fetch task requested from client: {client.name} ({client.get_token()})")
            token = client.get_token()

            engine = fl_ctx.get_engine()
            # shared_fl_ctx = FLContext()
            # shared_fl_ctx.set_run_number(request.meta.run_number)
            shared_fl_ctx = pickle.loads(proto_to_bytes(request.context["fl_context"]))
            fl_ctx.set_peer_context(shared_fl_ctx)

            with self.lock:
                # shareable = self.model_manager.get_shareable(self.fl_ctx)

                if self.server_runner is None or engine is None or self.engine.run_manager is None:
                    self.logger.info("server has no current run - asked client to end the run")
                    taskname = SpecialTaskName.END_RUN
                    task_id = ""
                    shareable = None
                else:
                    # taskname, task_id, shareable = self.controller.process_task_request(client, fl_ctx)
                    taskname, task_id, shareable = self.server_runner.process_task_request(client, fl_ctx)

                if shareable is None:
                    shareable = Shareable()

                task = fed_msg.CurrentTask(task_name=taskname)
                task.meta.CopyFrom(self.task_meta_info)
                meta_data = self.get_current_model_meta_data()

                # we need TASK_ID back as a cookie
                shareable.add_cookie(name=FLContextKey.TASK_ID, data=task_id)

                # we also need to make TASK_ID available to the client
                shareable.set_header(key=FLContextKey.TASK_ID, value=task_id)

                task.meta_data.CopyFrom(meta_data)

                current_model = shareable_to_modeldata(shareable, fl_ctx)
                task.data.CopyFrom(current_model)
                self.logger.info(f"Return task:{taskname} to client:{client.name} --- ({token}) ")

                # self.fl_ctx.merge_sticky(fl_ctx)

                return task

    def SubmitUpdate(self, request, context):
        """
        handling client's submission of the federated updates
        running aggregation if there are enough updates
        """
        # if not self.run_manager:
        #     context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server has stopped")

        if self.server_runner is None or self.engine.run_manager is None:
            # context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server has stopped")
            self.logger.info("ignored result submission since Server Engine isn't ready")
            context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server has stopped")

        # fl_ctx = self.fl_ctx.clone_sticky()
        with self.engine.new_context() as fl_ctx:

            # if self.status == ServerStatus.TRAINING_STOPPED or self.status == ServerStatus.TRAINING_NOT_STARTED:
            #     context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server training stopped")
            #     return

            contribution = request

            client = self.client_manager.validate_client(contribution.client, context)
            if client is None:
                response_comment = "Ignored the submit from invalid client. "
                self.logger.info(response_comment)
            else:
                token = client.get_token()

                with self.lock:
                    shareable = Shareable()
                    shareable = shareable.from_bytes(proto_to_bytes(request.data.params["data"]))
                    shared_fl_context = pickle.loads(proto_to_bytes(request.data.params["fl_context"]))

                    # fl_ctx.set_prop(FLContextKey.PEER_CONTEXT, shared_fl_context)
                    fl_ctx.set_peer_context(shared_fl_context)

                    shared_fl_context.set_prop(FLContextKey.SHAREABLE, shareable, private=False)

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

                    # fire_event(EventType.BEFORE_PROCESS_SUBMISSION, self.handlers, fl_ctx)

                    if self.save_contribution(client_contrib_id, contribution):
                        assert isinstance(shareable, Shareable)
                        # task_id = shared_fl_context.get_cookie(FLContextKey.TASK_ID)
                        task_id = shareable.get_cookie(FLContextKey.TASK_ID)
                        self.server_runner.process_submission(
                            client, contribution_task_name, task_id, shareable, fl_ctx
                        )

                        # fire_event(EventType.AFTER_PROCESS_SUBMISSION, self.handlers, fl_ctx)

            response_comment = "Received from {} ({} Bytes, {} seconds)".format(
                contribution.client.client_name,
                contribution.ByteSize(),
                time_seconds or "less than 1",
            )
            summary_info = fed_msg.FederatedSummary(comment=response_comment)
            summary_info.meta.CopyFrom(self.task_meta_info)

            # with self.lock:
            #     self.fl_ctx.merge_sticky(fl_ctx)

            return summary_info

    def AuxCommunicate(self, request, context):
        """
        handling client's submission of the federated updates
        running aggregation if there are enough updates
        """
        # if not self.run_manager:
        #     context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server has stopped")

        if self.server_runner is None or self.engine.run_manager is None:
            # context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server has stopped")
            self.logger.info("ignored result submission since Server Engine isn't ready")
            context.abort(grpc.StatusCode.OUT_OF_RANGE, "Server has stopped")

        # fl_ctx = self.fl_ctx.clone_sticky()
        with self.engine.new_context() as fl_ctx:

            contribution = request

            client = self.client_manager.validate_client(contribution.client, context)
            if client is None:
                response_comment = "Ignored the submit from invalid client. "
                self.logger.info(response_comment)
            else:
                token = client.get_token()

            shareable = Shareable()
            shareable = shareable.from_bytes(proto_to_bytes(request.data["data"]))
            shared_fl_context = pickle.loads(proto_to_bytes(request.data["fl_context"]))

            fl_ctx.set_peer_context(shared_fl_context)
            shareable.set_peer_props(shared_fl_context.get_all_public_props())

            shared_fl_context.set_prop(FLContextKey.SHAREABLE, shareable, private=False)

            topic = shareable.get_header(ReservedHeaderKey.TOPIC)
            # aux_runner = fl_ctx.get_aux_runner()
            # assert isinstance(aux_runner, ServerAuxRunner)
            reply = self.engine.dispatch(topic=topic, request=shareable, fl_ctx=fl_ctx)

            aux_reply = fed_msg.AuxReply()
            aux_reply.data.CopyFrom(shareable_to_modeldata(reply, fl_ctx))

            return aux_reply

    def Heartbeat(self, request, context):
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

        summary_info = fed_msg.FederatedSummary()
        return summary_info

    def Retrieve(self, request, context):
        client_name = request.client_name
        messages = self.admin_server.get_outgoing_requests(client_token=client_name) if self.admin_server else []

        response = admin_msg.Messages()
        # response.message.CopyFrom(messages)
        for m in messages:
            # message = response.message.add()
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

    def save_contribution(self, client_contrib_id, data):
        """
        save the client's current contribution.

        :return: True iff it is successfully saved
        """
        return True

    def start_run(self, run_number, run_root, conf, args):
        # self.status = ServerStatus.STARTING

        # Create the FL Engine
        workspace = Workspace(args.workspace, "server", args.config_folder)
        self.run_manager = RunManager(
            server_name=self.project_name,
            engine=self.engine,
            run_num=run_number,
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
            with self.engine.new_context() as fl_ctx:

                # with open(os.path.join(run_root, env_config)) as file:
                #     env = json.load(file)

                fl_ctx.set_prop(FLContextKey.APP_ROOT, run_root, sticky=True)
                fl_ctx.set_prop(FLContextKey.CURRENT_RUN, run_number, private=False, sticky=True)
                fl_ctx.set_prop(FLContextKey.WORKSPACE_ROOT, args.workspace, private=True, sticky=True)
                fl_ctx.set_prop(FLContextKey.ARGS, args, private=True, sticky=True)
                fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True)
                fl_ctx.set_prop(FLContextKey.SECURE_MODE, self.secure_train, private=True, sticky=True)

            self.server_runner = ServerRunner(config=self.runner_config, run_num=run_number, engine=self.engine)
            self.run_manager.add_handler(self.server_runner)

            # self.controller.initialize_run(self.fl_ctx)

            # return super().start()
            # self.status = ServerStatus.STARTED
            # self.run_engine()
            engine_thread = threading.Thread(target=self.run_engine)
            # heartbeat_thread.daemon = True
            engine_thread.start()

            while self.engine.engine_info.status != MachineStatus.STOPPED:
                # self.remove_dead_clients()
                if self.engine.asked_to_stop:
                    self.engine.abort_app_on_server()

                time.sleep(3)

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

    def stop_training(self):
        self.status = ServerStatus.STOPPED
        self.logger.info("Server app stopped.\n\n")

    def close(self):
        """
        shutdown the server.

        :return:
        """
        self.logger.info("shutting down server")

        return super().close()
