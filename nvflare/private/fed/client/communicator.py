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
import socket
import time
from typing import List, Optional

import grpc
from google.protobuf.struct_pb2 import Struct

import nvflare.private.fed.protos.federated_pb2 as fed_msg
import nvflare.private.fed.protos.federated_pb2_grpc as fed_service
from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.private.fed.utils.fed_utils import shareable_to_modeldata, make_context_data, make_shareeable_data


class Communicator:
    def __init__(
        self,
        ssl_args=None,
        secure_train=False,
        retry_timeout=30,
        client_state_processors: Optional[List[Filter]] = None,
        compression=None,
    ):
        self.ssl_args = ssl_args
        self.secure_train = secure_train

        self.verbose = False
        self.should_stop = False
        self.heartbeat_done = False
        self.retry = int(math.ceil(float(retry_timeout) / 5))
        self.client_state_processors = client_state_processors
        self.compression = compression

        self.logger = logging.getLogger(self.__class__.__name__)

    def set_up_channel(self, channel_dict, token=None):
        """
        Connect client to the server.

        :param channel_dict: grpc channel parameters
        :param token: client token
        :return: an initialised grpc channel
        """
        if self.secure_train:
            with open(self.ssl_args["ssl_root_cert"], "rb") as f:
                trusted_certs = f.read()
            with open(self.ssl_args["ssl_private_key"], "rb") as f:
                private_key = f.read()
            with open(self.ssl_args["ssl_cert"], "rb") as f:
                certificate_chain = f.read()

            credentials = grpc.ssl_channel_credentials(
                certificate_chain=certificate_chain, private_key=private_key, root_certificates=trusted_certs
            )

            # make sure that all headers are in lowecase,
            # otherwise grpc throws an exception
            call_credentials = grpc.metadata_call_credentials(
                lambda context, callback: callback((("x-custom-token", token),), None)
            )
            # use this if you want standard "Authorization" header
            # call_credentials = grpc.access_token_call_credentials(
            #     "x-custom-token")
            composite_credentials = grpc.composite_channel_credentials(credentials, call_credentials)
            channel = grpc.secure_channel(
                **channel_dict, credentials=composite_credentials, compression=self.compression
            )

        else:
            channel = grpc.insecure_channel(**channel_dict, compression=self.compression)
        return channel

    def get_client_state(self, project_name, token, fl_ctx: FLContext):
        """
        Client's meta data used to authenticate and communicate.

        :return: a ClientState message.
        """
        state_message = fed_msg.ClientState(token=token)
        state_message.meta.project.name = project_name
        # if self.client_state_processors and app_ctx:
        #     for t in self.client_state_processors:
        #         state_message = t.process(state_message, app_ctx)
        # state_message.meta.run_number = self.run_number
        state_message.meta.run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)

        context_data = make_context_data(fl_ctx)
        state_message.context["fl_context"].CopyFrom(context_data)

        return state_message

    def get_client_ip(self):
        """Return localhost IP.

        More robust than ``socket.gethostbyname(socket.gethostname())``. See
        https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib/28950776#28950776
        for more details.

        :return: The host IP.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))  # doesn't even have to be reachable
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

    def client_registration(self, client_name, servers, project_name):
        """
        Client's meta data used to authenticate and communicate.

        :return: a ClientLogin message.
        """
        local_ip = self.get_client_ip()

        login_message = fed_msg.ClientLogin(client_name=client_name, client_ip=local_ip)
        # login_message = fed_msg.ClientLogin(
        #     client_id=None, token=None, client_ip=local_ip)
        login_message.meta.project.name = project_name

        result, retry = None, self.retry
        retry = 1500  # retry register for 2 hours (7500s)
        with self.set_up_channel(servers[project_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    start_time = time.time()
                    result = stub.Register(login_message)
                    token = result.token
                    self.should_stop = False
                    break
                except grpc.RpcError as grpc_error:
                    self.grpc_error_handler(
                        servers[project_name],
                        grpc_error,
                        "client_registration",
                        start_time,
                        retry,
                        verbose=self.verbose,
                    )
                    excep = FLCommunicationError(grpc_error)
                    if isinstance(grpc_error, grpc.Call):
                        status_code = grpc_error.code()
                        if grpc.StatusCode.UNAUTHENTICATED == status_code:
                            raise excep
                    retry -= 1
                    time.sleep(5)
            if self.should_stop:
                raise excep
            if result is None:
                return None

        return token

    def getTask(self, servers, project_name, token, fl_ctx: FLContext):
        """
        Get registered with the remote server via channel,
        and fetch the server's model parameters.

        :param project_name: server identifier string
        :return: a CurrentTask message from server
        """
        global_model, retry = None, self.retry
        with self.set_up_channel(servers[project_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                # get the global model
                try:
                    start_time = time.time()
                    global_model = stub.GetTask(self.get_client_state(project_name, token, fl_ctx))
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    end_time = time.time()
                    self.logger.info(
                        f"Received from {project_name} server "
                        f" ({global_model.ByteSize()} Bytes). getTask time: {end_time - start_time} seconds"
                    )

                    task = fed_msg.CurrentTask()
                    task.meta.CopyFrom(global_model.meta)
                    task.meta_data.CopyFrom(global_model.meta_data)
                    task.data.CopyFrom(global_model.data)
                    task.task_name = global_model.task_name

                    return task
                except grpc.RpcError as grpc_error:
                    self.grpc_error_handler(
                        servers[project_name], grpc_error, "getTask", start_time, retry, verbose=self.verbose
                    )
                    excep = FLCommunicationError(grpc_error)
                    retry -= 1
                    time.sleep(5)
            if self.should_stop:
                raise excep

        # Failed to get global, return None
        return None

    def submitUpdate(self, servers, project_name, token, fl_ctx: FLContext, client_name, shareable, execute_task_name):
        """
        Submit the task execution result back to the server.
        Args:
            servers: FL servers
            project_name: server project name
            token: client token
            fl_ctx: fl_ctx
            client_name: client name
            shareable: execution task result shareable
            execute_task_name: execution task name

        Returns: server message from the server

        """
        client_state = self.get_client_state(project_name, token, fl_ctx)
        client_state.client_name = client_name
        contrib = self._get_communication_data(shareable, client_state, fl_ctx, execute_task_name)

        server_msg, retry = None, self.retry
        with self.set_up_channel(servers[project_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    start_time = time.time()
                    self.logger.info(f"Send submitUpdate to {project_name} server")
                    server_msg = stub.SubmitUpdate(contrib)
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    end_time = time.time()
                    self.logger.info(
                        f"Received comments: {server_msg.meta.project.name} {server_msg.comment}."
                        f" SubmitUpdate time: {end_time - start_time} seconds"
                    )
                    break
                except grpc.RpcError as grpc_error:
                    if isinstance(grpc_error, grpc.Call):
                        if grpc_error.details().startswith("Contrib"):
                            self.logger.info(f"submitUpdate failed: {grpc_error.details()}")
                            break  # outdated contribution, no need to retry
                    self.grpc_error_handler(
                        servers[project_name], grpc_error, "submitUpdate", start_time, retry, verbose=self.verbose
                    )
                    retry -= 1
                    time.sleep(5)
        return server_msg

    def auxCommunicate(self, servers, project_name, token, fl_ctx: FLContext, client_name, shareable, topic, timeout):
        """
        send the aux communication message to the server
        Args:
            servers: FL servers
            project_name: server project name
            token: client token
            fl_ctx: fl_ctx
            client_name: client name
            shareable: aux message shareable
            topic: aux message topic
            timeout: aux communication timeout

        Returns: servre response message

        """
        client_state = self.get_client_state(project_name, token, fl_ctx)
        client_state.client_name = client_name

        aux_message = fed_msg.AuxMessage()
        # set client auth. data
        aux_message.client.CopyFrom(client_state)

        # shareable.set_header("Topic", topic)
        aux_message.data["data"].CopyFrom(make_shareeable_data(shareable))
        aux_message.data["fl_context"].CopyFrom(make_context_data(fl_ctx))

        server_msg, retry = None, self.retry
        with self.set_up_channel(servers[project_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    start_time = time.time()
                    self.logger.info(f"Send AuxMessage to {project_name} server")
                    server_msg = stub.AuxCommunicate(aux_message, timeout=timeout)
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    break
                except grpc.RpcError as grpc_error:
                    self.grpc_error_handler(
                        servers[project_name], grpc_error, "AuxCommunicate", start_time, retry, verbose=self.verbose
                    )
                    retry -= 1
                    time.sleep(5)
        return server_msg

    def quit_remote(self, servers, task_name, token, fl_ctx: FLContext):
        """
        Sending the last message to the server before leaving.

        :param task_name: server task identifier
        :return: server's reply to the last message
        """
        server_message, retry = None, self.retry
        with self.set_up_channel(servers[task_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    start_time = time.time()
                    self.logger.info(f"Quitting server: {task_name}")
                    server_message = stub.Quit(self.get_client_state(task_name, token, fl_ctx))
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    end_time = time.time()
                    self.logger.info(
                        f"Received comment from server: {server_message.comment}. Quit time: {end_time - start_time} seconds"
                    )
                    break
                except grpc.RpcError as grpc_error:
                    self.grpc_error_handler(servers[task_name], grpc_error, "quit_remote", start_time, retry)
                    retry -= 1
                    time.sleep(3)
        return server_message

    def send_heartbeat(self, servers, task_name, token, client_name):
        message = fed_msg.Token()
        message.token = token
        message.client_name = client_name

        with self.set_up_channel(servers[task_name]) as channel:
            while not self.heartbeat_done:
                stub = fed_service.FederatedTrainingStub(channel)
                # retry the heartbeat call for 10 minutes
                retry = 120
                while retry > 0:
                    try:
                        self.logger.debug(f"Send {task_name} heartbeat {token}")
                        stub.Heartbeat(message)
                        break
                    except grpc.RpcError as grpc_error:
                        self.logger.debug(grpc_error)
                        excep = FLCommunicationError(grpc_error)
                        retry -= 1
                        time.sleep(5)
                        # pass
                # if retry <= 0:
                #     raise excep

                time.sleep(60)

    def _get_communication_data(self, shareable, client_state, fl_ctx: FLContext, execute_task_name):
        contrib = fed_msg.Contribution()
        # set client auth. data
        contrib.client.CopyFrom(client_state)
        contrib.task_name = execute_task_name

        current_model = shareable_to_modeldata(shareable, fl_ctx)
        contrib.data.CopyFrom(current_model)

        s = Struct()
        contrib.meta_data.CopyFrom(s)

        return contrib

    def grpc_error_handler(self, service, grpc_error, action, start_time, retry, verbose=False):
        """
        Handling grpc exceptions
        :param action:
        :param start_time:
        :param service:
        """
        status_code = None
        if isinstance(grpc_error, grpc.Call):
            status_code = grpc_error.code()

        if grpc.StatusCode.RESOURCE_EXHAUSTED == status_code:
            if grpc_error.details().startswith("No token"):
                self.logger.info("No token for this client in current round. " "Waiting for server new round. ")
                self.should_stop = False
                return

        self.logger.error(
            f"Action: {action} grpc communication error. retry: {retry}, First start till now: {time.time() - start_time} seconds."
        )
        if grpc.StatusCode.UNAVAILABLE == status_code:
            self.logger.error(
                f"Could not connect to server: {service.get('target')}\t"
                f"Setting flag for stopping training. {grpc_error.details()}"
            )
            self.should_stop = True

        if grpc.StatusCode.OUT_OF_RANGE == status_code:
            self.logger.error(
                f"Server training has stopped.\t" f"Setting flag for stopping training. {grpc_error.details()}"
            )
            self.should_stop = True

        if verbose:
            self.logger.info(grpc_error)
