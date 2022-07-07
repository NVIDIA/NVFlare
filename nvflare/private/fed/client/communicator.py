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
import socket
import time
from typing import List, Optional

import grpc
from google.protobuf.struct_pb2 import Struct

import nvflare.private.fed.protos.federated_pb2 as fed_msg
import nvflare.private.fed.protos.federated_pb2_grpc as fed_service
from nvflare.apis.filter import Filter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.utils.fed_utils import make_context_data, make_shareable_data, shareable_to_modeldata


def _get_client_state(project_name, token, ssid, fl_ctx: FLContext):
    """Client's metadata used to authenticate and communicate.

    Args:
        project_name: FL study project name
        token: client token
        ssid: service session ID
        fl_ctx: FLContext

    Returns:
        A ClientState message

    """
    state_message = fed_msg.ClientState(token=token, ssid=ssid)
    state_message.meta.project.name = project_name
    # state_message.meta.job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)

    context_data = make_context_data(fl_ctx)
    state_message.context["fl_context"].CopyFrom(context_data)

    return state_message


def _get_client_ip():
    """Return localhost IP.

    More robust than ``socket.gethostbyname(socket.gethostname())``. See
    https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib/28950776#28950776
    for more details.

    Returns:
        The host IP

    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # doesn't even have to be reachable
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def _get_communication_data(shareable, client_state, fl_ctx: FLContext, execute_task_name):
    contrib = fed_msg.Contribution()
    # set client auth. data
    contrib.client.CopyFrom(client_state)
    contrib.task_name = execute_task_name

    current_model = shareable_to_modeldata(shareable, fl_ctx)
    contrib.data.CopyFrom(current_model)

    s = Struct()
    contrib.meta_data.CopyFrom(s)

    return contrib


class Communicator:
    def __init__(
        self,
        ssl_args=None,
        secure_train=False,
        retry_timeout=30,
        client_state_processors: Optional[List[Filter]] = None,
        compression=None,
    ):
        """To init the Communicator.

        Args:
            ssl_args: SSL args
            secure_train: True/False to indicate if secure train
            retry_timeout: retry timeout in seconds
            client_state_processors: Client state processor filters
            compression: communicate compression algorithm
        """
        self.ssl_args = ssl_args
        self.secure_train = secure_train

        self.verbose = False
        self.should_stop = False
        self.heartbeat_done = False
        # TODO: should we change this back?
        # self.retry = int(math.ceil(float(retry_timeout) / 5))
        self.retry = 1
        self.client_state_processors = client_state_processors
        self.compression = compression

        self.logger = logging.getLogger(self.__class__.__name__)

    def set_up_channel(self, channel_dict, token=None):
        """Connect client to the server.

        Args:
            channel_dict: grpc channel parameters
            token: client token

        Returns:
            An initialised grpc channel

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

            # make sure that all headers are in lowercase,
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

    def client_registration(self, client_name, servers, project_name):
        """Client's metadata used to authenticate and communicate.

        Args:
            client_name: client name
            servers: FL servers
            project_name: FL study project name

        Returns:
            The client's token

        """
        local_ip = _get_client_ip()

        login_message = fed_msg.ClientLogin(client_name=client_name, client_ip=local_ip)
        login_message.meta.project.name = project_name

        with self.set_up_channel(servers[project_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while True:
                try:
                    result = stub.Register(login_message)
                    token = result.token
                    ssid = result.ssid
                    self.should_stop = False
                    break
                except grpc.RpcError as grpc_error:
                    self.grpc_error_handler(
                        servers[project_name],
                        grpc_error,
                        "client_registration",
                        verbose=self.verbose,
                    )
                    excep = FLCommunicationError(grpc_error)
                    if isinstance(grpc_error, grpc.Call):
                        status_code = grpc_error.code()
                        if grpc.StatusCode.UNAUTHENTICATED == status_code:
                            raise excep
                    time.sleep(5)
            if self.should_stop:
                raise excep
            if result is None:
                return None

        return token, ssid

    def getTask(self, servers, project_name, token, ssid, fl_ctx: FLContext):
        """Get a task from server.

        Args:
            servers: FL servers
            project_name: FL study project name
            token: client token
            ssid: service session ID
            fl_ctx: FLContext

        Returns:
            A CurrentTask message from server

        """
        task, retry = None, self.retry
        with self.set_up_channel(servers[project_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    start_time = time.time()
                    task = stub.GetTask(_get_client_state(project_name, token, ssid, fl_ctx))
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    end_time = time.time()

                    if task.task_name == SpecialTaskName.TRY_AGAIN:
                        self.logger.debug(
                            f"Received from {project_name} server "
                            f" ({task.ByteSize()} Bytes). getTask time: {end_time - start_time} seconds"
                        )
                    else:
                        self.logger.info(
                            f"Received from {project_name} server "
                            f" ({task.ByteSize()} Bytes). getTask time: {end_time - start_time} seconds"
                        )
                    return task
                except grpc.RpcError as grpc_error:
                    self.grpc_error_handler(servers[project_name], grpc_error, "getTask", verbose=self.verbose)
                    excep = FLCommunicationError(grpc_error)
                    retry -= 1
                    time.sleep(5)
            if self.should_stop:
                raise excep

        # Failed to get global, return None
        return None

    def submitUpdate(
        self, servers, project_name, token, ssid, fl_ctx: FLContext, client_name, shareable, execute_task_name
    ):
        """Submit the task execution result back to the server.

        Args:
            servers: FL servers
            project_name: server project name
            token: client token
            ssid: service session ID
            fl_ctx: fl_ctx
            client_name: client name
            shareable: execution task result shareable
            execute_task_name: execution task name

        Returns:
            A FederatedSummary message from the server.
        """
        client_state = _get_client_state(project_name, token, ssid, fl_ctx)
        client_state.client_name = client_name
        contrib = _get_communication_data(shareable, client_state, fl_ctx, execute_task_name)

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
                    self.grpc_error_handler(servers[project_name], grpc_error, "submitUpdate", verbose=self.verbose)
                    retry -= 1
                    time.sleep(5)
        return server_msg

    def auxCommunicate(
        self, servers, project_name, token, ssid, fl_ctx: FLContext, client_name, shareable, topic, timeout
    ):
        """Send the auxiliary communication message to the server.

        Args:
            servers: FL servers
            project_name: server project name
            token: client token
            ssid: service session ID
            fl_ctx: fl_ctx
            client_name: client name
            shareable: aux message shareable
            topic: aux message topic
            timeout: aux communication timeout

        Returns:
            An AuxReply message from server

        """
        client_state = _get_client_state(project_name, token, ssid, fl_ctx)
        client_state.client_name = client_name

        aux_message = fed_msg.AuxMessage()
        # set client auth. data
        aux_message.client.CopyFrom(client_state)

        # shareable.set_header("Topic", topic)
        aux_message.data["data"].CopyFrom(make_shareable_data(shareable))
        aux_message.data["fl_context"].CopyFrom(make_context_data(fl_ctx))

        server_msg, retry = None, self.retry
        with self.set_up_channel(servers[project_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    self.logger.debug(f"Send AuxMessage to {project_name} server")
                    server_msg = stub.AuxCommunicate(aux_message, timeout=timeout)
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    break
                except grpc.RpcError as grpc_error:
                    self.grpc_error_handler(servers[project_name], grpc_error, "AuxCommunicate", verbose=self.verbose)
                    retry -= 1
                    time.sleep(5)
        return server_msg

    def quit_remote(self, servers, task_name, token, ssid, fl_ctx: FLContext):
        """Sending the last message to the server before leaving.

        Args:
            servers: FL servers
            task_name: project name
            token: FL client token
            fl_ctx: FLContext

        Returns:
            server's reply to the last message

        """
        server_message, retry = None, self.retry
        with self.set_up_channel(servers[task_name]) as channel:
            stub = fed_service.FederatedTrainingStub(channel)
            while retry > 0:
                try:
                    start_time = time.time()
                    self.logger.info(f"Quitting server: {task_name}")
                    server_message = stub.Quit(_get_client_state(task_name, token, ssid, fl_ctx))
                    # Clear the stopping flag
                    # if the connection to server recovered.
                    self.should_stop = False

                    end_time = time.time()
                    self.logger.info(
                        f"Received comment from server: {server_message.comment}. Quit time: {end_time - start_time} seconds"
                    )
                    break
                except grpc.RpcError as grpc_error:
                    self.grpc_error_handler(servers[task_name], grpc_error, "quit_remote")
                    retry -= 1
                    time.sleep(3)
        return server_message

    def send_heartbeat(self, servers, task_name, token, ssid, client_name, engine: ClientEngineInternalSpec):
        message = fed_msg.Token()
        message.token = token
        message.ssid = ssid
        message.client_name = client_name

        while not self.heartbeat_done:
            try:
                with self.set_up_channel(servers[task_name]) as channel:
                    stub = fed_service.FederatedTrainingStub(channel)
                    # retry the heartbeat call for 10 minutes
                    retry = 2
                    while retry > 0:
                        try:
                            self.logger.debug(f"Send {task_name} heartbeat {token}")
                            job_ids = engine.get_all_job_ids()
                            del message.jobs[:]
                            message.jobs.extend(job_ids)
                            response = stub.Heartbeat(message)
                            self._clean_up_runs(engine, response)
                            break
                        except grpc.RpcError as grpc_error:
                            self.logger.debug(grpc_error)
                            retry -= 1
                            time.sleep(5)

                    time.sleep(30)
            except BaseException as e:
                self.logger.info(f"Failed to send heartbeat. Will try again. Exception: {str(e)}")
                time.sleep(5)

    def _clean_up_runs(self, engine, response):
        abort_runs = list(set(response.abort_jobs))
        display_runs = ",".join(abort_runs)
        try:
            if abort_runs:
                for job in abort_runs:
                    engine.abort_app(job)
                self.logger.info(f"These runs: {display_runs} are not running on the server. Aborted them.")
        except:
            self.logger.info(f"Failed to clean up the runs: {display_runs}")

    def grpc_error_handler(self, service, grpc_error, action, verbose=False):
        """Handling grpc exceptions.

        Args:
            service: FL service
            grpc_error: grpc error
            action: action to take
            verbose: verbose to error print out
        """
        status_code = None
        if isinstance(grpc_error, grpc.Call):
            status_code = grpc_error.code()

        if grpc.StatusCode.RESOURCE_EXHAUSTED == status_code:
            if grpc_error.details().startswith("No token"):
                self.logger.info("No token for this client in current round. " "Waiting for server new round. ")
                self.should_stop = False
                return

        self.logger.error(f"Action: {action} grpc communication error.")
        if grpc.StatusCode.UNAVAILABLE == status_code:
            self.logger.error(f"Could not connect to server: {service.get('target')}\t {grpc_error.details()}")
            self.should_stop = True

        if grpc.StatusCode.OUT_OF_RANGE == status_code:
            self.logger.error(
                f"Server training has stopped.\t" f"Setting flag for stopping training. {grpc_error.details()}"
            )
            self.should_stop = True

        if verbose:
            self.logger.info(grpc_error)
