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

from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import FLContextKey, ServerCommandKey, ServerCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.fuel.f3.cellnet.cell import FQCN, Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.utils import fobs
from nvflare.private.defs import CellChannel, CellChannelTopic, CellMessageHeaderKeys, SpecialTaskName, new_cell_message
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.security.logging import secure_format_exception


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


class Communicator:
    def __init__(
        self,
        ssl_args=None,
        secure_train=False,
        retry_timeout=30,
        client_state_processors: Optional[List[Filter]] = None,
        compression=None,
        cell: Cell = None,
    ):
        """To init the Communicator.

        Args:
            ssl_args: SSL args
            secure_train: True/False to indicate if secure train
            retry_timeout: retry timeout in seconds
            client_state_processors: Client state processor filters
            compression: communicate compression algorithm
        """
        self.cell = cell
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

        # login_message = fed_msg.ClientLogin(client_name=client_name, client_ip=local_ip)
        # login_message.meta.project.name = project_name
        #
        # with self.set_up_channel(servers[project_name]) as channel:
        #     stub = fed_service.FederatedTrainingStub(channel)
        #     while True:
        #         try:
        #             result = stub.Register(login_message)
        #             token = result.token
        #             ssid = result.ssid
        #             self.should_stop = False
        #             break
        #         except grpc.RpcError as grpc_error:
        #             self.grpc_error_handler(
        #                 servers[project_name],
        #                 grpc_error,
        #                 "client_registration",
        #                 verbose=self.verbose,
        #             )
        #             excep = FLCommunicationError(grpc_error, "grpc_error:client_registration")
        #             if isinstance(grpc_error, grpc.Call):
        #                 status_code = grpc_error.code()
        #                 if grpc.StatusCode.UNAUTHENTICATED == status_code:
        #                     raise excep
        #             time.sleep(5)
        #     if self.should_stop:
        #         raise excep
        #     if result is None:
        #         return None

        login_message = new_cell_message(
            {
                CellMessageHeaderKeys.CLIENT_NAME: client_name,
                CellMessageHeaderKeys.CLIENT_IP: local_ip,
                CellMessageHeaderKeys.PROJECT_NAME: project_name,
            }
        )

        while True:
            try:
                result = self.cell.send_request(
                    target=FQCN.ROOT_SERVER,
                    channel=CellChannel.TASK,
                    topic=CellChannelTopic.Register,
                    request=login_message,
                )
                return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
                if return_code == ReturnCode.UNAUTHENTICATED:
                    unauthenticated = result.get_header(MessageHeaderKey.ERROR)
                    raise FLCommunicationError({}, "error:client_registration " + unauthenticated)

                token = result.get_header(CellMessageHeaderKeys.TOKEN)
                ssid = result.get_header(CellMessageHeaderKeys.SSID)
                if not token and not self.should_stop:
                    time.sleep(5)
                else:
                    break

            except BaseException as ex:
                raise FLCommunicationError(ex, "error:client_registration")

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
        # task, retry = None, self.retry
        # with self.set_up_channel(servers[project_name]) as channel:
        #     stub = fed_service.FederatedTrainingStub(channel)
        #     while retry > 0:
        #         try:
        #             start_time = time.time()
        #             task = stub.GetTask(_get_client_state(project_name, token, ssid, fl_ctx))
        #             # Clear the stopping flag
        #             # if the connection to server recovered.
        #             self.should_stop = False
        #
        #             end_time = time.time()
        #
        #             if task.task_name == SpecialTaskName.TRY_AGAIN:
        #                 self.logger.debug(
        #                     f"Received from {project_name} server "
        #                     f" ({task.ByteSize()} Bytes). getTask time: {end_time - start_time} seconds"
        #                 )
        #             else:
        #                 self.logger.info(
        #                     f"Received from {project_name} server "
        #                     f" ({task.ByteSize()} Bytes). getTask time: {end_time - start_time} seconds"
        #                 )
        #             return task
        #         except grpc.RpcError as grpc_error:
        #             self.grpc_error_handler(servers[project_name], grpc_error, "getTask", verbose=self.verbose)
        #             excep = FLCommunicationError(grpc_error, "grpc_error:getTask")
        #             retry -= 1
        #             time.sleep(5)
        #     if self.should_stop:
        #         raise excep

        start_time = time.time()
        shareable = Shareable()
        shared_fl_ctx = FLContext()
        shared_fl_ctx.set_public_props(get_serializable_data(fl_ctx).get_all_public_props())
        shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)
        task_message = new_cell_message(
            {
                CellMessageHeaderKeys.TOKEN: token,
                CellMessageHeaderKeys.SSID: ssid,
                CellMessageHeaderKeys.PROJECT_NAME: project_name,
            },
            fobs.dumps(shareable),
        )
        job_id = str(shared_fl_ctx.get_prop(FLContextKey.CURRENT_RUN))

        fqcn = FQCN.join([FQCN.ROOT_SERVER, job_id])
        task = self.cell.send_request(
            target=fqcn, channel=CellChannel.SERVER_COMMAND, topic=ServerCommandNames.GET_TASK, request=task_message
        )
        end_time = time.time()
        return_code = task.get_header(MessageHeaderKey.RETURN_CODE)

        if return_code == ReturnCode.OK:
            size = len(task.payload)
            task.payload = fobs.loads(task.payload)
            task_name = task.payload.get_header(ServerCommandKey.TASK_NAME)
            if task_name == SpecialTaskName.TRY_AGAIN:
                time.sleep(5)
            self.logger.info(
                f"Received from {project_name} server "
                f" ({size} Bytes). getTask: {task_name} time: {end_time - start_time} seconds"
            )
        else:
            task = None
            time.sleep(5)

        return task

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
        # client_state = _get_client_state(project_name, token, ssid, fl_ctx)
        # client_state.client_name = client_name
        # contrib = _get_communication_data(shareable, client_state, fl_ctx, execute_task_name)
        #
        # server_msg, retry = None, self.retry
        # with self.set_up_channel(servers[project_name]) as channel:
        #     stub = fed_service.FederatedTrainingStub(channel)
        #     while retry > 0:
        #         try:
        #             start_time = time.time()
        #             self.logger.info(f"Send submitUpdate to {project_name} server")
        #             server_msg = stub.SubmitUpdate(contrib)
        #             # Clear the stopping flag
        #             # if the connection to server recovered.
        #             self.should_stop = False
        #
        #             end_time = time.time()
        #             self.logger.info(
        #                 f"Received comments: {server_msg.meta.project.name} {server_msg.comment}."
        #                 f" SubmitUpdate time: {end_time - start_time} seconds"
        #             )
        #             break
        #         except grpc.RpcError as grpc_error:
        #             if isinstance(grpc_error, grpc.Call):
        #                 if grpc_error.details().startswith("Contrib"):
        #                     self.logger.info(f"submitUpdate failed: {grpc_error.details()}")
        #                     break  # outdated contribution, no need to retry
        #             self.grpc_error_handler(servers[project_name], grpc_error, "submitUpdate", verbose=self.verbose)
        #             retry -= 1
        #             time.sleep(5)
        # return server_msg

        start_time = time.time()
        shared_fl_ctx = FLContext()
        shared_fl_ctx.set_public_props(get_serializable_data(fl_ctx).get_all_public_props())
        shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)

        # shareable.add_cookie(name=FLContextKey.TASK_ID, data=task_id)
        shareable.set_header(FLContextKey.TASK_NAME, execute_task_name)

        task_message = new_cell_message(
            {
                CellMessageHeaderKeys.TOKEN: token,
                CellMessageHeaderKeys.SSID: ssid,
                CellMessageHeaderKeys.PROJECT_NAME: project_name,
            },
            fobs.dumps(shareable),
        )
        job_id = str(shared_fl_ctx.get_prop(FLContextKey.CURRENT_RUN))

        fqcn = FQCN.join([FQCN.ROOT_SERVER, job_id])
        result = self.cell.send_request(
            target=fqcn,
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.SUBMIT_UPDATE,
            request=task_message,
        )
        end_time = time.time()
        return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
        self.logger.info(f" SubmitUpdate time: {end_time - start_time} seconds")

        return return_code

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
        # client_state = _get_client_state(project_name, token, ssid, fl_ctx)
        # client_state.client_name = client_name
        #
        # aux_message = fed_msg.AuxMessage()
        # # set client auth. data
        # aux_message.client.CopyFrom(client_state)
        #
        # # shareable.set_header("Topic", topic)
        # aux_message.data["data"].CopyFrom(make_shareable_data(shareable))
        # aux_message.data["fl_context"].CopyFrom(make_context_data(fl_ctx))
        #
        # server_msg, retry = None, self.retry
        # with self.set_up_channel(servers[project_name]) as channel:
        #     stub = fed_service.FederatedTrainingStub(channel)
        #     while retry > 0:
        #         try:
        #             self.logger.debug(f"Send AuxMessage to {project_name} server")
        #             server_msg = stub.AuxCommunicate(aux_message, timeout=timeout)
        #             # Clear the stopping flag
        #             # if the connection to server recovered.
        #             self.should_stop = False
        #
        #             break
        #         except grpc.RpcError as grpc_error:
        #             self.grpc_error_handler(servers[project_name], grpc_error, "AuxCommunicate", verbose=self.verbose)
        #             retry -= 1
        #             time.sleep(5)
        # return server_msg

        start_time = time.time()
        shared_fl_ctx = FLContext()
        shared_fl_ctx.set_public_props(get_serializable_data(fl_ctx).get_all_public_props())
        shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)

        # shareable.add_cookie(name=FLContextKey.TASK_ID, data=task_id)
        shareable.set_header(ServerCommandKey.TOPIC, topic)

        task_message = new_cell_message(
            {
                CellMessageHeaderKeys.TOKEN: token,
                CellMessageHeaderKeys.SSID: ssid,
                CellMessageHeaderKeys.PROJECT_NAME: project_name,
            },
            fobs.dumps(shareable),
        )
        job_id = str(shared_fl_ctx.get_prop(FLContextKey.CURRENT_RUN))

        fqcn = FQCN.join([FQCN.ROOT_SERVER, job_id])
        result = self.cell.send_request(
            target=fqcn,
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.AUX_COMMUNICATE,
            request=task_message,
        )
        end_time = time.time()
        return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
        self.logger.debug(f"Send AuxMessage to server. time: {end_time - start_time} seconds")

        return result

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
        # server_message, retry = None, self.retry
        # with self.set_up_channel(servers[task_name]) as channel:
        #     stub = fed_service.FederatedTrainingStub(channel)
        #     while retry > 0:
        #         try:
        #             start_time = time.time()
        #             self.logger.info(f"Quitting server: {task_name}")
        #             server_message = stub.Quit(_get_client_state(task_name, token, ssid, fl_ctx))
        #             # Clear the stopping flag
        #             # if the connection to server recovered.
        #             self.should_stop = False
        #
        #             end_time = time.time()
        #             self.logger.info(
        #                 f"Received comment from server: {server_message.comment}. Quit time: {end_time - start_time} seconds"
        #             )
        #             break
        #         except grpc.RpcError as grpc_error:
        #             self.grpc_error_handler(servers[task_name], grpc_error, "quit_remote")
        #             retry -= 1
        #             time.sleep(3)
        # return server_message

        quit_message = new_cell_message(
            {
                CellMessageHeaderKeys.TOKEN: token,
                CellMessageHeaderKeys.SSID: ssid,
                CellMessageHeaderKeys.PROJECT_NAME: task_name,
            }
        )
        try:
            result = self.cell.send_request(
                target=FQCN.ROOT_SERVER, channel=CellChannel.TASK, topic=CellChannelTopic.Quit, request=quit_message
            )
            return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
            if return_code == ReturnCode.UNAUTHENTICATED:
                unauthenticated = result.get_header(MessageHeaderKey.ERROR)
                raise FLCommunicationError({}, "error:client_quit " + unauthenticated)

            server_message = result.get_header(CellMessageHeaderKeys.MESSAGE)

        except BaseException as ex:
            raise FLCommunicationError(ex, "error:client_quit")

        return server_message

    def send_heartbeat(self, servers, task_name, token, ssid, client_name, engine: ClientEngineInternalSpec):
        # message = fed_msg.Token()
        # message.token = token
        # message.ssid = ssid
        # message.client_name = client_name

        while not self.heartbeat_done:
            try:
                # with self.set_up_channel(servers[task_name]) as channel:
                #     stub = fed_service.FederatedTrainingStub(channel)
                #     # retry the heartbeat call for 10 minutes
                #     retry = 2
                #     while retry > 0:
                #         try:
                #             self.logger.debug(f"Send {task_name} heartbeat {token}")
                #             job_ids = engine.get_all_job_ids()
                #             del message.jobs[:]
                #             message.jobs.extend(job_ids)
                #             response = stub.Heartbeat(message)
                #             self._clean_up_runs(engine, response)
                #             break
                #         except grpc.RpcError:
                #             retry -= 1
                #             time.sleep(5)

                job_ids = engine.get_all_job_ids()
                heartbeat_message = new_cell_message(
                    {
                        CellMessageHeaderKeys.TOKEN: token,
                        CellMessageHeaderKeys.SSID: ssid,
                        CellMessageHeaderKeys.CLIENT_NAME: client_name,
                        CellMessageHeaderKeys.PROJECT_NAME: task_name,
                        CellMessageHeaderKeys.JOB_IDS: job_ids,
                    }
                )

                try:
                    result = self.cell.send_request(
                        target=FQCN.ROOT_SERVER,
                        channel=CellChannel.TASK,
                        topic=CellChannelTopic.HEART_BEAT,
                        request=heartbeat_message,
                    )
                    return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
                    if return_code == ReturnCode.UNAUTHENTICATED:
                        unauthenticated = result.get_header(MessageHeaderKey.ERROR)
                        raise FLCommunicationError({}, "error:client_quit " + unauthenticated)

                    # server_message = result.get_header(CellMessageHeaderKeys.MESSAGE)
                    abort_jobs = result.get_header(CellMessageHeaderKeys.ABORT_JOBS, [])
                    self._clean_up_runs(engine, abort_jobs)

                except BaseException as ex:
                    raise FLCommunicationError(ex, "error:client_quit")

                time.sleep(30)
            except BaseException as e:
                self.logger.info(f"Failed to send heartbeat. Will try again. Exception: {secure_format_exception(e)}")
                time.sleep(5)

    def _clean_up_runs(self, engine, abort_runs):
        # abort_runs = list(set(response.abort_jobs))
        display_runs = ",".join(abort_runs)
        try:
            if abort_runs:
                for job in abort_runs:
                    engine.abort_app(job)
                self.logger.info(f"These runs: {display_runs} are not running on the server. Aborted them.")
        except:
            self.logger.info(f"Failed to clean up the runs: {display_runs}")
