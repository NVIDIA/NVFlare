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

import logging
import socket
import time
from typing import List, Optional

from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_constant import ReturnCode as ShareableRC
from nvflare.apis.fl_constant import ServerCommandKey, ServerCommandNames
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
        client_state_processors: Optional[List[Filter]] = None,
        compression=None,
        cell: Cell = None,
        client_register_interval=2,
        timeout=5.0,
    ):
        """To init the Communicator.

        Args:
            ssl_args: SSL args
            secure_train: True/False to indicate if secure train
            client_state_processors: Client state processor filters
            compression: communicate compression algorithm
        """
        self.cell = cell
        self.ssl_args = ssl_args
        self.secure_train = secure_train

        self.verbose = False
        self.should_stop = False
        self.heartbeat_done = False
        self.client_state_processors = client_state_processors
        self.compression = compression
        self.client_register_interval = client_register_interval
        self.timeout = timeout

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

        login_message = new_cell_message(
            {
                CellMessageHeaderKeys.CLIENT_NAME: client_name,
                CellMessageHeaderKeys.CLIENT_IP: local_ip,
                CellMessageHeaderKeys.PROJECT_NAME: project_name,
            }
        )

        start = time.time()
        while not self.cell:
            self.logger.info("Waiting for the client cell to be created.")
            if time.time() - start > 15.0:
                raise RuntimeError("Client cell could not be created. Failed to login the client.")
            time.sleep(0.5)

        while not self.cell.is_cell_connected(FQCN.ROOT_SERVER):
            time.sleep(0.1)
            if time.time() - start > 30.0:
                raise FLCommunicationError("error:Could not connect to the server for client_registration.")

        while True:
            try:
                result = self.cell.send_request(
                    target=FQCN.ROOT_SERVER,
                    channel=CellChannel.SERVER_MAIN,
                    topic=CellChannelTopic.Register,
                    request=login_message,
                    timeout=self.timeout,
                )
                return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
                if return_code == ReturnCode.UNAUTHENTICATED:
                    unauthenticated = result.get_header(MessageHeaderKey.ERROR)
                    raise FLCommunicationError("error:client_registration " + unauthenticated)

                token = result.get_header(CellMessageHeaderKeys.TOKEN)
                ssid = result.get_header(CellMessageHeaderKeys.SSID)
                if not token and not self.should_stop:
                    time.sleep(self.client_register_interval)
                else:
                    break

            except Exception as ex:
                raise FLCommunicationError("error:client_registration", ex)

        return token, ssid

    def pull_task(self, servers, project_name, token, ssid, fl_ctx: FLContext):
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
        start_time = time.time()
        shareable = Shareable()
        shared_fl_ctx = FLContext()
        shared_fl_ctx.set_public_props(get_serializable_data(fl_ctx).get_all_public_props())
        shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)
        client_name = fl_ctx.get_identity_name()
        task_message = new_cell_message(
            {
                CellMessageHeaderKeys.TOKEN: token,
                CellMessageHeaderKeys.CLIENT_NAME: client_name,
                CellMessageHeaderKeys.SSID: ssid,
                CellMessageHeaderKeys.PROJECT_NAME: project_name,
            },
            fobs.dumps(shareable),
        )
        job_id = str(shared_fl_ctx.get_prop(FLContextKey.CURRENT_RUN))

        fqcn = FQCN.join([FQCN.ROOT_SERVER, job_id])
        task = self.cell.send_request(
            target=fqcn,
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.GET_TASK,
            request=task_message,
            timeout=self.timeout,
            optional=True,
        )
        end_time = time.time()
        return_code = task.get_header(MessageHeaderKey.RETURN_CODE)

        if return_code == ReturnCode.OK:
            size = len(task.payload)
            task.payload = fobs.loads(task.payload)
            task_name = task.payload.get_header(ServerCommandKey.TASK_NAME)
            fl_ctx.set_prop(FLContextKey.SSID, ssid)
            if task_name not in [SpecialTaskName.END_RUN, SpecialTaskName.TRY_AGAIN]:
                self.logger.info(
                    f"Received from {project_name} server "
                    f" ({size} Bytes). getTask: {task_name} time: {end_time - start_time} seconds"
                )
        elif return_code == ReturnCode.AUTHENTICATION_ERROR:
            self.logger.warning("get_task request authentication failed.")
            time.sleep(5.0)
            return None
        else:
            task = None
            self.logger.warning(f"Failed to get_task from {project_name} server. Will try it again.")

        return task

    def submit_update(
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
            ReturnCode
        """
        start_time = time.time()
        shared_fl_ctx = FLContext()
        shared_fl_ctx.set_public_props(get_serializable_data(fl_ctx).get_all_public_props())
        shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)

        # shareable.add_cookie(name=FLContextKey.TASK_ID, data=task_id)
        shareable.set_header(FLContextKey.TASK_NAME, execute_task_name)
        task_ssid = fl_ctx.get_prop(FLContextKey.SSID)
        if task_ssid != ssid:
            self.logger.warning("submit_update request failed because SSID mismatch.")
            return ReturnCode.INVALID_SESSION
        rc = shareable.get_return_code()
        optional = rc == ShareableRC.TASK_ABORTED

        task_message = new_cell_message(
            {
                CellMessageHeaderKeys.TOKEN: token,
                CellMessageHeaderKeys.CLIENT_NAME: client_name,
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
            timeout=self.timeout,
            optional=optional,
        )
        end_time = time.time()
        return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
        self.logger.info(
            f" SubmitUpdate size: {len(task_message.payload)} Bytes. time: {end_time - start_time} seconds"
        )

        return return_code

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
        client_name = fl_ctx.get_identity_name()
        quit_message = new_cell_message(
            {
                CellMessageHeaderKeys.TOKEN: token,
                CellMessageHeaderKeys.CLIENT_NAME: client_name,
                CellMessageHeaderKeys.SSID: ssid,
                CellMessageHeaderKeys.PROJECT_NAME: task_name,
            }
        )
        try:
            result = self.cell.send_request(
                target=FQCN.ROOT_SERVER,
                channel=CellChannel.SERVER_MAIN,
                topic=CellChannelTopic.Quit,
                request=quit_message,
                timeout=self.timeout,
            )
            return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
            if return_code == ReturnCode.UNAUTHENTICATED:
                self.logger.info(f"Client token: {token} has been removed from the server.")

            server_message = result.get_header(CellMessageHeaderKeys.MESSAGE)

        except Exception as ex:
            raise FLCommunicationError("error:client_quit", ex)

        return server_message

    def send_heartbeat(self, servers, task_name, token, ssid, client_name, engine: ClientEngineInternalSpec, interval):
        fl_ctx = engine.new_context()
        simulate_mode = fl_ctx.get_prop(FLContextKey.SIMULATE_MODE, False)
        wait_times = int(interval / 2)
        num_heartbeats_sent = 0
        heartbeats_log_interval = 10
        while not self.heartbeat_done:
            try:
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
                        channel=CellChannel.SERVER_MAIN,
                        topic=CellChannelTopic.HEART_BEAT,
                        request=heartbeat_message,
                        timeout=self.timeout,
                    )
                    return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
                    if return_code == ReturnCode.UNAUTHENTICATED:
                        unauthenticated = result.get_header(MessageHeaderKey.ERROR)
                        raise FLCommunicationError("error:client_quit " + unauthenticated)

                    num_heartbeats_sent += 1
                    if num_heartbeats_sent % heartbeats_log_interval == 0:
                        self.logger.info(f"Client: {client_name} has sent {num_heartbeats_sent} heartbeats.")

                    if not simulate_mode:
                        # server_message = result.get_header(CellMessageHeaderKeys.MESSAGE)
                        abort_jobs = result.get_header(CellMessageHeaderKeys.ABORT_JOBS, [])
                        self._clean_up_runs(engine, abort_jobs)
                    else:
                        if return_code != ReturnCode.OK:
                            break

                except Exception as ex:
                    raise FLCommunicationError("error:client_quit", ex)

                for i in range(wait_times):
                    time.sleep(2)
                    if self.heartbeat_done:
                        break
            except Exception as e:
                self.logger.info(f"Failed to send heartbeat. Will try again. Exception: {secure_format_exception(e)}")
                time.sleep(5)

    def _clean_up_runs(self, engine, abort_runs):
        # abort_runs = list(set(response.abort_jobs))
        display_runs = ",".join(abort_runs)
        try:
            if abort_runs:
                for job in abort_runs:
                    engine.abort_app(job)
                self.logger.debug(f"These runs: {display_runs} are not running on the server. Aborted them.")
        except:
            self.logger.debug(f"Failed to clean up the runs: {display_runs}")
