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
import traceback
import uuid
from typing import List, Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.fl_constant import ReturnCode as ShareableRC
from nvflare.apis.fl_constant import SecureTrainConst, ServerCommandKey, ServerCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.fl_context_utils import gen_new_peer_ctx
from nvflare.fuel.f3.cellnet.core_cell import FQCN, CoreCell
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import format_size
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.private.defs import CellChannel, CellChannelTopic, CellMessageHeaderKeys, SpecialTaskName, new_cell_message
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.utils.fed_utils import get_scope_prop, set_scope_prop
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, IdentityVerifier, load_crt_bytes
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
        cell: CoreCell = None,
        client_register_interval=2,
        timeout=5.0,
        maint_msg_timeout=5.0,
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
        self.maint_msg_timeout = maint_msg_timeout

        # token and token_signature are issued by the Server after the client is authenticated
        # they are added to every message going to the server as proof of authentication
        self.token = None
        self.token_signature = None
        self.ssid = None
        self.client_name = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_auth(self, client_name, token, token_signature, ssid):
        self.ssid = ssid
        self.token_signature = token_signature
        self.token = token
        self.client_name = client_name

        # put auth properties in database so that they can be used elsewhere
        set_scope_prop(scope_name=client_name, key=FLMetaKey.AUTH_TOKEN, value=token)
        set_scope_prop(scope_name=client_name, key=FLMetaKey.AUTH_TOKEN_SIGNATURE, value=token_signature)

    def set_cell(self, cell):
        self.cell = cell

        # set filter to add additional auth headers
        cell.core_cell.add_outgoing_reply_filter(channel="*", topic="*", cb=self._add_auth_headers)
        cell.core_cell.add_outgoing_request_filter(channel="*", topic="*", cb=self._add_auth_headers)

    def _add_auth_headers(self, message: CellMessage):
        if self.ssid:
            message.set_header(CellMessageHeaderKeys.SSID, self.ssid)

        if self.client_name:
            message.set_header(CellMessageHeaderKeys.CLIENT_NAME, self.client_name)

        if self.token:
            message.set_header(CellMessageHeaderKeys.TOKEN, self.token)
            message.set_header(CellMessageHeaderKeys.TOKEN_SIGNATURE, self.token_signature)

    def _challenge_server(self, client_name, expected_host, root_cert_file):
        # ask server for its info and make sure that it matches expected host
        my_nonce = str(uuid.uuid4())
        headers = {IdentityChallengeKey.COMMON_NAME: client_name, IdentityChallengeKey.NONCE: my_nonce}
        challenge = new_cell_message(headers, None)
        result = self.cell.send_request(
            target=FQCN.ROOT_SERVER,
            channel=CellChannel.SERVER_MAIN,
            topic=CellChannelTopic.Challenge,
            request=challenge,
            timeout=self.maint_msg_timeout,
        )
        return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
        error = result.get_header(MessageHeaderKey.ERROR, "")
        self.logger.info(f"challenge result: {return_code} {error}")
        if return_code != ReturnCode.OK:
            if return_code in [ReturnCode.TARGET_UNREACHABLE, ReturnCode.COMM_ERROR]:
                # trigger retry
                return None
            err = result.get_header(MessageHeaderKey.ERROR, "")
            raise FLCommunicationError(f"failed to challenge server: {return_code}: {err}")

        reply = result.payload
        assert isinstance(reply, Shareable)
        server_nonce = reply.get(IdentityChallengeKey.NONCE)
        cert_bytes = reply.get(IdentityChallengeKey.CERT)
        server_cert = load_crt_bytes(cert_bytes)
        server_signature = reply.get(IdentityChallengeKey.SIGNATURE)
        server_cn = reply.get(IdentityChallengeKey.COMMON_NAME)

        if server_cn != expected_host:
            raise FLCommunicationError(f"expected server identity is '{expected_host}' but got '{server_cn}'")

        # Use IdentityVerifier to validate:
        # - the server cert can be validated with the root cert. Note that all sites have the same root cert!
        # - the asserted CN matches the CN on the server cert
        # - signature received from the server is valid
        id_verifier = IdentityVerifier(root_cert_file=root_cert_file)
        id_verifier.verify_common_name(
            asserter_cert=server_cert, asserted_cn=server_cn, nonce=my_nonce, signature=server_signature
        )

        self.logger.info(f"verified server identity '{expected_host}'")
        return server_nonce

    def client_registration(self, client_name, project_name, fl_ctx: FLContext):
        """Register the client with the FLARE Server.
        Note that the client no longer needs to be directly connected with the Server!
        Since the client may be connected with the Server indirectly (e.g. via bridge nodes or proxy), in the secure
        mode, the client authentication cannot be based on the connection's TLS cert. Instead, the server and the
        client will explicitly authenticate each other using their provisioned PKI credentials, as follows:
        1. Make sure that the Server is authentic. The client sends a Challenge request with a random nonce.
        The server is expected to return the following in its reply:
            - its cert and common name (Server_CN)
            - signature on the received client nonce + Server_CN
            - a random Server Nonce. This will be used for the server to validate the client's identity in the
            Registration request.
        The client then validates to make sure:
            - the Server_CN is the same as presented in the server cert
            - the Server_CN is the same as configured in the client's config (fed_client.json)
            - the signature is valid
        2. Client sends Registration request that contains:
            - client cert and common name (Client_CN)
            - signature on the received Server Nonce + Client_CN
        The Server then validates to make sure:
            - the Client_CN is the same as presented in the client cert
            - the signature is valid
        NOTE: we do not explicitly validate certs' expiration time. This is because currently the same certs are
        also used for SSL connections, which already validate expiration.

        Args:
            client_name: client name
            project_name: FL study project name
            fl_ctx: FLContext

        Returns:
            The client's token

        """
        start = time.time()
        while not self.cell:
            self.logger.info("Waiting for the client cell to be created.")
            if time.time() - start > 15.0:
                raise RuntimeError("Client cell could not be created. Failed to login the client.")
            time.sleep(0.5)

        local_ip = _get_client_ip()
        shareable = Shareable()
        shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
        shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)

        secure_mode = fl_ctx.get_prop(FLContextKey.SECURE_MODE, False)
        if secure_mode:
            # explicitly authenticate with the Server
            expected_host = None
            server_config = fl_ctx.get_prop(FLContextKey.SERVER_CONFIG)
            if server_config:
                server0 = server_config[0]
                expected_host = server0.get("identity")

            if not expected_host:
                # the provision was done with an old version
                # to be backward compatible, we expect the host to be the server host we connected to
                # we get the host name from DataBus!
                expected_host = get_scope_prop(scope_name=client_name, key=FLContextKey.SERVER_HOST_NAME)

            if not expected_host:
                raise RuntimeError("cannot determine expected_host")

            client_config = fl_ctx.get_prop(FLContextKey.CLIENT_CONFIG)
            if not client_config:
                raise RuntimeError(f"missing {FLContextKey.CLIENT_CONFIG} in FL Context")
            private_key_file = client_config.get(SecureTrainConst.PRIVATE_KEY)
            cert_file = client_config.get(SecureTrainConst.SSL_CERT)
            root_cert_file = client_config.get(SecureTrainConst.SSL_ROOT_CERT)

            while True:
                server_nonce = self._challenge_server(client_name, expected_host, root_cert_file)
                if server_nonce is None and not self.should_stop:
                    # retry
                    self.logger.info(f"re-challenge after {self.client_register_interval} seconds")
                    time.sleep(self.client_register_interval)
                else:
                    break

            id_asserter = IdentityAsserter(private_key_file=private_key_file, cert_file=cert_file)
            cn_signature = id_asserter.sign_common_name(nonce=server_nonce)
            shareable[IdentityChallengeKey.CERT] = id_asserter.cert_data
            shareable[IdentityChallengeKey.SIGNATURE] = cn_signature
            shareable[IdentityChallengeKey.COMMON_NAME] = id_asserter.cn
            self.logger.info(f"sent identity info for client {client_name}")

        headers = {
            CellMessageHeaderKeys.CLIENT_NAME: client_name,
            CellMessageHeaderKeys.CLIENT_IP: local_ip,
            CellMessageHeaderKeys.PROJECT_NAME: project_name,
        }
        login_message = new_cell_message(headers, shareable)

        self.logger.info("Trying to register with server ...")
        while True:
            try:
                result = self.cell.send_request(
                    target=FQCN.ROOT_SERVER,
                    channel=CellChannel.SERVER_MAIN,
                    topic=CellChannelTopic.Register,
                    request=login_message,
                    timeout=self.maint_msg_timeout,
                )
                return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
                self.logger.info(f"register RC: {return_code}")
                if return_code == ReturnCode.UNAUTHENTICATED:
                    reason = result.get_header(MessageHeaderKey.ERROR)
                    self.logger.error(f"registration rejected: {reason}")
                    raise FLCommunicationError("error:client_registration " + reason)

                token = result.get_header(CellMessageHeaderKeys.TOKEN)
                token_signature = result.get_header(CellMessageHeaderKeys.TOKEN_SIGNATURE, "NA")
                ssid = result.get_header(CellMessageHeaderKeys.SSID)
                if not token and not self.should_stop:
                    time.sleep(self.client_register_interval)
                else:
                    self.set_auth(client_name, token, token_signature, ssid)
                    break

            except Exception as ex:
                traceback.print_exc()
                raise FLCommunicationError("error:client_registration", ex)

        return token, token_signature, ssid

    def pull_task(self, project_name, token, ssid, fl_ctx: FLContext, timeout=None):
        """Get a task from server.

        Args:
            project_name: FL study project name
            token: client token
            ssid: service session ID
            fl_ctx: FLContext
            timeout: how long to wait for response from server

        Returns:
            A CurrentTask message from server

        """
        start_time = time.time()
        shareable = Shareable()
        shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
        shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)
        client_name = fl_ctx.get_identity_name()
        task_message = new_cell_message(
            {
                CellMessageHeaderKeys.PROJECT_NAME: project_name,
            },
            shareable,
        )
        job_id = str(shared_fl_ctx.get_prop(FLContextKey.CURRENT_RUN))

        if not timeout:
            timeout = self.timeout

        fqcn = FQCN.join([FQCN.ROOT_SERVER, job_id])
        task = self.cell.send_request(
            target=fqcn,
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.GET_TASK,
            request=task_message,
            timeout=timeout,
            optional=True,
        )
        end_time = time.time()
        return_code = task.get_header(MessageHeaderKey.RETURN_CODE)

        if return_code == ReturnCode.OK:
            size = task.get_header(MessageHeaderKey.PAYLOAD_LEN)
            task_name = task.payload.get_header(ServerCommandKey.TASK_NAME)
            fl_ctx.set_prop(FLContextKey.SSID, ssid, sticky=False)
            if task_name not in [SpecialTaskName.END_RUN, SpecialTaskName.TRY_AGAIN]:
                self.logger.info(
                    f"Received from {project_name} server. getTask: {task_name} size: {format_size(size)} "
                    f"({size} Bytes) time: {end_time - start_time:.6f} seconds"
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
        self, project_name, token, ssid, fl_ctx: FLContext, client_name, shareable, execute_task_name, timeout=None
    ):
        """Submit the task execution result back to the server.

        Args:
            project_name: server project name
            token: client token
            ssid: service session ID
            fl_ctx: fl_ctx
            client_name: client name
            shareable: execution task result shareable
            execute_task_name: execution task name
            timeout: how long to wait for response from server

        Returns:
            ReturnCode
        """
        start_time = time.time()
        shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
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
                CellMessageHeaderKeys.PROJECT_NAME: project_name,
            },
            shareable,
        )
        job_id = str(shared_fl_ctx.get_prop(FLContextKey.CURRENT_RUN))

        if not timeout:
            timeout = self.timeout

        fqcn = FQCN.join([FQCN.ROOT_SERVER, job_id])
        result = self.cell.send_request(
            target=fqcn,
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.SUBMIT_UPDATE,
            request=task_message,
            timeout=timeout,
            optional=optional,
        )
        end_time = time.time()
        return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
        size = task_message.get_header(MessageHeaderKey.PAYLOAD_LEN)
        self.logger.info(
            f" SubmitUpdate size: {format_size(size)} ({size} Bytes). time: {end_time - start_time:.6f} seconds"
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
        shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
        shareable = Shareable()
        shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)
        client_name = fl_ctx.get_identity_name()
        quit_message = new_cell_message(
            {
                CellMessageHeaderKeys.PROJECT_NAME: task_name,
            },
            shareable,
        )
        try:
            result = self.cell.send_request(
                target=FQCN.ROOT_SERVER,
                channel=CellChannel.SERVER_MAIN,
                topic=CellChannelTopic.Quit,
                request=quit_message,
                timeout=self.maint_msg_timeout,
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
                engine.fire_event(EventType.BEFORE_CLIENT_HEARTBEAT, fl_ctx)
                shareable = Shareable()
                shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
                shareable.set_header(ServerCommandKey.PEER_FL_CONTEXT, shared_fl_ctx)

                job_ids = engine.get_all_job_ids()
                heartbeat_message = new_cell_message(
                    {
                        CellMessageHeaderKeys.PROJECT_NAME: task_name,
                        CellMessageHeaderKeys.JOB_IDS: job_ids,
                    },
                    shareable,
                )

                try:
                    result = self.cell.send_request(
                        target=FQCN.ROOT_SERVER,
                        channel=CellChannel.SERVER_MAIN,
                        topic=CellChannelTopic.HEART_BEAT,
                        request=heartbeat_message,
                        timeout=self.maint_msg_timeout,
                    )
                    return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
                    if return_code == ReturnCode.UNAUTHENTICATED:
                        unauthenticated = result.get_header(MessageHeaderKey.ERROR)
                        raise FLCommunicationError("error:client_quit " + unauthenticated)

                    num_heartbeats_sent += 1
                    if num_heartbeats_sent % heartbeats_log_interval == 0:
                        self.logger.debug(f"Client: {client_name} has sent {num_heartbeats_sent} heartbeats.")

                    if not simulate_mode:
                        # server_message = result.get_header(CellMessageHeaderKeys.MESSAGE)
                        abort_jobs = result.get_header(CellMessageHeaderKeys.ABORT_JOBS, [])
                        self._clean_up_runs(engine, abort_jobs)
                    else:
                        if return_code != ReturnCode.OK:
                            break

                except Exception as ex:
                    raise FLCommunicationError("error:client_quit", ex)

                engine.fire_event(EventType.AFTER_CLIENT_HEARTBEAT, fl_ctx)
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
