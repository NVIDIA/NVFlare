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
import threading
import time
from typing import List, Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey
from nvflare.apis.fl_constant import ReturnCode as ShareableRC
from nvflare.apis.fl_constant import SecureTrainConst, ServerCommandKey, ServerCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.shareable import Shareable, make_copy
from nvflare.apis.signal import Signal
from nvflare.apis.utils.fl_context_utils import gen_new_peer_ctx
from nvflare.fuel.data_event.utils import get_scope_property, set_scope_property
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import format_size
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.sec.authn import set_add_auth_headers_filters
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.defs import (
    CellChannel,
    CellChannelTopic,
    CellMessageHeaderKeys,
    ClientType,
    SpecialTaskName,
    new_cell_message,
)
from nvflare.private.fed.authenticator import Authenticator
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.security.logging import secure_format_exception

from .utils import determine_parent_fqcn


class Communicator:
    def __init__(
        self,
        client_config=None,
        secure_train=False,
        client_state_processors: Optional[List[Filter]] = None,
        compression=None,
        cell: Cell = None,
        client_register_interval=2,
        timeout=5.0,
        maint_msg_timeout=5.0,
    ):
        """To init the Communicator.

        Args:
            client_config: client configuration data
            secure_train: True/False to indicate if secure train
            client_state_processors: Client state processor filters
            compression: communicate compression algorithm
        """
        self.cell = cell
        self.client_config = client_config
        self.secure_train = secure_train

        self.verbose = False
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
        self.token_verifier = None
        self.abort_signal = Signal()
        self.engine = None
        self.last_task_id = None  # ID of the last task received
        self.pending_task = None  # the task currently being processed
        self.logger = get_obj_logger(self)
        self._state_lock = threading.Lock()
        tmp_ctx = FLContext()
        tmp_ctx.set_prop(
            key=ReservedKey.IDENTITY_NAME,
            value=client_config["client_name"],
            private=False,
            sticky=True,
        )
        self._peer_ctx = tmp_ctx

    """
    To call set_add_auth_headers_filters, both cell and token must be available.
    The set_cell is called when cell becomes available, set_auth is called when token becomes available.
    In CP, set_cell happens before set_auth, hence we call set_add_auth_headers_filters in set_auth for CP.
    In CJ, set_auth happens before set_cell, hence we call set_add_auth_headers_filters in set_cell for CJ.
    """

    def set_auth(self, client_name, token, token_signature, ssid):
        self.ssid = ssid
        self.token_signature = token_signature
        self.token = token
        self.client_name = client_name

        if self.cell:
            # for CP
            set_add_auth_headers_filters(self.cell, client_name, token, token_signature, ssid)

        # put auth properties in data bus so that they can be used elsewhere
        set_scope_property(scope_name=client_name, key=FLMetaKey.AUTH_TOKEN, value=token)
        set_scope_property(scope_name=client_name, key=FLMetaKey.AUTH_TOKEN_SIGNATURE, value=token_signature)

    def set_cell(self, cell):
        self.cell = cell
        if self.token:
            # for CJ
            set_add_auth_headers_filters(self.cell, self.client_name, self.token, self.token_signature, self.ssid)

        # set CB to receive task messages from children
        cell.register_request_cb(
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.GET_TASK,
            cb=self._process_get_task,
        )

        cell.register_request_cb(
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.SUBMIT_UPDATE,
            cb=self._process_submit_result,
        )

    @staticmethod
    def _make_try_again():
        shareable = Shareable()
        shareable.set_header(key=FLContextKey.TASK_ID, value="")
        shareable.set_header(key=ServerCommandKey.TASK_NAME, value=SpecialTaskName.TRY_AGAIN)
        return shareable

    def _process_get_task(self, request: CellMessage):
        req = request.payload
        origin = request.get_header(MessageHeaderKey.ORIGIN)
        if not isinstance(req, Shareable):
            self.logger.error(f"Bad get_task request from {origin}")

        # note: the self.pending_task is unset by "submit_update", which could happen at any time.
        # we first assign self.pending_task to a different var (pending_task) and use this var in our processing.
        pending_task = self.pending_task
        pending_task_id = None
        if not self.engine or not pending_task:
            task = self._make_try_again()
        else:
            assert isinstance(pending_task, Shareable)
            last_task_id = req.get_header(ServerCommandKey.LAST_TASK_ID)
            task_id = pending_task.get_header(FLContextKey.TASK_ID)
            if last_task_id == task_id:
                self.logger.debug(f"same task request from {origin=}: {last_task_id=} - ask it to try again")
                task = self._make_try_again()
            elif not pending_task.get_header(ReservedKey.TASK_IS_READY):
                self.logger.debug(f"task {task_id} not ready - ask it to try again")
                task = self._make_try_again()
            else:
                # we'll send the pending task to the child.
                # make a copy of the task - only headers are copied!
                task = make_copy(pending_task, exclude_headers=[ReservedKey.TASK_IS_READY])
                pending_task_id = task_id

        if self.engine:
            if pending_task_id:
                # fire event to notify others that the pending task is sent to a child client
                with self.engine.new_context() as fl_ctx:
                    requesting_client_ctx = req.get_peer_context()
                    fl_ctx.set_peer_context(requesting_client_ctx)
                    fl_ctx.set_prop(FLContextKey.TASK_ID, pending_task_id, private=True, sticky=False)
                    self.engine.fire_event(EventType.TASK_ASSIGNMENT_SENT, fl_ctx)
                    is_processed = fl_ctx.get_prop(FLContextKey.EVENT_PROCESSED)
                    if not is_processed:
                        # no one listened or processed this event
                        self.logger.warning(
                            f"event {EventType.TASK_ASSIGNMENT_SENT} for task {pending_task_id} is not processed"
                        )

        task.set_peer_context(self._peer_ctx)
        return new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, task)

    def _process_submit_result(self, request: CellMessage):
        if not self.engine:
            # this could happen only when we crashed after task was pulled and restarted
            # since we don't have CJ restart capability this is impossible currently.
            self.logger.error("received submit_result while no engine")
            return new_cell_message({}, Shareable())

        with self.engine.new_context() as fl_ctx:
            assert isinstance(fl_ctx, FLContext)
            result = request.payload
            assert isinstance(result, Shareable)
            peer_ctx = result.get_peer_context()
            if peer_ctx:
                fl_ctx.set_peer_context(peer_ctx)

                # we also need to set peer_props since some app code expects it.
                result.set_peer_props(peer_ctx.get_all_public_props())

            fl_ctx.set_prop(
                key=FLContextKey.TASK_RESULT,
                value=result,
                private=True,
                sticky=False,
            )
            self.engine.fire_event(EventType.TASK_RESULT_RECEIVED, fl_ctx)
            is_processed = fl_ctx.get_prop(FLContextKey.EVENT_PROCESSED)
            if not is_processed:
                # no one listened or processed this event
                task_id = result.get_header(ReservedKey.TASK_ID)
                self.logger.warning(f"event {EventType.TASK_RESULT_RECEIVED} for task {task_id} is not processed")

        return new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, Shareable())

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

        shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
        private_key_file = None
        root_cert_file = None
        cert_file = None

        secure_mode = fl_ctx.get_prop(FLContextKey.SECURE_MODE, False)
        expected_host = None

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
                expected_host = get_scope_property(scope_name=client_name, key=FLContextKey.SERVER_HOST_NAME)

            if not expected_host:
                raise RuntimeError("cannot determine expected_host")

            client_config = fl_ctx.get_prop(FLContextKey.CLIENT_CONFIG)
            if not client_config:
                raise RuntimeError(f"missing {FLContextKey.CLIENT_CONFIG} in FL Context")
            private_key_file = client_config.get(SecureTrainConst.PRIVATE_KEY)
            cert_file = client_config.get(SecureTrainConst.SSL_CERT)
            root_cert_file = client_config.get(SecureTrainConst.SSL_ROOT_CERT)

        authenticator = Authenticator(
            cell=self.cell,
            project_name=project_name,
            client_name=client_name,
            client_type=ClientType.REGULAR,
            expected_sp_identity=expected_host,
            secure_mode=secure_mode,
            root_cert_file=root_cert_file,
            private_key_file=private_key_file,
            cert_file=cert_file,
            msg_timeout=self.maint_msg_timeout,
            retry_interval=self.client_register_interval,
        )

        token, signature, ssid, token_verifier = authenticator.authenticate(shared_fl_ctx, self.abort_signal)
        self.token_verifier = token_verifier
        self.set_auth(client_name, token, signature, ssid)
        return token, signature, ssid

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
        if not self.engine:
            self.engine = fl_ctx.get_engine()
            self._peer_ctx = gen_new_peer_ctx(fl_ctx)

        start_time = time.time()
        shareable = Shareable()
        shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
        shareable.set_peer_context(shared_fl_ctx)
        if self.last_task_id:
            shareable.set_header(ServerCommandKey.LAST_TASK_ID, self.last_task_id)

        task_message = new_cell_message(
            {
                CellMessageHeaderKeys.PROJECT_NAME: project_name,
            },
            shareable,
        )
        job_id = fl_ctx.get_job_id()

        if not timeout:
            timeout = self.timeout

        parent_fqcn = determine_parent_fqcn(self.client_config, fl_ctx)
        self.logger.debug(f"pulling task from parent FQCN: {parent_fqcn}")

        fqcn = FQCN.join([parent_fqcn, job_id])
        task = self.cell.send_request(
            target=fqcn,
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.GET_TASK,
            request=task_message,
            timeout=timeout,
            optional=True,
            abort_signal=fl_ctx.get_run_abort_signal(),
        )
        end_time = time.time()
        return_code = task.get_header(MessageHeaderKey.RETURN_CODE)

        if return_code == ReturnCode.OK:
            size = task.get_header(MessageHeaderKey.PAYLOAD_LEN)
            task_data = task.payload
            if not isinstance(task_data, Shareable):
                self.logger.error(f"bad task from {parent_fqcn}: expect Shareable but got {type(task_data)}")

            task_name = task_data.get_header(ServerCommandKey.TASK_NAME)
            self.logger.debug(f"received task from parent {parent_fqcn}: {task_name=}")
            fl_ctx.set_prop(FLContextKey.SSID, ssid, sticky=False)
            if task_name not in [SpecialTaskName.END_RUN, SpecialTaskName.TRY_AGAIN]:
                self.logger.info(
                    f"Received from {parent_fqcn}. getTask: {task_name} size: {format_size(size)} "
                    f"({size} Bytes) time: {end_time - start_time:.6f} seconds"
                )
                self.last_task_id = task_data.get_header(FLContextKey.TASK_ID)
                self.pending_task = task_data
        elif return_code == ReturnCode.AUTHENTICATION_ERROR:
            self.logger.warning("get_task request authentication failed.")
            return None
        else:
            task = None
            self.logger.warning(f"Failed to get_task from {parent_fqcn}. Will try it again.")

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
        # Set the pending_task to None immediately to reduce the chance that we send this task to a child
        # while we are still processing.
        self.pending_task = None

        start_time = time.time()
        shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
        shareable.set_peer_context(shared_fl_ctx)

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
        job_id = fl_ctx.get_job_id()

        if not timeout:
            timeout = self.timeout

        parent_fqcn = determine_parent_fqcn(self.client_config, fl_ctx)
        self.logger.debug(f"submitting update to parent FQCN: {parent_fqcn}")

        fqcn = FQCN.join([parent_fqcn, job_id])
        result = self.cell.send_request(
            target=fqcn,
            channel=CellChannel.SERVER_COMMAND,
            topic=ServerCommandNames.SUBMIT_UPDATE,
            request=task_message,
            timeout=timeout,
            optional=optional,
            abort_signal=fl_ctx.get_run_abort_signal(),
        )
        end_time = time.time()
        return_code = result.get_header(MessageHeaderKey.RETURN_CODE)
        size = task_message.get_header(MessageHeaderKey.PAYLOAD_LEN)
        self.logger.info(
            f"SubmitUpdate to: {parent_fqcn}. size: {format_size(size)} ({size} Bytes). time: {end_time - start_time:.6f} seconds"
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
        self.abort_signal.trigger(True)
        shared_fl_ctx = gen_new_peer_ctx(fl_ctx)
        shareable = Shareable()
        shareable.set_peer_context(shared_fl_ctx)
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
                shareable.set_peer_context(shared_fl_ctx)

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
