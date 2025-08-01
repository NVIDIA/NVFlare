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

from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import nvflare.fuel.f3.streaming.file_downloader as downloader
from nvflare.apis.fl_constant import ConnectionSecurity, FLContextKey, ProcessType, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.streaming import ConsumerFactory, ObjectProducer, StreamableEngine, StreamContext
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.app_common.streamers.file_streamer import FileStreamer
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.client.event import EventContext, EventHandler, EventPropKey, EventType
from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import (
    ConfirmMethod,
    InternalCommands,
    MetaKey,
    ProtoKey,
    ReplyKeyword,
    StreamChannel,
    StreamTopic,
    make_error,
    validate_proto,
)
from nvflare.fuel.hci.reg import CommandEntry, CommandModule, CommandRegister
from nvflare.fuel.hci.table import Table
from nvflare.fuel.sec.authn import set_add_auth_headers_filters
from nvflare.fuel.utils.admin_name_utils import new_admin_client_name
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.aux_runner import AuxMsgTarget, AuxRunner
from nvflare.private.defs import ClientType
from nvflare.private.fed.authenticator import Authenticator, validate_auth_headers
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, TokenVerifier, get_cn_from_cert, load_cert_file
from nvflare.private.stream_runner import HeaderKey, ObjectStreamer
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .api_spec import (
    AdminAPISpec,
    AdminConfigKey,
    CommandContext,
    CommandCtxKey,
    CommandInfo,
    ReplyProcessor,
    UidSource,
)
from .api_status import APIStatus

_CMD_TYPE_UNKNOWN = 0
_CMD_TYPE_CLIENT = 1
_CMD_TYPE_SERVER = 2

MAX_AUTO_LOGIN_TRIES = 300
AUTO_LOGIN_INTERVAL = 1.5


class FileWaiter(threading.Event):

    def __init__(self, tx_id):
        super().__init__()
        self.tx_id = tx_id
        self.stream_ctx = None
        self.last_progress_time = time.time()

    def get_stream_ctx(self):
        return self.stream_ctx


class ResultKey(object):

    STATUS = ProtoKey.STATUS
    DETAILS = ProtoKey.DETAILS
    META = ProtoKey.META


class _ServerReplyJsonProcessor(object):
    def __init__(self, ctx: CommandContext):
        if not isinstance(ctx, CommandContext):
            raise TypeError(f"ctx is not an instance of CommandContext. but get {type(ctx)}")
        self.ctx = ctx

    def process_server_reply(self, resp):
        """Process the server reply and store the status/details into API's `command_result`
        NOTE: this func is used for receive_and_process(), which is defined by conn!
        This method does not tale CommandContext!

        Args:
            resp: The raw response that returns by the server.
        """
        api = self.ctx.get_api()
        api.debug("Server Reply: {}".format(resp))

        ctx = self.ctx

        # this resp is what is usually directly used to return, straight from server
        ctx.set_command_result(resp)
        reply_processor = ctx.get_reply_processor()
        if reply_processor is None:
            reply_processor = _DefaultReplyProcessor()

        reply_processor.reply_start(ctx, resp)

        if resp is not None:
            data = resp[ProtoKey.DATA]
            for item in data:
                it = item[ProtoKey.TYPE]
                if it == ProtoKey.STRING:
                    reply_processor.process_string(ctx, item[ProtoKey.DATA])
                elif it == ProtoKey.SUCCESS:
                    reply_processor.process_success(ctx, item[ProtoKey.DATA])
                elif it == ProtoKey.ERROR:
                    reply_processor.process_error(ctx, item[ProtoKey.DATA])
                    break
                elif it == ProtoKey.TABLE:
                    table = Table(None)
                    table.set_rows(item[ProtoKey.ROWS])
                    reply_processor.process_table(ctx, table)
                elif it == ProtoKey.DICT:
                    reply_processor.process_dict(ctx, item[ProtoKey.DATA])
                elif it == ProtoKey.TOKEN:
                    reply_processor.process_token(ctx, item[ProtoKey.DATA])
                elif it == ProtoKey.SHUTDOWN:
                    reply_processor.process_shutdown(ctx, item[ProtoKey.DATA])
                    break
                else:
                    reply_processor.protocol_error(ctx, "Invalid item type: " + it)
                    break
            meta = resp.get(ProtoKey.META)
            if meta:
                ctx.set_meta(meta)
        else:
            reply_processor.protocol_error(ctx, "Protocol Error")

        reply_processor.reply_done(ctx)


class _DefaultReplyProcessor(ReplyProcessor):
    def process_shutdown(self, ctx: CommandContext, msg: str):
        api = ctx.get_prop(CommandCtxKey.API)
        api.shutdown_received = True
        api.shutdown_msg = msg


class _LoginReplyProcessor(ReplyProcessor):
    """Reply processor for handling login and setting the token for the admin client."""

    def process_string(self, ctx: CommandContext, item: str):
        api = ctx.get_api()
        api.login_result = item

    def process_token(self, ctx: CommandContext, token: str):
        api = ctx.get_api()
        api.token = token


class _CmdListReplyProcessor(ReplyProcessor):
    """Reply processor to register available commands after getting back a table of commands from the server."""

    def process_table(self, ctx: CommandContext, table: Table):
        api = ctx.get_api()
        for i in range(len(table.rows)):
            if i == 0:
                # this is header
                continue

            row = table.rows[i]
            if len(row) < 5:
                return

            scope = row[0]
            cmd_name = row[1]
            desc = row[2]
            usage = row[3]
            confirm = row[4]
            client_cmd = None
            visible = True
            if len(row) > 5:
                client_cmd = row[5]
            if len(row) > 6:
                visible = row[6].lower() in ["true", "yes"]

            # if confirm == 'auth' and not client.require_login:
            # the user is not authenticated - skip this command
            # continue
            api.server_cmd_reg.add_command(
                scope_name=scope,
                cmd_name=cmd_name,
                desc=desc,
                usage=usage,
                handler=None,
                authz_func=None,
                visible=visible,
                confirm=confirm,
                client_cmd=client_cmd,
                map_client_cmd=True,
            )

        api.server_cmd_received = True


class AdminAPI(AdminAPISpec, StreamableEngine):
    def __init__(
        self,
        user_name: str,
        admin_config: dict,
        cmd_modules: Optional[List] = None,
        debug: bool = False,
        auto_login_max_tries: int = 15,
        event_handlers=None,
    ):
        """API to keep certs, keys and connection information and to execute admin commands through do_command.

        Args:
            cmd_modules: command modules to load and register. Note that FileTransferModule is initialized here with upload_dir and download_dir if cmd_modules is None.
            user_name: Username to authenticate with FL server
            debug: Whether to print debug messages, which can help with diagnosing problems. False by default.
            auto_login_max_tries: maximum number of tries to auto-login.
        """
        super().__init__()
        if cmd_modules is None:
            from .file_transfer import FileTransferModule

            upload_dir = admin_config.get(AdminConfigKey.UPLOAD_DIR, "transfer")
            download_dir = admin_config.get(AdminConfigKey.DOWNLOAD_DIR, "transfer")
            cmd_modules = [FileTransferModule(upload_dir=upload_dir, download_dir=download_dir)]
        elif not isinstance(cmd_modules, list):
            raise TypeError("cmd_modules must be a list, but got {}".format(type(cmd_modules)))
        else:
            for m in cmd_modules:
                if not isinstance(m, CommandModule):
                    raise TypeError(
                        "cmd_modules must be a list of CommandModule, but got element of type {}".format(type(m))
                    )

        if not event_handlers:
            event_handlers = []

        if event_handlers:
            if not isinstance(event_handlers, list):
                raise TypeError(f"event_handlers must be a list but got {type(event_handlers)}")
            for h in event_handlers:
                if not isinstance(h, EventHandler):
                    raise TypeError(f"item in event_handlers must be EventHandler but got {type(h)}")

        for m in cmd_modules:
            if isinstance(m, EventHandler):
                event_handlers.append(m)

        self.logger = get_obj_logger(self)
        self.conn_sec = admin_config.get(AdminConfigKey.CONNECTION_SECURITY)
        self.project_name = admin_config.get(AdminConfigKey.PROJECT_NAME)
        self.server_identity = admin_config.get(AdminConfigKey.SERVER_IDENTITY, "server")
        self.scheme = admin_config.get(AdminConfigKey.CONNECTION_SCHEME, "grpc")
        self.ca_cert = admin_config.get(AdminConfigKey.CA_CERT)
        self.client_cert = admin_config.get(AdminConfigKey.CLIENT_CERT)
        self.client_key = admin_config.get(AdminConfigKey.CLIENT_KEY)
        self.uid_source = admin_config.get(AdminConfigKey.UID_SOURCE, UidSource.USER_INPUT)
        self.host = admin_config.get(AdminConfigKey.HOST, "localhost")
        self.port = admin_config.get(AdminConfigKey.PORT, 8002)
        self.default_login_timeout = admin_config.get(AdminConfigKey.LOGIN_TIMEOUT, 10.0)
        self.file_download_progress_timeout = admin_config.get(AdminConfigKey.FILE_DOWNLOAD_PROGRESS_TIMEOUT, 5.0)
        self.authenticate_msg_timeout = admin_config.get(AdminConfigKey.AUTHENTICATE_MSG_TIMEOUT, 5.0)
        self.user_name = user_name
        self.event_handlers = event_handlers

        if not self.ca_cert:
            raise ConfigError("missing CA Cert file name")
        if not self.client_cert:
            raise ConfigError("missing Client Cert file name")
        if not self.client_key:
            raise ConfigError("missing Client Key file name")

        if self.uid_source == UidSource.CERT:
            # We'll find the username from the client cert
            cert = load_cert_file(self.client_cert)
            self.user_name = get_cn_from_cert(cert)

        if not self.user_name:
            raise Exception("user_name is required.")

        if debug:
            self._debug = debug
        else:
            self._debug = admin_config.get(AdminConfigKey.WITH_DEBUG, False)

        self.cmd_timeout = None

        # for login
        self.token = None
        self.login_result = None

        self.server_cmd_reg = CommandRegister(app_ctx=self)
        self.client_cmd_reg = CommandRegister(app_ctx=self)
        self.server_cmd_received = False

        self.all_cmds = []
        self.cmd_modules = cmd_modules

        # for shutdown
        self.shutdown_received = False
        self.shutdown_msg = None

        self.server_sess_active = False
        self.shutdown_asked = False

        self.sess_monitor_thread = None
        self.sess_monitor_active = False

        # create the FSM for session monitoring
        if auto_login_max_tries < 0 or auto_login_max_tries > MAX_AUTO_LOGIN_TRIES:
            raise ValueError(f"auto_login_max_tries is out of range: [0, {MAX_AUTO_LOGIN_TRIES}]")
        self.auto_login_max_tries = auto_login_max_tries

        self.closed = False
        self.in_logout = False
        self.cell = None
        self.aux_runner = None
        self.object_streamer = None
        self.fl_ctx_mgr = FLContextManager(
            engine=self,
            identity_name=self.user_name,
            private_stickers={FLContextKey.PROCESS_TYPE: ProcessType.CLIENT_PARENT},
        )
        self.file_download_waiters = {}  # tx_id => Threading.Event

    def new_context(self):
        return self.fl_ctx_mgr.new_context()

    def connect(self, timeout=None):
        if timeout is not None:
            # validate provided timeout value
            if not isinstance(timeout, (int, float)):
                raise ValueError(f"timeout must be a number but got {type(timeout)}")

            if timeout <= 0:
                raise ValueError(f"timeout must be a number > 0 but got {timeout}")
        else:
            # use value configured in admin config
            timeout = self.default_login_timeout

        print("Connecting to FLARE ...")
        if self.cell:
            return

        my_fqcn = new_admin_client_name()
        credentials = {
            DriverParams.CA_CERT.value: self.ca_cert,
            DriverParams.CLIENT_CERT.value: self.client_cert,
            DriverParams.CLIENT_KEY.value: self.client_key,
        }

        root_url = f"{self.scheme}://{self.host}:{self.port}"
        secure_conn = True
        if self.conn_sec:
            conn_sec = self.conn_sec.lower()
            credentials[DriverParams.CONNECTION_SECURITY.value] = conn_sec
            if conn_sec == ConnectionSecurity.CLEAR:
                secure_conn = False

        flare_decomposers.register()

        self.debug(f"Creating cell: {my_fqcn=} {root_url=} {secure_conn=} {credentials=}")

        self.cell = Cell(
            fqcn=my_fqcn,
            root_url=root_url,
            secure=secure_conn,
            credentials=credentials,
            create_internal_listener=False,
            parent_url=None,
        )

        self.cell.register_request_cb(
            channel=CellChannel.HCI,
            topic="SESSION_EXPIRED",
            cb=self._handle_session_expired,
        )

        NetAgent(self.cell)
        self.cell.start()

        # authenticate
        authenticator = Authenticator(
            cell=self.cell,
            project_name=self.project_name,
            client_name=self.user_name,
            client_type=ClientType.ADMIN,
            expected_sp_identity=self.server_identity,
            secure_mode=True,  # always True to authenticate the cell endpoint!
            root_cert_file=self.ca_cert,
            private_key_file=self.client_key,
            cert_file=self.client_cert,
            msg_timeout=self.authenticate_msg_timeout,
            retry_interval=1.0,
            timeout=timeout,
        )

        abort_signal = Signal()
        shared_fl_ctx = FLContext()
        shared_fl_ctx.set_public_props({ReservedKey.IDENTITY_NAME: self.user_name})
        token, token_signature, ssid, token_verifier = authenticator.authenticate(
            shared_fl_ctx=shared_fl_ctx,
            abort_signal=abort_signal,
        )

        if not isinstance(token_verifier, TokenVerifier):
            raise RuntimeError(f"expect token_verifier to be TokenVerifier but got {type(token_verifier)}")

        set_add_auth_headers_filters(self.cell, self.user_name, token, token_signature, ssid)

        self.cell.core_cell.add_incoming_filter(
            channel="*",
            topic="*",
            cb=validate_auth_headers,
            token_verifier=token_verifier,
            logger=self.logger,
        )
        self.debug(f"Successfully authenticated to {self.server_identity}: {token=} {ssid=}")

        self.aux_runner = AuxRunner(self)
        self.object_streamer = ObjectStreamer(self.aux_runner)

        self.cell.register_request_cb(
            channel=CellChannel.AUX_COMMUNICATION,
            topic="*",
            cb=self._handle_aux_message,
        )

    def _handle_aux_message(self, request: CellMessage) -> CellMessage:
        assert isinstance(request, CellMessage), "request must be CellMessage but got {}".format(type(request))
        data = request.payload

        topic = request.get_header(MessageHeaderKey.TOPIC)
        with self.new_context() as fl_ctx:
            reply = self.aux_runner.dispatch(topic=topic, request=data, fl_ctx=fl_ctx)

            if reply is not None:
                return_message = CellMessage({}, reply)
                return_message.set_header(MessageHeaderKey.RETURN_CODE, CellReturnCode.OK)
            else:
                return_message = CellMessage({}, None)
            return return_message

    def download_file(self, source_fqcn: str, ref_id: str, file_name: str):
        err, file_path = downloader.download_file(
            cell=self.cell,
            ref_id=ref_id,
            from_fqcn=source_fqcn,
            per_request_timeout=self.file_download_progress_timeout,
        )
        if err:
            print(f"failed to receive file {file_name}: {err}")
            return None

        file_stats = os.stat(file_path)
        num_bytes_received = file_stats.st_size
        Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
        shutil.move(file_path, file_name)
        return num_bytes_received

    def get_cell(self):
        return self.cell

    def _handle_session_expired(self, message: CellMessage):
        self.debug("received session timeout from server")
        self.close()
        self.fire_session_event(EventType.SESSION_TIMEOUT, message.payload)

    def debug(self, msg):
        if self._debug:
            print(f"DEBUG: {msg}")

    def fire_event(self, event_type: str, ctx: EventContext):
        self.debug(f"firing event {event_type}")
        if self.event_handlers:
            for h in self.event_handlers:
                h.handle_event(event_type, ctx)

    def set_command_timeout(self, timeout: float):
        if not isinstance(timeout, (int, float)):
            raise TypeError(f"timeout must be a number but got {type(timeout)}")
        timeout = float(timeout)
        if timeout <= 0.0:
            raise ValueError(f"invalid timeout value {timeout} - must be > 0.0")

        self.cmd_timeout = timeout

    def unset_command_timeout(self):
        self.cmd_timeout = None

    def _new_event_context(self):
        ctx = EventContext()
        ctx.set_prop(EventPropKey.USER_NAME, self.user_name)
        ctx.set_prop(EventPropKey.API, self)
        return ctx

    def fire_session_event(self, event_type: str, msg: str = ""):
        ctx = self._new_event_context()
        if msg:
            ctx.set_prop(EventPropKey.MSG, msg)
        self.fire_event(event_type, ctx)

    def _try_login(self):
        resp = None
        for i in range(self.auto_login_max_tries):
            try:
                self.fire_session_event(EventType.TRYING_LOGIN, "Trying to login, please wait ...")
            except Exception as ex:
                print(f"exception handling event {EventType.TRYING_LOGIN}: {secure_format_exception(ex)}")
                return {
                    ResultKey.STATUS: APIStatus.ERROR_RUNTIME,
                    ResultKey.DETAILS: f"exception handling event {EventType.TRYING_LOGIN}",
                }

            resp = self._user_login()

            status = resp.get(ResultKey.STATUS)
            if status in [APIStatus.SUCCESS, APIStatus.ERROR_AUTHENTICATION, APIStatus.ERROR_CERT]:
                if status == APIStatus.SUCCESS:
                    self.fire_session_event(EventType.LOGIN_SUCCESS)
                else:
                    self.fire_session_event(EventType.LOGIN_FAILURE)
                return resp
            time.sleep(AUTO_LOGIN_INTERVAL)
        if resp is None:
            resp = {
                ResultKey.STATUS: APIStatus.ERROR_RUNTIME,
                ResultKey.DETAILS: f"Auto login failed after {self.auto_login_max_tries} tries",
            }
            self.fire_session_event(EventType.LOGIN_FAILURE)
        return resp

    def login(self):
        try:
            self.fire_session_event(EventType.BEFORE_LOGIN)
            result = self._try_login()
            self.debug(f"login result is {result}")
        except Exception as e:
            result = {
                ResultKey.STATUS: APIStatus.ERROR_RUNTIME,
                ResultKey.DETAILS: f"Exception occurred ({secure_format_exception(e)}) when trying to login - please try later",
            }
        return result

    def _load_client_cmds_from_modules(self, cmd_modules):
        if cmd_modules:
            for m in cmd_modules:
                self.client_cmd_reg.register_module(m, include_invisible=True)

    def _load_client_cmds_from_module_specs(self, cmd_module_specs):
        if cmd_module_specs:
            for m in cmd_module_specs:
                self.client_cmd_reg.register_module_spec(m, include_invisible=True)

    def register_command(self, cmd_entry):
        self.all_cmds.append(cmd_entry.name)

    def logout(self):
        """Send logout command to server."""
        if self.in_logout:
            return None

        self.in_logout = True
        try:
            resp = self.server_execute(InternalCommands.LOGOUT)
        finally:
            # make sure to close
            self.close()
        return resp

    def close(self):
        # this method can be called multiple times
        if self.closed:
            return

        self.closed = True
        self.server_sess_active = False
        self.shutdown_asked = True
        self.shutdown_streamer()
        if self.cell:
            self.cell.stop()

    def _get_command_list_from_server(self) -> bool:
        self.server_cmd_received = False
        self.server_execute(InternalCommands.GET_CMD_LIST, _CmdListReplyProcessor())
        self.server_cmd_reg.finalize(self.register_command)
        if not self.server_cmd_received:
            return False
        return True

    def _after_login(self) -> dict:
        result = self._get_command_list_from_server()
        if not result:
            return {
                ResultKey.STATUS: APIStatus.ERROR_RUNTIME,
                ResultKey.DETAILS: "Can't fetch command list from server.",
            }

        # prepare client modules
        # we may have additional dynamically created cmd modules based on server commands
        extra_module_specs = []
        if self.server_cmd_reg.mapped_cmds:
            for c in self.server_cmd_reg.mapped_cmds:
                for m in self.cmd_modules:
                    new_module_spec = m.generate_module_spec(c)
                    if new_module_spec is not None:
                        extra_module_specs.append(new_module_spec)

        self._load_client_cmds_from_modules(self.cmd_modules)
        if extra_module_specs:
            self._load_client_cmds_from_module_specs(extra_module_specs)
        self.client_cmd_reg.finalize(self.register_command)
        self.server_sess_active = True
        return {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.DETAILS: "Login success"}

    def is_ready(self) -> bool:
        """Whether the API is ready for executing commands."""
        return self.server_sess_active

    def _user_login(self):
        """Login user

        Returns:
            A dict of login status and details
        """
        command = f"{InternalCommands.CERT_LOGIN} {self.user_name}"

        id_asserter = IdentityAsserter(private_key_file=self.client_key, cert_file=self.client_cert)
        cn_signature = id_asserter.sign_common_name(nonce="")

        headers = {
            "user_name": self.user_name,
            "cert": id_asserter.cert_data,
            "signature": cn_signature,
        }

        self.login_result = None
        self.server_execute(command, _LoginReplyProcessor(), headers=headers)
        if self.login_result is None:
            return {
                ResultKey.STATUS: APIStatus.ERROR_RUNTIME,
                ResultKey.DETAILS: "Communication Error - please try later",
            }
        elif self.login_result == "REJECT":
            return {
                ResultKey.STATUS: APIStatus.ERROR_AUTHENTICATION,
                ResultKey.DETAILS: "Incorrect user name or password",
            }
        return self._after_login()

    def _send_to_cell(self, ctx: CommandContext):
        command = ctx.get_command()
        json_processor = ctx.get_json_processor()
        process_json_func = json_processor.process_server_reply

        conn = Connection()
        conn.append_command(command)
        if self.token:
            conn.append_token(self.token)

        if self.cmd_timeout:
            conn.update_meta({MetaKey.CMD_TIMEOUT: self.cmd_timeout})

        custom_props = ctx.get_custom_props()
        if custom_props:
            conn.update_meta({MetaKey.CUSTOM_PROPS: custom_props})

        cmd_props = ctx.get_command_props()
        if cmd_props:
            conn.update_meta({MetaKey.CMD_PROPS: cmd_props})

        timeout = self.cmd_timeout
        if not timeout:
            timeout = 5.0

        requester = ctx.get_requester()
        if requester:
            try:
                reply = requester.send_request(self, conn, ctx)
            except:
                traceback.print_exc()
                process_json_func(make_error(f"{type(requester)} failed to send request to Admin Server"))
                return
        else:
            request = CellMessage(payload=conn.close(), headers=ctx.get_command_headers())
            cell_reply = self.cell.send_request(
                channel=CellChannel.HCI,
                topic="command",
                target=FQCN.ROOT_SERVER,
                request=request,
                timeout=timeout,
            )
            reply = cell_reply.payload

        if reply:
            try:
                json_data = validate_proto(reply)
                process_json_func(json_data)
            except:
                traceback.print_exc()
                process_json_func(make_error(f"{ReplyKeyword.COMM_FAILURE} with Admin Server"))

    def _try_command(self, cmd_ctx: CommandContext):
        """Try to execute a command on server side.

        Args:
            cmd_ctx: The command to execute.
        """
        self.debug(f"sending command '{cmd_ctx.get_command()}'")

        json_processor = _ServerReplyJsonProcessor(cmd_ctx)
        process_json_func = json_processor.process_server_reply
        cmd_ctx.set_json_processor(json_processor)

        event_ctx = self._new_event_context()
        event_ctx.set_prop(EventPropKey.CMD_NAME, cmd_ctx.get_command_name())
        event_ctx.set_prop(EventPropKey.CMD_CTX, cmd_ctx)

        try:
            self.fire_event(EventType.BEFORE_EXECUTE_CMD, event_ctx)
        except Exception as ex:
            secure_log_traceback()
            process_json_func(
                make_error(f"exception handling event {EventType.BEFORE_EXECUTE_CMD}: {secure_format_exception(ex)}")
            )
            return

        # see whether any event handler has set "custom_props"
        custom_props = event_ctx.get_prop(EventPropKey.CUSTOM_PROPS)
        if custom_props:
            cmd_ctx.set_custom_props(custom_props)

        try:
            self._send_to_cell(cmd_ctx)
        except Exception as e:
            if self._debug:
                secure_log_traceback()
            traceback.print_exc()
            process_json_func(
                make_error(f"{ReplyKeyword.COMM_FAILURE} with Admin Server: {secure_format_exception(e)}")
            )

    def _get_command_detail(self, command):
        """Get command details

        Args:
          command (str): command

        Returns: tuple of (cmd_type, cmd_name, args, entries)
        """
        args = split_to_args(command)
        cmd_name = args[0]

        # check client side commands
        entries = self.client_cmd_reg.get_command_entries(cmd_name)
        if len(entries) > 0:
            return _CMD_TYPE_CLIENT, cmd_name, args, entries

        # check server side commands
        entries = self.server_cmd_reg.get_command_entries(cmd_name)
        if len(entries) > 0:
            return _CMD_TYPE_SERVER, cmd_name, args, entries

        return _CMD_TYPE_UNKNOWN, cmd_name, args, None

    def check_command(self, command: str) -> CommandInfo:
        """Checks the specified command for processing info

        Args:
            command: command to be checked

        Returns: command processing info

        """
        cmd_type, cmd_name, _, entries = self._get_command_detail(command)

        if cmd_type == _CMD_TYPE_UNKNOWN:
            return CommandInfo.UNKNOWN

        if len(entries) > 1:
            return CommandInfo.AMBIGUOUS

        ent = entries[0]
        assert isinstance(ent, CommandEntry)
        if ent.confirm == ConfirmMethod.AUTH:
            return CommandInfo.CONFIRM_AUTH
        elif ent.confirm == ConfirmMethod.YESNO:
            return CommandInfo.CONFIRM_YN
        else:
            return CommandInfo.OK

    def _new_command_context(self, command, args, ent: CommandEntry):
        ctx = CommandContext()
        ctx.set_api(self)
        ctx.set_command(command)
        ctx.set_command_args(args)
        ctx.set_command_entry(ent)
        return ctx

    def _do_client_command(self, command, args, ent: CommandEntry):
        ctx = self._new_command_context(command, args, ent)
        return_result = ent.handler(args, ctx)
        result = ctx.get_command_result()
        if return_result:
            return return_result
        if result is None:
            return {ResultKey.STATUS: APIStatus.ERROR_RUNTIME, ResultKey.DETAILS: "Client did not respond"}
        return result

    def upload_file(self, file_name: str, conn: Connection):
        stream_ctx = {"conn_data": conn.close()}
        with self.new_context() as fl_ctx:
            rc, replies = FileStreamer.stream_file(
                channel=StreamChannel.UPLOAD,
                topic=StreamTopic.FOLDER,
                stream_ctx=stream_ctx,
                file_name=file_name,
                fl_ctx=fl_ctx,
                targets=[FQCN.ROOT_SERVER],  # to server
            )
            if rc != ReturnCode.OK:
                self.logger.error(f"failed to stream file to server: {rc}")
                return None
            reply = replies.get(FQCN.ROOT_SERVER)
            assert isinstance(reply, Shareable)
            end_result = reply.get_header(HeaderKey.END_RESULT)
            return end_result

    def do_command(self, command: str, props=None):
        """A convenient method to call commands using string.

        Args:
          command (str): command
          props: additional props

        Returns:
            Object containing status and details (or direct response from server, which originally was just time and data)
        """
        cmd_type, cmd_name, args, entries = self._get_command_detail(command)
        if cmd_type == _CMD_TYPE_UNKNOWN:
            return {
                ResultKey.STATUS: APIStatus.ERROR_SYNTAX,
                ResultKey.DETAILS: f"Command {cmd_name} not found",
            }

        if len(entries) > 1:
            return {
                ResultKey.STATUS: APIStatus.ERROR_SYNTAX,
                ResultKey.DETAILS: f"Ambiguous command {cmd_name} - qualify with scope",
            }

        ent = entries[0]
        if cmd_type == _CMD_TYPE_CLIENT:
            return self._do_client_command(command=command, args=args, ent=ent)

        # server command
        if not self.server_sess_active:
            return {
                ResultKey.STATUS: APIStatus.ERROR_INACTIVE_SESSION,
                ResultKey.DETAILS: "Session is inactive, please try later",
            }

        return self.server_execute(command, cmd_entry=ent, props=props)

    def server_execute(self, command, reply_processor=None, cmd_entry=None, cmd_ctx=None, props=None, headers=None):
        if self.in_logout and command != InternalCommands.LOGOUT:
            return {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.DETAILS: "session is logging out"}

        args = split_to_args(command)
        if cmd_ctx:
            ctx = cmd_ctx
        else:
            ctx = self._new_command_context(command, args, cmd_entry)
        ctx.set_command(command)

        if props:
            self.debug(f"server_execute: set cmd props to ctx {props}")
            ctx.set_command_props(props)

        if headers:
            self.debug(f"setting cmd headers: {headers}")
            ctx.set_command_headers(headers)

        start = time.time()
        ctx.set_reply_processor(reply_processor)
        self._try_command(ctx)
        secs = time.time() - start
        usecs = int(secs * 1000000)

        self.debug(f"server_execute Done [{usecs} usecs] {datetime.now()}")

        result = ctx.get_command_result()
        meta = ctx.get_meta()
        if result is None:
            return {ResultKey.STATUS: APIStatus.ERROR_SERVER_CONNECTION, ResultKey.DETAILS: "Server did not respond"}
        if meta:
            result[ResultKey.META] = meta

        if ResultKey.STATUS not in result:
            result[ResultKey.STATUS] = self._determine_api_status(result)
        return result

    def _determine_api_status(self, result):
        status = result.get(ResultKey.STATUS)
        if status:
            return status

        data = result.get(ProtoKey.DATA)
        if not data:
            return APIStatus.ERROR_RUNTIME

        reply_data_list = []
        for d in data:
            if isinstance(d, dict):
                t = d.get(ProtoKey.TYPE)
                if t == ProtoKey.SUCCESS:
                    return APIStatus.SUCCESS
                if t == ProtoKey.STRING or t == ProtoKey.ERROR:
                    reply_data_list.append(d[ProtoKey.DATA])
        reply_data_full_response = "\n".join(reply_data_list)
        if ReplyKeyword.SESSION_INACTIVE in reply_data_full_response:
            return APIStatus.ERROR_INACTIVE_SESSION
        if ReplyKeyword.WRONG_SERVER in reply_data_full_response:
            return APIStatus.ERROR_SERVER_CONNECTION
        if ReplyKeyword.COMM_FAILURE in reply_data_full_response:
            return APIStatus.ERROR_SERVER_CONNECTION
        if ReplyKeyword.INVALID_CLIENT in reply_data_full_response:
            return APIStatus.ERROR_INVALID_CLIENT
        if ReplyKeyword.UNKNOWN_SITE in reply_data_full_response:
            return APIStatus.ERROR_INVALID_CLIENT
        if ReplyKeyword.NOT_AUTHORIZED in reply_data_full_response:
            return APIStatus.ERROR_AUTHORIZATION
        return APIStatus.SUCCESS

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
        """Send a stream of Shareable objects to receivers.

        Args:
            channel: the channel for this stream
            topic: topic of the stream
            stream_ctx: context of the stream
            targets: receiving sites
            producer: the ObjectProducer that can produces the stream of Shareable objects
            fl_ctx: the FLContext object
            optional: whether the stream is optional
            secure: whether to use P2P security

        Returns: result from the generator's reply processing

        """
        assert isinstance(self.object_streamer, ObjectStreamer)
        return self.object_streamer.stream(
            channel=channel,
            topic=topic,
            stream_ctx=stream_ctx,
            producer=producer,
            fl_ctx=fl_ctx,
            optional=optional,
            secure=secure,
            targets=[AuxMsgTarget.server_target()],  # only stream to server!
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
        """Register a ConsumerFactory for specified app channel and topic.
        Once a new streaming request is received for the channel/topic, the registered factory will be used
        to create an ObjectConsumer object to handle the new stream.

        Note: the factory should generate a new ObjectConsumer every time get_consumer() is called. This is because
        multiple streaming sessions could be going on at the same time. Each streaming session should have its
        own ObjectConsumer.

        Args:
            channel: app channel
            topic: app topic
            factory: the factory to be registered
            stream_done_cb: the callback to be called when streaming is done on receiving side
            consumed_cb: the callback to be called after a chunk is processed

        Returns: None

        """
        assert isinstance(self.object_streamer, ObjectStreamer)
        self.object_streamer.register_stream_processing(
            channel=channel,
            topic=topic,
            factory=factory,
            stream_done_cb=stream_done_cb,
            consumed_cb=consumed_cb,
            **cb_kwargs,
        )

    def shutdown_streamer(self):
        """Shutdown the engine's streamer.

        Returns: None

        """
        if self.object_streamer:
            assert isinstance(self.object_streamer, ObjectStreamer)
            self.object_streamer.shutdown()
