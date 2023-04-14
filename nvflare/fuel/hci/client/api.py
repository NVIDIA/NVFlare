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

import socket
import ssl
import threading
import time
from datetime import datetime
from typing import List, Optional

from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.conn import Connection, receive_and_process
from nvflare.fuel.hci.proto import ConfirmMethod, InternalCommands, ProtoKey, make_error
from nvflare.fuel.hci.reg import CommandEntry, CommandModule, CommandRegister
from nvflare.fuel.hci.table import Table
from nvflare.fuel.utils.fsm import FSM, State
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .api_spec import (
    AdminAPISpec,
    ApiPocValue,
    CommandContext,
    CommandCtxKey,
    CommandInfo,
    ReplyProcessor,
    ServiceFinder,
)
from .api_status import APIStatus

_CMD_TYPE_UNKNOWN = 0
_CMD_TYPE_CLIENT = 1
_CMD_TYPE_SERVER = 2


class ResultKey(object):

    STATUS = "status"
    DETAILS = "details"
    META = "meta"


def session_event_cb_signature(event_type: str, info: str):
    """
    This defines the signature of session_event callback.
    When creating the AdminAPI object, you can provide a session event callback function.
    This function is called when a session event happens.

    Args:
        event_type: the event type
        info: information of the event

    Returns:

    """
    pass


class _ServerReplyJsonProcessor(object):
    def __init__(self, ctx: CommandContext):
        api = ctx.get_api()
        self.debug = api.debug
        self.ctx = ctx

    def process_server_reply(self, resp):
        """Process the server reply and store the status/details into API's `command_result`
        NOTE: this func is used for receive_and_process(), which is defined by conn!
        This method does not tale CommandContext!

        Args:
            resp: The raw response that returns by the server.
        """
        if self.debug:
            print("DEBUG: Server Reply: {}".format(resp))

        ctx = self.ctx
        assert isinstance(ctx, CommandContext)

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


class SessionEventType(object):

    WAIT_FOR_SERVER_ADDR = "wait_for_server_addr"
    SERVER_ADDR_OBTAINED = "server_addr_obtained"
    SESSION_CLOSED = "session_closed"  # close the current session
    LOGIN_SUCCESS = "login_success"  # logged in to server
    LOGIN_FAILURE = "login_failure"  # cannot login to server
    TRYING_LOGIN = "trying_login"  # still try to login
    SP_ADDR_CHANGED = "sp_addr_changed"  # service provider address changed
    SESSION_TIMEOUT = "session_timeout"  # server timed out current session


_STATE_NAME_WAIT_FOR_SERVER_ADDR = "wait_for_server_addr"
_STATE_NAME_LOGIN = "login"
_STATE_NAME_OPERATE = "operate"

_SESSION_LOGGING_OUT = "session is logging out"


class _WaitForServerAddress(State):
    def __init__(self, api):
        State.__init__(self, _STATE_NAME_WAIT_FOR_SERVER_ADDR)
        self.api = api

    def execute(self, **kwargs):
        api = self.api
        api.fire_session_event(SessionEventType.WAIT_FOR_SERVER_ADDR, "Trying to obtain server address")
        with api.new_addr_lock:
            if api.new_host and api.new_port and api.new_ssid:
                api.fire_session_event(
                    SessionEventType.SERVER_ADDR_OBTAINED, f"Obtained server address: {api.new_host}:{api.new_port}"
                )
                return _STATE_NAME_LOGIN
            else:
                # stay here
                return ""


class _TryLogin(State):
    def __init__(self, api):
        State.__init__(self, _STATE_NAME_LOGIN)
        self.api = api

    def enter(self):
        api = self.api
        api.server_sess_active = False

        # use lock here since the service finder (in another thread) could change the
        # address at this moment
        with api.new_addr_lock:
            new_host = api.new_host
            new_port = api.new_port
            new_ssid = api.new_ssid

        # set the address for login
        with api.addr_lock:
            api.host = new_host
            api.port = new_port
            api.ssid = new_ssid

    def execute(self, **kwargs):
        api = self.api
        result = api.auto_login()
        if result[ResultKey.STATUS] == APIStatus.SUCCESS:
            api.server_sess_active = True
            api.fire_session_event(
                SessionEventType.LOGIN_SUCCESS, f"Logged into server at {api.host}:{api.port} with SSID: {api.ssid}"
            )
            return _STATE_NAME_OPERATE

        details = result.get(ResultKey.DETAILS, "")
        if details != _SESSION_LOGGING_OUT:
            api.fire_session_event(SessionEventType.LOGIN_FAILURE, details)

        return FSM.STATE_NAME_EXIT


class _Operate(State):
    def __init__(self, api, sess_check_interval):
        State.__init__(self, _STATE_NAME_OPERATE)
        self.api = api
        self.last_sess_check_time = None
        self.sess_check_interval = sess_check_interval

    def enter(self):
        self.api.server_sess_active = True

    def execute(self, **kwargs):
        # check whether server addr has changed
        api = self.api
        with api.new_addr_lock:
            new_host = api.new_host
            new_port = api.new_port
            new_ssid = api.new_ssid

        with api.addr_lock:
            cur_host = api.host
            cur_port = api.port
            cur_ssid = api.ssid

        if new_host != cur_host or new_port != cur_port or cur_ssid != new_ssid:
            # need to relogin
            api.fire_session_event(SessionEventType.SP_ADDR_CHANGED, f"Server address changed to {new_host}:{new_port}")
            return _STATE_NAME_LOGIN

        # check server session status
        if not self.sess_check_interval:
            return ""

        if not self.last_sess_check_time or time.time() - self.last_sess_check_time >= self.sess_check_interval:
            self.last_sess_check_time = time.time()
            result = api.check_session_status_on_server()
            details = result.get(ResultKey.DETAILS, "")
            status = result[ResultKey.STATUS]
            if status in APIStatus.ERROR_INACTIVE_SESSION:
                if details != _SESSION_LOGGING_OUT:
                    api.fire_session_event(SessionEventType.SESSION_TIMEOUT, details)

                # end the session
                return FSM.STATE_NAME_EXIT

        return ""


class AdminAPI(AdminAPISpec):
    def __init__(
        self,
        user_name: str,
        service_finder: ServiceFinder,
        ca_cert: str = "",
        client_cert: str = "",
        client_key: str = "",
        upload_dir: str = "",
        download_dir: str = "",
        cmd_modules: Optional[List] = None,
        poc: bool = False,
        debug: bool = False,
        session_event_cb=None,
        session_timeout_interval=None,
        session_status_check_interval=None,
    ):
        """Underlying API to keep certs, keys and connection information and to execute admin commands through do_command.

        Args:
            ca_cert: path to CA Cert file, by default provisioned rootCA.pem
            client_cert: path to admin client Cert file, by default provisioned as client.crt
            client_key: path to admin client Key file, by default provisioned as client.key
            upload_dir: File transfer upload directory. Folders uploaded to the server to be deployed must be here. Folder must already exist and be accessible.
            download_dir: File transfer download directory. Can be same as upload_dir. Folder must already exist and be accessible.
            cmd_modules: command modules to load and register. Note that FileTransferModule is initialized here with upload_dir and download_dir if cmd_modules is None.
            service_finder: used to obtain the primary service provider to set the host and port of the active server
            user_name: Username to authenticate with FL server
            poc: Whether to enable poc mode for using the proof of concept example without secure communication.
            debug: Whether to print debug messages, which can help with diagnosing problems. False by default.
            session_event_cb: the session event callback
            session_timeout_interval: if specified, automatically close the session after inactive for this long
            session_status_check_interval: how often to check session status with server
        """
        super().__init__()
        if cmd_modules is None:
            from .file_transfer import FileTransferModule

            cmd_modules = [FileTransferModule(upload_dir=upload_dir, download_dir=download_dir)]
        elif not isinstance(cmd_modules, list):
            raise TypeError("cmd_modules must be a list, but got {}".format(type(cmd_modules)))
        else:
            for m in cmd_modules:
                if not isinstance(m, CommandModule):
                    raise TypeError(
                        "cmd_modules must be a list of CommandModule, but got element of type {}".format(type(m))
                    )
        if not isinstance(service_finder, ServiceFinder):
            raise TypeError("service_finder should be ServiceFinder but got {}".format(type(service_finder)))

        cmd_module = service_finder.get_command_module()
        if cmd_module:
            cmd_modules.append(cmd_module)

        self.service_finder = service_finder
        self.host = None
        self.port = None
        self.ssid = None
        self.addr_lock = threading.Lock()

        self.new_host = None
        self.new_port = None
        self.new_ssid = None
        self.new_addr_lock = threading.Lock()

        self.poc_key = None
        self.poc = poc
        if self.poc:
            self.poc_key = ApiPocValue.ADMIN
        else:
            if len(ca_cert) <= 0:
                raise Exception("missing CA Cert file name")
            self.ca_cert = ca_cert
            if len(client_cert) <= 0:
                raise Exception("missing Client Cert file name")
            self.client_cert = client_cert
            if len(client_key) <= 0:
                raise Exception("missing Client Key file name")
            self.client_key = client_key

            self.service_finder.set_secure_context(
                ca_cert_path=self.ca_cert, cert_path=self.client_cert, private_key_path=self.client_key
            )
        self.debug = debug

        # for login
        self.token = None
        self.login_result = None
        if not user_name:
            raise Exception("user_name is required.")
        self.user_name = user_name

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

        if session_event_cb is not None:
            assert callable(session_event_cb), "session_event_cb must be callable"
        self.session_event_cb = session_event_cb

        # create the FSM for session monitoring
        fsm = FSM("session monitor")
        fsm.add_state(_WaitForServerAddress(self))
        fsm.add_state(_TryLogin(self))
        fsm.add_state(_Operate(self, session_status_check_interval))
        self.fsm = fsm

        self.session_timeout_interval = session_timeout_interval
        self.last_sess_activity_time = time.time()

        self.closed = False
        self.in_logout = False
        self.service_finder.start(self._handle_sp_address_change)
        self._start_session_monitor()

    def fire_session_event(self, event_type: str, msg: str):
        if self.session_event_cb is not None:
            self.session_event_cb(event_type, msg)

    def _handle_sp_address_change(self, host: str, port: int, ssid: str):
        with self.addr_lock:
            if host == self.host and port == self.port and ssid == self.ssid:
                # no change
                return

        with self.new_addr_lock:
            self.new_host = host
            self.new_port = port
            self.new_ssid = ssid

    def _try_auto_login(self):
        resp = {}
        for i in range(5):
            self.fire_session_event(SessionEventType.TRYING_LOGIN, "Trying to login, please wait ...")

            if self.poc:
                resp = self.login_with_poc(username=self.user_name, poc_key=self.poc_key)
            else:
                resp = self.login(username=self.user_name)
            if resp[ResultKey.STATUS] in [APIStatus.SUCCESS, APIStatus.ERROR_AUTHENTICATION, APIStatus.ERROR_CERT]:
                return resp
            time.sleep(1.0)
        return resp

    def auto_login(self):
        try:
            result = self._try_auto_login()
            if self.debug:
                print(f"DEBUG: login result is {result}")
        except:
            result = {
                ResultKey.STATUS: APIStatus.ERROR_RUNTIME,
                ResultKey.DETAILS: "Exception occurred when trying to login - please try later",
            }
        return result

    def _load_client_cmds_from_modules(self, cmd_modules):
        if cmd_modules:
            for m in cmd_modules:
                self.client_cmd_reg.register_module(m, include_invisible=False)

    def _load_client_cmds_from_module_specs(self, cmd_module_specs):
        if cmd_module_specs:
            for m in cmd_module_specs:
                self.client_cmd_reg.register_module_spec(m, include_invisible=False)

    def register_command(self, cmd_entry):
        self.all_cmds.append(cmd_entry.name)

    def _start_session_monitor(self, interval=0.2):
        self.sess_monitor_thread = threading.Thread(target=self._monitor_session, args=(interval,), daemon=True)
        self.sess_monitor_active = True
        self.sess_monitor_thread.daemon = True
        self.sess_monitor_thread.start()

    def _close_session_monitor(self):
        self.sess_monitor_active = False
        if self.sess_monitor_thread:
            self.sess_monitor_thread = None
        if self.debug:
            print("DEBUG: session monitor closed!")

    def check_session_status_on_server(self):
        return self.server_execute("_check_session")

    def _do_monitor_session(self, interval):
        self.fsm.set_current_state(_STATE_NAME_WAIT_FOR_SERVER_ADDR)
        while True:
            time.sleep(interval)

            if not self.sess_monitor_active:
                return ""

            if self.shutdown_asked:
                return ""

            if self.shutdown_received:
                return ""

            # see whether the session should be timed out for inactivity
            if (
                self.last_sess_activity_time
                and self.session_timeout_interval
                and time.time() - self.last_sess_activity_time > self.session_timeout_interval
            ):
                return "Your session is ended due to inactivity"

            next_state = self.fsm.execute()
            if not next_state:
                if self.fsm.error:
                    return self.fsm.error
                else:
                    return ""

    def _monitor_session(self, interval):
        try:
            msg = self._do_monitor_session(interval)
        except:
            msg = "exception occurred"

        self.server_sess_active = False
        self.fire_session_event(SessionEventType.SESSION_CLOSED, msg)

        # this is in the session_monitor thread - do not close the monitor or we'll run into
        # "cannot join current thread" error!
        self.close(close_session_monitor=False)

    def logout(self):
        """Send logout command to server."""
        self.in_logout = True
        resp = self.server_execute(InternalCommands.LOGOUT)
        self.close()
        return resp

    def close(self, close_session_monitor: bool = True):
        # this method can be called multiple times
        if self.closed:
            return

        self.closed = True
        self.service_finder.stop()
        self.server_sess_active = False
        self.shutdown_asked = True
        if close_session_monitor:
            self._close_session_monitor()

    def _get_command_list_from_server(self) -> bool:
        self.server_cmd_received = False
        self.server_execute(InternalCommands.GET_CMD_LIST, _CmdListReplyProcessor())
        self.server_cmd_reg.finalize(self.register_command)
        if not self.server_cmd_received:
            return False
        return True

    def _login(self) -> dict:
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

    def login(self, username: str):
        """Login using certification files and retrieve server side commands.

        Args:
            username: Username

        Returns:
            A dict of status and details
        """
        self.login_result = None
        self.server_execute(f"{InternalCommands.CERT_LOGIN} {username}", _LoginReplyProcessor())
        if self.login_result is None:
            return {
                ResultKey.STATUS: APIStatus.ERROR_RUNTIME,
                ResultKey.DETAILS: "Communication Error - please try later",
            }
        elif self.login_result == "REJECT":
            return {ResultKey.STATUS: APIStatus.ERROR_CERT, ResultKey.DETAILS: "Incorrect user name or certificate"}

        return self._login()

    def login_with_poc(self, username: str, poc_key: str):
        """Login using key for proof of concept example.

        Args:
            username: Username
            poc_key: key used for proof of concept admin login

        Returns:
            A dict of login status and details
        """
        self.login_result = None
        self.server_execute(f"{InternalCommands.PWD_LOGIN} {username} {poc_key}", _LoginReplyProcessor())
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
        return self._login()

    def _send_to_sock(self, sock, ctx: CommandContext):
        command = ctx.get_command()
        json_processor = ctx.get_json_processor()
        process_json_func = json_processor.process_server_reply

        conn = Connection(sock, self)
        conn.append_command(command)
        if self.token:
            conn.append_token(self.token)

        conn.close()
        ok = receive_and_process(sock, process_json_func)
        if not ok:
            process_json_func(
                make_error("Failed to communicate with Admin Server {} on {}".format(self.host, self.port))
            )

    def _try_command(self, cmd_ctx: CommandContext):
        """Try to execute a command on server side.

        Args:
            cmd_ctx: The command to execute.
        """
        # process_json_func can't return data because how "receive_and_process" is written.
        if self.debug:
            print(f"DEBUG: sending command '{cmd_ctx.get_command()}'")

        json_processor = _ServerReplyJsonProcessor(cmd_ctx)
        process_json_func = json_processor.process_server_reply
        cmd_ctx.set_json_processor(json_processor)

        with self.addr_lock:
            sp_host = self.host
            sp_port = self.port

        if self.debug:
            print(f"DEBUG: use server address {sp_host}:{sp_port}")

        try:
            if not self.poc:
                # SSL communication
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
                ssl_ctx.verify_mode = ssl.CERT_REQUIRED
                ssl_ctx.check_hostname = False

                ssl_ctx.load_verify_locations(self.ca_cert)
                ssl_ctx.load_cert_chain(certfile=self.client_cert, keyfile=self.client_key)

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    with ssl_ctx.wrap_socket(sock) as ssock:
                        ssock.connect((sp_host, sp_port))
                        self._send_to_sock(ssock, cmd_ctx)
            else:
                # poc without certs
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((sp_host, sp_port))
                    self._send_to_sock(sock, cmd_ctx)
        except Exception as e:
            if self.debug:
                secure_log_traceback()

            process_json_func(
                make_error(
                    "Failed to communicate with Admin Server {} on {}: {}".format(
                        sp_host, sp_port, secure_format_exception(e)
                    )
                )
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
        cmd_type, cmd_name, args, entries = self._get_command_detail(command)

        if cmd_type == _CMD_TYPE_UNKNOWN:
            return CommandInfo.UNKNOWN

        if len(entries) > 1:
            return CommandInfo.AMBIGUOUS

        ent = entries[0]
        assert isinstance(ent, CommandEntry)
        if ent.confirm == ConfirmMethod.AUTH:
            return CommandInfo.CONFIRM_AUTH
        elif ent.confirm == ConfirmMethod.PASSWORD:
            return CommandInfo.CONFIRM_PWD
        elif ent.confirm == ConfirmMethod.USER_NAME:
            return CommandInfo.CONFIRM_USER_NAME
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

    def do_command(self, command):
        """A convenient method to call commands using string.

        Args:
          command (str): command

        Returns:
            Object containing status and details (or direct response from server, which originally was just time and data)
        """
        self.last_sess_activity_time = time.time()

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

        return self.server_execute(command, cmd_entry=ent)

    def server_execute(self, command, reply_processor=None, cmd_entry=None):
        if self.in_logout:
            return {ResultKey.STATUS: APIStatus.SUCCESS, ResultKey.DETAILS: "session is logging out"}

        args = split_to_args(command)
        ctx = self._new_command_context(command, args, cmd_entry)
        start = time.time()
        ctx.set_reply_processor(reply_processor)
        self._try_command(ctx)
        secs = time.time() - start
        usecs = int(secs * 1000000)

        if self.debug:
            print(f"DEBUG: server_execute Done [{usecs} usecs] {datetime.now()}")

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
        if "session_inactive" in reply_data_full_response:
            return APIStatus.ERROR_INACTIVE_SESSION
        if "wrong server" in reply_data_full_response:
            return APIStatus.ERROR_SERVER_CONNECTION
        if "Failed to communicate" in reply_data_full_response:
            return APIStatus.ERROR_SERVER_CONNECTION
        if "invalid client" in reply_data_full_response:
            return APIStatus.ERROR_INVALID_CLIENT
        if "unknown site" in reply_data_full_response:
            return APIStatus.ERROR_INVALID_CLIENT
        if "not authorized" in reply_data_full_response:
            return APIStatus.ERROR_AUTHORIZATION
        return APIStatus.SUCCESS
