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

from __future__ import annotations

import socket
import ssl
import threading
import time
import traceback
from datetime import datetime
from typing import List, Optional

from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.conn import Connection, receive_and_process
from nvflare.fuel.hci.proto import make_error, ConfirmMethod
from nvflare.fuel.hci.reg import CommandModule, CommandRegister, CommandEntry
from nvflare.fuel.hci.table import Table

from .api_spec import AdminAPISpec, ServiceFinder, ReplyProcessor, CommandContext, CommandCtxKey, CommandInfo
from .api_status import APIStatus

_CMD_TYPE_UNKNOWN = 0
_CMD_TYPE_CLIENT = 1
_CMD_TYPE_SERVER = 2


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
            data = resp["data"]
            for item in data:
                it = item["type"]
                if it == "string":
                    reply_processor.process_string(ctx, item["data"])
                elif it == "success":
                    reply_processor.process_success(ctx, item["data"])
                elif it == "error":
                    reply_processor.process_error(ctx, item["data"])
                    break
                elif it == "table":
                    table = Table(None)
                    table.set_rows(item["rows"])
                    reply_processor.process_table(ctx, table)
                elif it == "dict":
                    reply_processor.process_dict(ctx, item["data"])
                elif it == "token":
                    reply_processor.process_token(ctx, item["data"])
                elif it == "shutdown":
                    reply_processor.process_shutdown(ctx, item["data"])
                    break
                else:
                    reply_processor.protocol_error(ctx, "Invalid item type: " + it)
                    break
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
            if len(row) > 5:
                client_cmd = row[5]

            # if confirm == 'auth' and not client.require_login:
            # the user is not authenticated - skip this command
            # continue
            print("add command: {}.{}.{}".format(scope, cmd_name, client_cmd))
            api.server_cmd_reg.add_command(
                scope_name=scope,
                cmd_name=cmd_name,
                desc=desc,
                usage=usage,
                handler=None,
                authz_func=None,
                visible=True,
                confirm=confirm,
                client_cmd=client_cmd,
                map_client_cmd=True
            )

        api.server_cmd_received = True


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
    ):
        """Underlying API to keep certs, keys and connection information and to execute admin commands through do_command.

        Args:
            host: cn provisioned for the server, with this fully qualified domain name resolving to the IP of the FL server. This may be set by the OverseerAgent.
            port: port provisioned as admin_port for FL admin communication, by default provisioned as 8003, must be int if provided. This may be set by the OverseerAgent.
            ca_cert: path to CA Cert file, by default provisioned rootCA.pem
            client_cert: path to admin client Cert file, by default provisioned as client.crt
            client_key: path to admin client Key file, by default provisioned as client.key
            upload_dir: File transfer upload directory. Folders uploaded to the server to be deployed must be here. Folder must already exist and be accessible.
            download_dir: File transfer download directory. Can be same as upload_dir. Folder must already exist and be accessible.
            cmd_modules: command modules to load and register. Note that FileTransferModule is initialized here with upload_dir and download_dir if cmd_modules is None.
            service_finder: used to obtain the primary service provider to set the host and port of the active server
            auto_login: Whether to use stored credentials to automatically log in (required to be True with OverseerAgent to provide high availability)
            user_name: Username to authenticate with FL server
            poc: Whether to enable poc mode for using the proof of concept example without secure communication.
            debug: Whether to print debug messages, which can help with diagnosing problems. False by default.
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
        assert isinstance(service_finder, ServiceFinder), \
            "service_finder should be ServiceFinder but got {}".format(type(service_finder))

        cmd_module = service_finder.get_command_module()
        if cmd_module:
            cmd_modules.append(cmd_module)

        self.service_finder = service_finder
        self.host = None
        self.port = None
        self.ssid = None
        self.addr_lock = threading.Lock()

        self.poc = poc
        if self.poc:
            self.poc_key = "admin"
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
                ca_cert_path=self.ca_cert,
                cert_path=self.client_cert,
                private_key_path=self.client_key
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

        self.sess_monitor_thread = None
        self.sess_monitor_active = False

        self.service_finder.start(self._handle_sp_address_change)

    def _handle_sp_address_change(self, host: str, port: int, ssid: str):
        # if needing to log out of previous server, this may be where to issue server_execute("_logout")
        with self.addr_lock:
            self.host = host
            self.port = port
            self.ssid = ssid

        print(
            f"Got service provider address: host={host} admin_port={port} SSID={ssid}"
        )

        thread = threading.Thread(target=self._login_sp)
        thread.start()

    def _login_sp(self):
        if not self._auto_login():
            print("cannot log in, shutting down...")
            self.shutdown_received = True

    def _auto_login(self):
        try_count = 0
        while try_count < 5:
            if self.poc:
                self.login_with_poc(username=self.user_name, poc_key=self.poc_key)
                print(f"login_result: {self.login_result} token: {self.token}")
                if self.login_result == "OK":
                    return True
                elif self.login_result == "REJECT":
                    print("Incorrect key for POC mode.")
                    return False
                else:
                    print("Communication Error - please try later")
                    try_count += 1
            else:
                self.login(username=self.user_name)
                if self.login_result == "OK":
                    return True
                elif self.login_result == "REJECT":
                    print("Incorrect user name or certificate.")
                    return False
                else:
                    print("Communication Error - please try later")
                    try_count += 1
            time.sleep(1.0)
        return False

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

    def start_session_monitor(self, session_ended_callback, interval=5):
        if self.sess_monitor_thread and self.sess_monitor_thread.is_alive():
            self.close_session_monitor()
        self.sess_monitor_thread = threading.Thread(
            target=self._check_session, args=(session_ended_callback, interval), daemon=True
        )
        self.sess_monitor_active = True
        self.sess_monitor_thread.start()

    def close_session_monitor(self):
        self.sess_monitor_active = False
        if self.sess_monitor_thread and self.sess_monitor_thread.is_alive():
            self.sess_monitor_thread.join()
            self.sess_monitor_thread = None

    def _check_session(self, session_ended_callback, interval):
        error_msg = ""
        connection_error_counter = 0
        while True:
            time.sleep(interval)

            if not self.sess_monitor_active:
                return

            if self.shutdown_received:
                error_msg = self.shutdown_msg
                break

            resp = self.server_execute("_check_session")
            status = resp["status"]

            connection_error_counter += 1
            if status != APIStatus.ERROR_SERVER_CONNECTION:
                connection_error_counter = 0

            if status in APIStatus.ERROR_INACTIVE_SESSION or (
                status in APIStatus.ERROR_SERVER_CONNECTION and connection_error_counter > 60 // interval
            ):
                for item in resp["data"]:
                    if item["type"] == "error":
                        error_msg = item["data"]
                break

        self.server_sess_active = False
        session_ended_callback(error_msg)

    def logout(self):
        """Send logout command to server."""
        resp = self.server_execute("_logout")
        self.server_sess_active = False
        return resp

    def get_command_list_from_server(self):
        # get command list from server
        self.server_cmd_received = False
        self.server_execute("_commands", _CmdListReplyProcessor())
        self.server_cmd_reg.finalize(self.register_command)
        if not self.server_cmd_received:
            return {"status": APIStatus.ERROR_RUNTIME, "details": "Communication Error - please try later"}

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
        return {"status": APIStatus.SUCCESS, "details": "Login success"}

    def is_ready(self) -> bool:
        """Whether the API is ready for executing commands.
        """
        return self.token is not None

    def login(self, username: str):
        """Login using certification files and retrieve server side commands.

        Args:
            username: Username

        Returns:
            A dict of status and details
        """
        self.login_result = None
        print("trying to login with cert ...")
        self.server_execute(f"_cert_login {username}", _LoginReplyProcessor())
        if self.login_result is None:
            return {"status": APIStatus.ERROR_RUNTIME, "details": "Communication Error - please try later"}
        elif self.login_result == "REJECT":
            return {"status": APIStatus.ERROR_CERT, "details": "Incorrect user name or certificate"}

        # get command list from server
        return self.get_command_list_from_server()

    def login_with_poc(self, username: str, poc_key: str):
        """Login using key for proof of concept example.

        Args:
            username: Username
            poc_key: key used for proof of concept admin login

        Returns:
            A dict of login status and details
        """
        self.login_result = None
        self.server_execute(f"_login {username} {poc_key}", _LoginReplyProcessor())
        if self.login_result is None:
            return {"status": APIStatus.ERROR_RUNTIME, "details": "Communication Error - please try later"}
        elif self.login_result == "REJECT":
            return {"status": APIStatus.ERROR_CERT, "details": "Incorrect user name or certificate"}

        # get command list from server
        return self.get_command_list_from_server()

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

        json_processor = _ServerReplyJsonProcessor(cmd_ctx)
        process_json_func = json_processor.process_server_reply
        cmd_ctx.set_json_processor(json_processor)

        with self.addr_lock:
            sp_host = self.host
            sp_port = self.port

        try:
            if not self.poc:
                # SSL communication
                ssl_ctx = ssl.create_default_context()
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
        except Exception as ex:
            if self.debug:
                traceback.print_exc()

            process_json_func(
                make_error("Failed to communicate with Admin Server {} on {}: {}".format(
                    sp_host, sp_port, ex))
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
            return {"status": APIStatus.ERROR_RUNTIME, "details": "Client did not respond"}
        return result

    def do_command(self, command):
        """A convenient method to call commands using string.

        Args:
          command (str): command

        Returns:
            Object containing status and details (or direct response from server, which originally was just time and data)
        """
        cmd_type, cmd_name, args, entries = self._get_command_detail(command)
        if cmd_type == _CMD_TYPE_UNKNOWN:
            return {
                "status": APIStatus.ERROR_SYNTAX,
                "details": f"Command {cmd_name} not found",
            }

        if len(entries) > 1:
            return {
                "status": APIStatus.ERROR_SYNTAX,
                "details": f"Ambiguous command {cmd_name} - qualify with scope",
            }

        ent = entries[0]
        if cmd_type == _CMD_TYPE_CLIENT:
            return self._do_client_command(
                command=command,
                args=args,
                ent=ent
            )

        # server command
        if not self.server_sess_active:
            return {"status": APIStatus.ERROR_INACTIVE_SESSION,
                    "details": "Session is inactive"}

        return self.server_execute(command, cmd_entry=ent)

    def server_execute(self, command, reply_processor=None, cmd_entry=None):
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
        if result is None:
            return {"status": APIStatus.ERROR_SERVER_CONNECTION, "details": "Server did not respond"}
        if "data" in result:
            for item in result["data"]:
                if item["type"] == "error":
                    if "session_inactive" in item["data"]:
                        result.update({"status": APIStatus.ERROR_INACTIVE_SESSION})
                    elif any(
                        err in item["data"] for err in ("Failed to communicate with Admin Server", "wrong server")
                    ):
                        result.update({"status": APIStatus.ERROR_SERVER_CONNECTION})
        if "status" not in result:
            result.update({"status": APIStatus.SUCCESS})
        return result
