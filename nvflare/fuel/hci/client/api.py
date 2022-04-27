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

from nvflare.apis.overseer_spec import SP, OverseerAgent
from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.conn import Connection, receive_and_process
from nvflare.fuel.hci.proto import make_error
from nvflare.fuel.hci.reg import CommandModule, CommandRegister
from nvflare.fuel.hci.security import get_certificate_common_name
from nvflare.fuel.hci.table import Table
from nvflare.ha.ha_admin_cmds import HACommandModule

from .api_spec import AdminAPISpec, ReplyProcessor
from .api_status import APIStatus


class _DefaultReplyProcessor(ReplyProcessor):
    def process_shutdown(self, api: AdminAPI, msg: str):
        api.shutdown_received = True
        api.shutdown_msg = msg


class _LoginReplyProcessor(ReplyProcessor):
    """Reply processor for handling login and setting the token for the admin client."""

    def process_string(self, api: AdminAPI, item: str):
        api.login_result = item

    def process_token(self, api: AdminAPI, token: str):
        api.token = token


class _CmdListReplyProcessor(ReplyProcessor):
    """Reply processor to register available commands after getting back a table of commands from the server."""

    def process_table(self, api: AdminAPI, table: Table):
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
                visible=True,
                confirm=confirm,
            )

        api.server_cmd_received = True


class AdminAPI(AdminAPISpec):
    def __init__(
        self,
        host=None,
        port=None,
        ca_cert: str = "",
        client_cert: str = "",
        client_key: str = "",
        upload_dir: str = "",
        download_dir: str = "",
        server_cn=None,
        cmd_modules: Optional[List] = None,
        overseer_agent: OverseerAgent = None,
        auto_login: bool = False,
        user_name: str = None,
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
            server_cn: server cn (only used for validating server cn)
            cmd_modules: command modules to load and register. Note that FileTransferModule is initialized here with upload_dir and download_dir if cmd_modules is None.
            overseer_agent: initialized OverseerAgent to obtain the primary service provider to set the host and port of the active server
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
        cmd_modules.append(HACommandModule())

        self.overseer_agent = overseer_agent
        self.host = host
        self.port = port
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
            if not isinstance(self.overseer_agent, OverseerAgent):
                raise Exception("overseer_agent is missing but must be provided for secure context.")
            self.overseer_agent.set_secure_context(
                ca_path=self.ca_cert, cert_path=self.client_cert, prv_key_path=self.client_key
            )
        if self.overseer_agent:
            self.overseer_agent.start(self._overseer_callback)
        self.server_cn = server_cn
        self.debug = debug

        # for overseer agent
        self.ssid = None

        # for login
        self.token = None
        self.login_result = None
        if auto_login:
            self.auto_login = True
            if not user_name:
                raise Exception("for auto_login, user_name is required.")
            self.user_name = user_name

        self.server_cmd_reg = CommandRegister(app_ctx=self)
        self.client_cmd_reg = CommandRegister(app_ctx=self)
        self.server_cmd_received = False

        self.all_cmds = []
        self._load_client_cmds(cmd_modules)

        # for shutdown
        self.shutdown_received = False
        self.shutdown_msg = None

        self.server_sess_active = False

        self.sess_monitor_thread = None
        self.sess_monitor_active = False

    def _overseer_callback(self, overseer_agent):
        sp = overseer_agent.get_primary_sp()
        self._set_primary_sp(sp)

    def _set_primary_sp(self, sp: SP):
        if sp and sp.primary is True:
            if self.host != sp.name or self.port != int(sp.admin_port) or self.ssid != sp.service_session_id:
                # if needing to log out of previous server, this may be where to issue server_execute("_logout")
                self.host = sp.name
                self.port = int(sp.admin_port)
                self.ssid = sp.service_session_id
                print(
                    f"Got primary SP {self.host}:{sp.fl_port}:{self.port} from overseer. Host: {self.host} Admin_port: {self.port} SSID: {self.ssid}"
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

    def _load_client_cmds(self, cmd_modules):
        if cmd_modules:
            for m in cmd_modules:
                self.client_cmd_reg.register_module(m, include_invisible=False)
        self.client_cmd_reg.finalize(self.register_command)

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

    def login(self, username: str):
        """Login using certification files and retrieve server side commands.

        Args:
            username: Username

        Returns:
            A dict of status and details
        """
        self.login_result = None
        self._try_command(f"_cert_login {username}", _LoginReplyProcessor())
        if self.login_result is None:
            return {"status": APIStatus.ERROR_RUNTIME, "details": "Communication Error - please try later"}
        elif self.login_result == "REJECT":
            return {"status": APIStatus.ERROR_CERT, "details": "Incorrect user name or certificate"}

        # get command list from server
        self.server_cmd_received = False
        self._try_command("_commands", _CmdListReplyProcessor())
        self.server_cmd_reg.finalize(self.register_command)
        if not self.server_cmd_received:
            return {"status": APIStatus.ERROR_RUNTIME, "details": "Communication Error - please try later"}

        self.server_sess_active = True
        return {"status": APIStatus.SUCCESS, "details": "Login success"}

    def login_with_poc(self, username: str, poc_key: str):
        """Login using key for proof of concept example.

        Args:
            username: Username
            poc_key: key used for proof of concept admin login

        Returns:
            A dict of login status and details
        """
        self.login_result = None
        self._try_command(f"_login {username} {poc_key}", _LoginReplyProcessor())
        if self.login_result is None:
            return {"status": APIStatus.ERROR_RUNTIME, "details": "Communication Error - please try later"}
        elif self.login_result == "REJECT":
            return {"status": APIStatus.ERROR_CERT, "details": "Incorrect user name or certificate"}

        # get command list from server
        self.server_cmd_received = False
        self._try_command("_commands", _CmdListReplyProcessor())
        self.server_cmd_reg.finalize(self.register_command)
        if not self.server_cmd_received:
            return {"status": APIStatus.ERROR_RUNTIME, "details": "Communication Error - please try later"}

        self.server_sess_active = True
        return {"status": APIStatus.SUCCESS, "details": "Login success"}

    def _send_to_sock(self, sock, command, process_json_func):
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

    def _process_server_reply(self, resp):
        """Process the server reply and store the status/details into API's `command_result`

        Args:
            resp: The raw response that returns by the server.
        """
        if self.debug:
            print("DEBUG: Server Reply: {}".format(resp))
        # this resp is what is usually directly used to return, straight from server
        self.set_command_result(resp)
        reply_processor = _DefaultReplyProcessor() if self.reply_processor is None else self.reply_processor

        reply_processor.reply_start(self, resp)

        if resp is not None:
            data = resp["data"]
            for item in data:
                it = item["type"]
                if it == "string":
                    reply_processor.process_string(self, item["data"])
                elif it == "success":
                    reply_processor.process_success(self, item["data"])
                elif it == "error":
                    reply_processor.process_error(self, item["data"])
                    break
                elif it == "table":
                    table = Table(None)
                    table.set_rows(item["rows"])
                    reply_processor.process_table(self, table)
                elif it == "dict":
                    reply_processor.process_dict(self, item["data"])
                elif it == "token":
                    reply_processor.process_token(self, item["data"])
                elif it == "shutdown":
                    reply_processor.process_shutdown(self, item["data"])
                    break
                else:
                    reply_processor.protocol_error(self, "Invalid item type: " + it)
                    break
        else:
            reply_processor.protocol_error(self, "Protocol Error")

        reply_processor.reply_done(self)

    def _try_command(self, command, reply_processor):
        """Try to execute a command on server side.

        Args:
            command: The command to execute.
            reply_processor: An instance of ReplyProcessor

        """
        # process_json_func can't return data because how "receive_and_process" is written.
        self.reply_processor = reply_processor
        process_json_func = self._process_server_reply
        try:
            if not self.poc:
                # SSL communication
                ctx = ssl.create_default_context()
                ctx.verify_mode = ssl.CERT_REQUIRED
                ctx.check_hostname = False

                ctx.load_verify_locations(self.ca_cert)
                ctx.load_cert_chain(certfile=self.client_cert, keyfile=self.client_key)

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    with ctx.wrap_socket(sock) as ssock:
                        ssock.connect((self.host, self.port))
                        if self.server_cn:
                            # validate server CN
                            cn = get_certificate_common_name(ssock.getpeercert())
                            if cn != self.server_cn:
                                process_json_func(
                                    make_error("wrong server: expecting {} but connected {}".format(self.server_cn, cn))
                                )
                                return

                        self._send_to_sock(ssock, command, process_json_func)
            else:
                # poc without certs
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((self.host, self.port))
                    self._send_to_sock(sock, command, process_json_func)
        except Exception as ex:
            if self.debug:
                traceback.print_exc()

            process_json_func(
                make_error("Failed to communicate with Admin Server {} on {}: {}".format(self.host, self.port, ex))
            )

    def do_command(self, command):
        """A convenient method to call commands using string.

        Args:
          command (str): command

        Returns:
            Object containing status and details (or direct response from server, which originally was just time and data)
        """
        args = split_to_args(command)
        cmd_name = args[0]
        self.set_command_result(None)

        # check client side commands
        entries = self.client_cmd_reg.get_command_entries(cmd_name)
        if len(entries) > 1:
            return {
                "status": APIStatus.ERROR_SYNTAX,
                "details": f"Ambiguous client command {cmd_name} - qualify with scope",
            }
        elif len(entries) == 1:
            self.set_command_result(None)
            ent = entries[0]
            return_result = ent.handler(args, self)
            result = self.get_command_result()
            if return_result:
                return return_result
            if result is None:
                return {"status": APIStatus.ERROR_RUNTIME, "details": "Client did not respond"}
            return result

        # check server side commands
        entries = self.server_cmd_reg.get_command_entries(cmd_name)
        if len(entries) <= 0:
            return {
                "status": APIStatus.ERROR_SYNTAX,
                "details": f"Command {cmd_name} not found in server or client cmds",
            }
        elif len(entries) > 1:
            return {
                "status": APIStatus.ERROR_SYNTAX,
                "details": f"Ambiguous server command {cmd_name} - qualify with scope",
            }
        return self.server_execute(command)

    def server_execute(self, command, reply_processor=None):
        if not self.server_sess_active:
            return {"status": APIStatus.ERROR_INACTIVE_SESSION, "details": "API session is inactive"}

        self.set_command_result(None)
        start = time.time()
        self._try_command(command, reply_processor)
        secs = time.time() - start
        usecs = int(secs * 1000000)

        if self.debug:
            print(f"DEBUG: server_execute Done [{usecs} usecs] {datetime.now()}")

        result = self.get_command_result()
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
        self.set_command_result(result)
        return result
