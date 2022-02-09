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
import time
import traceback
from datetime import datetime

from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.conn import Connection, receive_and_process
from nvflare.fuel.hci.proto import make_error
from nvflare.fuel.hci.reg import CommandModule, CommandRegister
from nvflare.fuel.hci.security import get_certificate_common_name
from nvflare.fuel.hci.table import Table

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
    """Underlying API to keep certs, keys and connection information and to execute admin commands through do_command.

    Args:
        host: cn provisioned for the project, with this fully qualified domain name resolving to the IP of the FL server
        port: port provisioned as admin_port for FL admin communication, by default provisioned as 8003, must be int
        ca_cert: path to CA Cert file, by default provisioned rootCA.pem
        client_cert: path to admin client Cert file, by default provisioned as client.crt
        client_key: path to admin client Key file, by default provisioned as client.key
        upload_dir: File transfer upload directory. Folders uploaded to the server to be deployed must be here. Folder must already exist and be accessible.
        download_dir: File transfer download directory. Can be same as upload_dir. Folder must already exist and be accessible.
        server_cn: server cn (only used for validating server cn)
        cmd_modules: command modules to load and register
        poc: Whether to enable poc mode for using the proof of concept example without secure communication.
        debug: Whether to print debug messages, which can help with diagnosing problems. False by default.
    """

    def __init__(
        self,
        host,
        port,
        ca_cert="",
        client_cert="",
        client_key="",
        upload_dir="",
        download_dir="",
        server_cn=None,
        cmd_modules=None,
        poc: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        if cmd_modules is None:
            from .file_transfer import FileTransferModule

            cmd_modules = [FileTransferModule(upload_dir=upload_dir, download_dir=download_dir)]
        self.host = host
        self.port = port
        self.poc = poc
        if not self.poc:
            if len(ca_cert) <= 0:
                raise Exception("missing CA Cert file name")
            self.ca_cert = ca_cert
            if len(client_cert) <= 0:
                raise Exception("missing Client Cert file name")
            self.client_cert = client_cert
            if len(client_key) <= 0:
                raise Exception("missing Client Key file name")
            self.client_key = client_key
        self.server_cn = server_cn
        self.debug = debug

        # for login
        self.token = None
        self.login_result = None

        self.server_cmd_reg = CommandRegister(app_ctx=self)
        self.client_cmd_reg = CommandRegister(app_ctx=self)
        self.server_cmd_received = False

        self.all_cmds = []
        self._load_client_cmds(cmd_modules)

        # for shutdown
        self.shutdown_received = False
        self.shutdown_msg = None

    def _load_client_cmds(self, cmd_modules):
        if cmd_modules:
            if not isinstance(cmd_modules, list):
                raise TypeError("cmd_modules must be a list")
            for m in cmd_modules:
                if not isinstance(m, CommandModule):
                    raise TypeError("cmd_modules must be a list of CommandModule")
                self.client_cmd_reg.register_module(m, include_invisible=False)
        self.client_cmd_reg.finalize(self.register_command)

    def register_command(self, cmd_entry):
        self.all_cmds.append(cmd_entry.name)

    def logout(self):
        """Send logout command to server."""
        return self.server_execute("_logout")

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

        return {"status": APIStatus.SUCCESS, "details": "Login success"}

    def login_with_password(self, username: str, password: str):
        """Login using password for poc example.

        Args:
            username: Username
            password: password

        Returns:
            A dict of login status and details
        """
        self.login_result = None
        self._try_command(f"_login {username} {password}", _LoginReplyProcessor())
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
            ent.handler(args, self)
            result = self.get_command_result()
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
        self.set_command_result(None)
        start = time.time()
        self._try_command(command, reply_processor)
        secs = time.time() - start
        usecs = int(secs * 1000000)

        if self.debug:
            print(f"DEBUG: server_execute Done [{usecs} usecs] {datetime.now()}")

        result = self.get_command_result()
        if result is None:
            return {"status": APIStatus.ERROR_RUNTIME, "details": "Server did not respond"}
        if "status" not in result:
            result.update({"status": APIStatus.SUCCESS})
        self.set_command_result(result)
        return result
