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

import cmd
import getpass
import json
import os
import signal
import time
import traceback
import threading
from datetime import datetime
from enum import Enum
from functools import partial
from typing import List, Optional

from nvflare.fuel.hci.cmd_arg_utils import join_args, split_to_args
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandRegister, CommandSpec
from nvflare.fuel.hci.security import hash_password, verify_password
from nvflare.fuel.hci.table import Table
from nvflare.ha.overseer_agent import HttpOverseerAgent
from nvflare.ha.dummy_overseer_agent import DummyOverseerAgent
from nvflare.apis.overseer_spec import SP

from .api import AdminAPI
from .api_status import APIStatus


class _BuiltInCmdModule(CommandModule):
    def get_spec(self):
        return CommandModuleSpec(
            name="",
            cmd_specs=[
                CommandSpec(name="bye", description="exit from the client", usage="bye", handler_func=None),
                CommandSpec(name="help", description="get command help information", usage="help", handler_func=None),
                CommandSpec(
                    name="lpwd", description="print local work dir of the admin client", usage="lpwd", handler_func=None
                ),
            ],
        )


class CredentialType(str, Enum):
    PASSWORD = "password"
    CERT = "cert"


class AdminClient(cmd.Cmd):
    """Admin command prompt for submitting admin commands to the server through the CLI.

    Args:
        host: cn provisioned for the project, with this fully qualified domain name resolving to the IP of the FL server
        port: port provisioned as admin_port for FL admin communication, by default provisioned as 8003, must be int
        prompt: prompt to use for the command prompt
        ca_cert: path to CA Cert file, by default provisioned rootCA.pem
        client_cert: path to admin client Cert file, by default provisioned as client.crt
        client_key: path to admin client Key file, by default provisioned as client.key
        server_cn: server cn
        require_login: whether to require login
        credential_type: what type of credential to use
        cmd_modules: command modules to load and register
        debug: whether to print debug messages. False by default.
    """

    def __init__(
        self,
        host,
        port: int,
        prompt: str = "> ",
        ca_cert=None,
        client_cert=None,
        client_key=None,
        server_cn=None,
        require_login: bool = False,
        credential_type: str = CredentialType.PASSWORD,
        cmd_modules: Optional[List] = None,
        debug: bool = False,
    ):
        cmd.Cmd.__init__(self)
        self.intro = "Type help or ? to list commands.\n"
        self.prompt = prompt
        self.require_login = require_login
        self.credential_type = credential_type
        self.user_name = None
        self.password = None
        self.pwd = None

        self.debug = debug
        self.out_file = None
        self.no_stdout = False

        if not isinstance(credential_type, CredentialType):
            raise TypeError("invalid credential_type {}".format(credential_type))

        modules = [_BuiltInCmdModule()]
        if cmd_modules:
            if not isinstance(cmd_modules, list):
                raise TypeError("cmd_modules must be a list.")
            for m in cmd_modules:
                if not isinstance(m, CommandModule):
                    raise TypeError("cmd_modules must be a list of CommandModule")
                modules.append(m)

        poc = True if self.credential_type == CredentialType.PASSWORD else False

        self.api = AdminAPI(
            host=host,
            port=port,
            ca_cert=ca_cert,
            client_cert=client_cert,
            client_key=client_key,
            server_cn=server_cn,
            cmd_modules=modules,
            debug=self.debug,
            poc=poc,
        )

        signal.signal(signal.SIGUSR1, partial(self.session_signal_handler))

        self.ssid = None
        self.overseer_agent = self._create_overseer_agent()

        if self.credential_type == CredentialType.CERT:
            if self.overseer_agent:
                self.overseer_agent.set_secure_context(ca_path=ca_cert,
                                                   cert_path=client_cert,
                                                   prv_key_path=client_key)

        self.overseer_agent.start(self.overseer_callback)

    def _create_overseer_agent(self):
        overseer_agent = HttpOverseerAgent(
            overseer_end_point="http://127.0.0.1:5000/api/v1",
            project="example_project",
            role="admin",
            name="localhost",
            heartbeat_interval=6,
        )
        overseer_agent = DummyOverseerAgent(
            sp_end_point="localhost:8002:8003",
            heartbeat_interval=6,
        )

        return overseer_agent

    def overseer_callback(self, overseer_agent):
        sp = overseer_agent.get_primary_sp()
        self.set_primary_sp(sp)

    def set_primary_sp(self, sp: SP):
        if sp and sp.primary is True:
            if self.api.host != sp.name or self.api.port != int(sp.admin_port) or self.ssid != sp.service_session_id:
                self.api.host = sp.name
                self.api.port = int(sp.admin_port)
                self.ssid = sp.service_session_id
                print(f"Got primary SP. Host: {self.api.host} Admin_port: {self.api.port} SSID: {self.ssid}")

                thread = threading.Thread(target=self._login_sp)
                thread.start()

    def _login_sp(self):
        self.do_bye("logout")
        self.login()

    def session_ended(self, message):
        self.write_error(message)
        os.kill(os.getpid(), signal.SIGUSR1)

    def session_signal_handler(self, signum, frame):
        self.api.close_session_monitor()
        raise ConnectionError

    def _set_output_file(self, file, no_stdout):
        self._close_output_file()
        self.out_file = file
        self.no_stdout = no_stdout

    def _close_output_file(self):
        if self.out_file:
            self.out_file.close()
            self.out_file = None
        self.no_stdout = False

    def do_bye(self, arg):
        """exit from the client"""
        if self.require_login:
            self.api.server_execute("_logout")
        return True

    def do_lpwd(self, arg):
        """print local current work dir"""
        self.write_string(os.getcwd())

    def emptyline(self):
        return

    def _show_one_command(self, cmd_name, reg):
        entries = reg.get_command_entries(cmd_name)
        if len(entries) <= 0:
            self.write_string("Undefined command {}\n".format(cmd_name))
            return

        for e in entries:
            if not e.visible:
                continue

            if len(e.scope.name) > 0:
                self.write_string("Command: {}.{}".format(e.scope.name, cmd_name))
            else:
                self.write_string("Command: {}".format(cmd_name))

            self.write_string("Description: {}".format(e.desc))
            self.write_string("Usage: {}\n".format(e.usage))

    def _show_commands(self, reg: CommandRegister):
        table = Table(["Scope", "Command", "Description"])
        for scope_name in sorted(reg.scopes):
            scope = reg.scopes[scope_name]
            for cmd_name in sorted(scope.entries):
                e = scope.entries[cmd_name]
                if e.visible:
                    table.add_row([scope_name, cmd_name, e.desc])
        self.write_table(table)

    def do_help(self, arg):
        if len(arg) <= 0:
            self.write_string("Client Commands")
            self._show_commands(self.api.client_cmd_reg)

            self.write_string("\nServer Commands")
            self._show_commands(self.api.server_cmd_reg)
        else:
            server_cmds = []
            local_cmds = []
            parts = arg.split()
            for p in parts:
                entries = self.api.client_cmd_reg.get_command_entries(p)
                if len(entries) > 0:
                    local_cmds.append(p)

                entries = self.api.server_cmd_reg.get_command_entries(p)
                if len(entries) > 0:
                    server_cmds.append(p)

            if len(local_cmds) > 0:
                self.write_string("Client Commands")
                self.write_string("---------------")
                for cmd_name in local_cmds:
                    self._show_one_command(cmd_name, self.api.client_cmd_reg)

            if len(server_cmds) > 0:
                self.write_string("Server Commands")
                self.write_string("---------------")
                for cmd_name in server_cmds:
                    self._show_one_command(cmd_name, self.api.server_cmd_reg)

    def complete(self, text, state):
        results = [x + " " for x in self.api.all_cmds if x.startswith(text)] + [None]
        return results[state]

    def default(self, line):
        self._close_output_file()
        try:
            return self._do_default(line)
        except KeyboardInterrupt:
            self.write_stdout("\n")
        except BaseException as ex:
            if self.debug:
                traceback.print_exc()
            self.write_stdout("exception occurred: {}".format(ex))
        self._close_output_file()

    def _do_default(self, line):
        args = split_to_args(line)
        cmd_name = args[0]

        # check for file output
        out_file_name = None
        no_stdout = False
        out_arg_idx = 0
        for i in range(len(args)):
            arg = args[i]
            if arg.startswith(">") and out_file_name is not None:
                self.write_error("only one output file is supported")
                return

            if arg.startswith(">>"):
                # only output to file
                out_file_name = arg[2:]
                no_stdout = True
                out_arg_idx = i
            elif arg.startswith(">"):
                # only output to file
                out_file_name = arg[1:]
                no_stdout = False
                out_arg_idx = i

        if out_file_name is not None:
            if len(out_file_name) <= 0:
                self.write_error("output file name must not be empty")
                return
            args.pop(out_arg_idx)
            line = join_args(args)
            try:
                out_file = open(out_file_name, "w")
            except BaseException as ex:
                self.write_error("cannot open file {}: {}".format(out_file_name, ex))
                return

            self._set_output_file(out_file, no_stdout)

        # check client command first
        entries = self.api.client_cmd_reg.get_command_entries(cmd_name)
        if len(entries) > 1:
            self.write_string("Ambiguous client command {} - qualify with scope".format(cmd_name))
            return
        elif len(entries) == 1:
            ent = entries[0]
            resp = ent.handler(args, self.api)
            self.print_resp(resp)
            if resp["status"] == APIStatus.ERROR_INACTIVE_SESSION:
                return True
            return

        entries = self.api.server_cmd_reg.get_command_entries(cmd_name)
        if len(entries) <= 0:
            self.write_string("Undefined server command {}".format(cmd_name))
            return
        elif len(entries) > 1:
            self.write_string("Ambiguous server command {} - qualify with scope".format(cmd_name))
            return

        ent = entries[0]
        confirm_method = ent.confirm
        if ent.confirm == "auth":
            if self.credential_type == CredentialType.PASSWORD:
                confirm_method = "pwd"
            elif self.user_name:
                confirm_method = "username"
            else:
                confirm_method = "yesno"

        if confirm_method == "yesno":
            answer = input("Are you sure (Y/N): ")
            answer = answer.lower()
            if answer != "y" and answer != "yes":
                return
        elif confirm_method == "username":
            answer = input("Confirm with User Name: ")
            if answer != self.user_name:
                self.write_string("user name mismatch")
                return
        elif confirm_method == "pwd":
            pwd = getpass.getpass("Enter password to confirm: ")
            if not verify_password(self.pwd, pwd):
                self.write_string("Not authenticated")
                return

        start = time.time()
        resp = self.api.server_execute(line)
        secs = time.time() - start
        usecs = int(secs * 1000000)
        done = "Done [{} usecs] {}".format(usecs, datetime.now())
        self.print_resp(resp)
        if resp["status"] == APIStatus.ERROR_INACTIVE_SESSION:
            return True
        self.write_stdout(done)
        if self.api.shutdown_received:
            # exit the client
            self.write_string(self.api.shutdown_msg)
            return True

    def cmdloop(self, intro=None):
        """Repeatedly issue a prompt, accept input, parse an initial prefix
        off the received input, and dispatch to action methods, passing them
        the remainder of the line as argument.

        Overriding what is in cmd.Cmd to handle exiting client on Ctrl+D (EOF).
        """

        self.preloop()
        if self.use_rawinput and self.completekey:
            try:
                import readline

                self.old_completer = readline.get_completer()
                readline.set_completer(self.complete)
                readline.parse_and_bind(self.completekey + ": complete")
            except ImportError:
                pass
        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                self.stdout.write(str(self.intro) + "\n")
            stop = None
            while not stop:
                if self.cmdqueue:
                    line = self.cmdqueue.pop(0)
                else:
                    if self.use_rawinput:
                        try:
                            line = input(self.prompt)
                        except (EOFError, ConnectionError):
                            line = "bye"
                        except KeyboardInterrupt:
                            self.stdout.write("\n")
                            line = "\n"
                    else:
                        self.stdout.write(self.prompt)
                        self.stdout.flush()
                        line = self.stdin.readline()
                        if not len(line):
                            line = "EOF"
                        else:
                            line = line.rstrip("\r\n")
                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
            self.postloop()
        finally:
            if self.use_rawinput and self.completekey:
                try:
                    import readline

                    readline.set_completer(self.old_completer)
                except ImportError:
                    pass

    def run(self):

        try:
            while self.api.token is None:
                time.sleep(1.0)

            # self.api.start_session_monitor(self.session_ended)
            self.cmdloop(intro='Type ? to list commands; type "? cmdName" to show usage of a command.')
        finally:
            self.overseer_agent.end()

    def login(self):
        if self.require_login:
            if self.user_name:
                user_name = self.user_name
            else:
                user_name = input("User Name: ")

            if self.credential_type == CredentialType.PASSWORD:
                while True:
                    if self.password:
                        pwd = self.password
                    else:
                        pwd = getpass.getpass("Password: ")
                    # print(f"host: {self.api.host} port: {self.api.port}")
                    self.api.login_with_password(username=user_name, password=pwd)
                    self.stdout.write(f"login_result: {self.api.login_result} token: {self.api.token}\n{self.prompt}")
                    if self.api.login_result == "OK":
                        self.user_name = user_name
                        self.password = pwd
                        self.pwd = hash_password(pwd)
                        break
                    elif self.api.login_result == "REJECT":
                        print("Incorrect password - please try again")
                    else:
                        print("Communication Error - please try later")
                        return False
            elif self.credential_type == CredentialType.CERT:
                self.api.login(username=user_name)
                if self.api.login_result == "OK":
                    self.user_name = user_name
                elif self.api.login_result == "REJECT":
                    print("Incorrect user name or certificate")
                    return False
                else:
                    print("Communication Error - please try later")
                    return False
        return True

    def print_resp(self, resp: dict):
        """Prints the server response

        Args:
            resp (dict): The server response.
        """
        if "details" in resp:
            if isinstance(resp["details"], str):
                self.write_string(resp["details"])
            if isinstance(resp["details"], Table):
                self.write_table(resp["details"])

        if "data" in resp:
            for item in resp["data"]:
                item_type = item["type"]
                if item_type == "string":
                    self.write_string(item["data"])
                elif item_type == "table":
                    table = Table(None)
                    table.set_rows(item["rows"])
                    self.write_table(table)
                elif item_type == "error":
                    self.write_error(item["data"])
                elif item_type == "dict":
                    self.write_dict(item["data"])

        if "details" not in resp and "data" not in resp:
            self.write_string("Response is not correct.")

    def write_stdout(self, data: str):
        self.stdout.write(data + "\n")

    def _write(self, content: str):
        if not self.no_stdout:
            self.stdout.write(content)
        if self.out_file:
            self.out_file.write(content)

    def write_string(self, data: str):
        content = data + "\n"
        self._write(content)

    def write_table(self, table: Table):
        if not self.no_stdout:
            table.write(self.stdout)
        if self.out_file:
            table.write(self.out_file)

    def write_dict(self, data: dict):
        content = json.dumps(data, indent=2) + "\n"
        self._write(content)

    def write_error(self, err: str):
        content = "Error: " + err + "\n"
        self._write(content)

    def flush(self):
        if not self.no_stdout:
            self.stdout.flush()
        if self.out_file:
            self.out_file.flush()
