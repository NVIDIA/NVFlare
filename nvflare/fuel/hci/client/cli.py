# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from datetime import datetime
from typing import List, Optional

from nvflare.fuel.hci.cmd_arg_utils import join_args, split_to_args
from nvflare.fuel.hci.proto import CredentialType
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandRegister, CommandSpec
from nvflare.fuel.hci.security import hash_password, verify_password
from nvflare.fuel.hci.table import Table
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .api import AdminAPI, CommandInfo, SessionEventType
from .api_spec import ServiceFinder
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


class AdminClient(cmd.Cmd):
    """Admin command prompt for submitting admin commands to the server through the CLI.

    Args:
        host: cn provisioned for the server, with this fully qualified domain name resolving to the IP of the FL server. This may be set by the OverseerAgent.
        port: port provisioned as admin_port for FL admin communication, by default provisioned as 8003, must be int if provided. This may be set by the OverseerAgent.
        prompt: prompt to use for the command prompt
        ca_cert: path to CA Cert file, by default provisioned rootCA.pem
        client_cert: path to admin client Cert file, by default provisioned as client.crt
        client_key: path to admin client Key file, by default provisioned as client.key
        credential_type: what type of credential to use
        cmd_modules: command modules to load and register
        idp_agent: IdpAgent to obtain the primary service provider to set the host and port of the active server
        debug: whether to print debug messages. False by default.
    """

    def __init__(
        self,
        prompt: str = "> ",
        credential_type: CredentialType = CredentialType.PASSWORD,
        ca_cert=None,
        client_cert=None,
        client_key=None,
        upload_dir="",
        download_dir="",
        cmd_modules: Optional[List] = None,
        service_finder: ServiceFinder = None,
        session_timeout_interval=900,  # close the client after 15 minutes of inactivity
        debug: bool = False,
    ):
        cmd.Cmd.__init__(self)
        self.intro = "Type help or ? to list commands.\n"
        self.prompt = prompt
        self.user_name = "admin"
        self.pwd = None
        self.credential_type = credential_type

        self.service_finder = service_finder
        self.debug = debug
        self.out_file = None
        self.no_stdout = False
        self.stopped = False  # use this flag to prevent unnecessary signal exception

        if not isinstance(service_finder, ServiceFinder):
            raise TypeError("service_finder must be ServiceProvider but got {}.".format(type(service_finder)))

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

        self._get_login_creds()

        self.api = AdminAPI(
            ca_cert=ca_cert,
            client_cert=client_cert,
            client_key=client_key,
            upload_dir=upload_dir,
            download_dir=download_dir,
            cmd_modules=modules,
            service_finder=self.service_finder,
            user_name=self.user_name,
            debug=self.debug,
            poc=poc,
            session_event_cb=self.handle_session_event,
            session_timeout_interval=session_timeout_interval,
            session_status_check_interval=1800,  # check server for session status every 30 minutes
        )
        # signal.signal(signal.SIGUSR1, partial(self.session_signal_handler))
        signal.signal(signal.SIGUSR1, self.session_signal_handler)

    def handle_session_event(self, event_type: str, message: str):
        if self.debug:
            print(f"DEBUG: received session event: {event_type}")

        if message:
            self.write_string(message)

        if event_type == SessionEventType.SESSION_CLOSED:
            os.kill(os.getpid(), signal.SIGUSR1)

    def session_signal_handler(self, signum, frame):
        if self.stopped:
            return

        # the signal is only for the main thread
        # the session monitor thread signals the main thread to stop
        if self.debug:
            print("DEBUG: signal received to close session")

        self.api.close()

        # use exception to interrupt the main cmd loop
        raise RuntimeError("Session Closed")

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
        """Exit from the client.

        If the arg is not logout, in other words, the user is issuing the bye command to shut down the client, or it is
        called by inputting the EOF character, a message will display that the admin client is shutting down."""
        if arg != "logout":
            print("Shutting down admin client, please wait...")
        self.api.logout()
        return True

    def do_lpwd(self, arg):
        """print local current work dir"""
        self.write_string(os.getcwd())

    def emptyline(self):
        return

    def _show_one_command(self, cmd_name, reg, show_invisible=False):
        entries = reg.get_command_entries(cmd_name)
        if len(entries) <= 0:
            self.write_string("Undefined command {}\n".format(cmd_name))
            return

        for e in entries:
            if not e.visible and not show_invisible:
                continue

            if len(e.scope.name) > 0:
                self.write_string("Command: {}.{}".format(e.scope.name, cmd_name))
            else:
                self.write_string("Command: {}".format(cmd_name))

            self.write_string("Description: {}".format(e.desc))
            self.write_string("Usage: {}\n".format(e.usage))

    def _show_commands(self, reg: CommandRegister):
        table = Table(["Command", "Description"])
        for scope_name in sorted(reg.scopes):
            scope = reg.scopes[scope_name]
            for cmd_name in sorted(scope.entries):
                e = scope.entries[cmd_name]
                if e.visible:
                    table.add_row([cmd_name, e.desc])
        self.write_table(table)

    def do_help(self, arg):
        if len(arg) <= 0:
            self.write_string("Client Initiated / Overseer Commands")
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
                    self._show_one_command(cmd_name, self.api.server_cmd_reg, show_invisible=True)

    def complete(self, text, state):
        results = [x + " " for x in self.api.all_cmds if x.startswith(text)] + [None]
        return results[state]

    def default(self, line):
        self._close_output_file()
        try:
            return self._do_default(line)
        except KeyboardInterrupt:
            self.write_stdout("\n")
        except Exception as e:
            if self.debug:
                secure_log_traceback()
            self.write_stdout(f"exception occurred: {secure_format_exception(e)}")
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
            except Exception as e:
                self.write_error(f"cannot open file {out_file_name}: {secure_format_exception(e)}")
                return

            self._set_output_file(out_file, no_stdout)

        # check client command first
        info = self.api.check_command(line)
        if info == CommandInfo.UNKNOWN:
            self.write_string("Undefined command {}".format(cmd_name))
            return
        elif info == CommandInfo.AMBIGUOUS:
            self.write_string("Ambiguous command {} - qualify with scope".format(cmd_name))
            return
        elif info == CommandInfo.CONFIRM_AUTH:
            if self.credential_type == CredentialType.PASSWORD:
                info = CommandInfo.CONFIRM_PWD
            elif self.user_name:
                info = CommandInfo.CONFIRM_USER_NAME
            else:
                info = CommandInfo.CONFIRM_YN

        if info == CommandInfo.CONFIRM_YN:
            answer = input("Are you sure (y/N): ")
            answer = answer.lower()
            if answer != "y" and answer != "yes":
                return
        elif info == CommandInfo.CONFIRM_USER_NAME:
            answer = input("Confirm with User Name: ")
            if answer != self.user_name:
                self.write_string("user name mismatch")
                return
        elif info == CommandInfo.CONFIRM_PWD:
            pwd = getpass.getpass("Enter password to confirm: ")
            if not verify_password(self.pwd, pwd):
                self.write_string("Not authenticated")
                return

        # execute the command!
        start = time.time()
        resp = self.api.do_command(line)
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
            while not self.api.is_ready():
                time.sleep(1.0)
                if self.api.shutdown_received:
                    return False

            self.cmdloop(intro='Type ? to list commands; type "? cmdName" to show usage of a command.')
        except RuntimeError as e:
            if self.debug:
                print(f"DEBUG: Exception {secure_format_exception(e)}")
        finally:
            self.stopped = True
            self.api.close()

    def _get_login_creds(self):
        if self.credential_type == CredentialType.PASSWORD:
            self.user_name = "admin"
            self.pwd = hash_password("admin")
        else:
            self.user_name = input("User Name: ")

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
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
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
