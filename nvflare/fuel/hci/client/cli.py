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
import json
import os
import signal
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import readline
except ImportError:
    readline = None

from nvflare.fuel.hci.cmd_arg_utils import join_args, parse_command_line
from nvflare.fuel.hci.proto import ProtoKey
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandRegister, CommandSpec
from nvflare.fuel.hci.table import Table
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .api import AdminAPI, CommandInfo
from .api_spec import AdminConfigKey, UidSource
from .api_status import APIStatus
from .event import EventContext, EventHandler, EventPropKey, EventType


class _SessionClosed(Exception):
    pass


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
                CommandSpec(
                    name="timeout", description="set/show command timeout", usage="timeout [value]", handler_func=None
                ),
            ],
        )


class AdminClient(cmd.Cmd, EventHandler):
    """Admin command prompt for submitting admin commands to the server through the CLI.

    Args:
        cmd_modules: command modules to load and register
        debug: whether to print debug messages. False by default.
        cli_history_size: the maximum number of commands to save in the cli history file. Defaults to 1000.
    """

    def __init__(
        self,
        admin_config: dict,
        cmd_modules: Optional[List] = None,
        debug: bool = False,
        username: str = "",
        handlers=None,
        cli_history_dir: str = str(Path.home() / ".nvflare"),
        cli_history_size: int = 1000,
    ):
        super().__init__()
        self.intro = "Type help or ? to list commands.\n"
        self.prompt = admin_config.get(AdminConfigKey.PROMPT, "> ")
        self.user_name = username
        self.debug = debug
        self.out_file = None
        self.no_stdout = False
        self.stopped = False  # use this flag to prevent unnecessary signal exception
        self.login_timeout = admin_config.get(AdminConfigKey.LOGIN_TIMEOUT)
        self.idle_timeout = admin_config.get(AdminConfigKey.IDLE_TIMEOUT, 900.0)
        self.last_active_time = time.time()

        if not cli_history_dir:
            raise Exception("missing cli_history_dir")

        modules = [_BuiltInCmdModule()]
        if cmd_modules:
            if not isinstance(cmd_modules, list):
                raise TypeError("cmd_modules must be a list.")
            for m in cmd_modules:
                if not isinstance(m, CommandModule):
                    raise TypeError("cmd_modules must be a list of CommandModule")
                modules.append(m)

        uid_source = admin_config.get(AdminConfigKey.UID_SOURCE, UidSource.USER_INPUT)
        if uid_source != UidSource.CERT:
            self.user_name = self._user_input("User Name: ")

        event_handlers = [self]
        if handlers:
            event_handlers.extend(handlers)

        self.api = AdminAPI(
            admin_config=admin_config,
            cmd_modules=modules,
            user_name=self.user_name,
            debug=self.debug,
            event_handlers=event_handlers,
        )

        if not os.path.isdir(cli_history_dir):
            os.mkdir(cli_history_dir)
        self.cli_history_file = os.path.join(cli_history_dir, ".admin_cli_history")

        if readline:
            readline.set_history_length(cli_history_size)

        # signal.signal(signal.SIGUSR1, partial(self.session_signal_handler))
        signal.signal(signal.SIGUSR1, self.session_signal_handler)

    def _monitor_user(self):
        while True:
            if time.time() - self.last_active_time > self.idle_timeout:
                # user has been idle for too long
                print(f"Logging out due to inactivity for {self.idle_timeout} seconds")
                self.api.logout()
                # We force to stop by killing the process because there is no way to interrupt the
                # command thread that is waiting for user input.
                # This "kill" will cause self.session_signal_handler to be executed in the thread that
                # runs the cmdloop!
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            else:
                time.sleep(1.0)

    def handle_event(self, event_type: str, ctx: EventContext):
        if self.debug:
            print(f"DEBUG: received session event: {event_type}")

        msg = ctx.get_prop(EventPropKey.MSG)
        if msg:
            self.write_string(msg)

        if event_type in [EventType.SESSION_TIMEOUT]:
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
        raise _SessionClosed("Session Closed")

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
        self.api.logout()
        return True

    def do_lpwd(self, arg):
        """print local current work dir"""
        self.write_string(os.getcwd())

    def do_timeout(self, arg):
        if not arg:
            # display current setting
            t = self.api.cmd_timeout
            if t:
                self.write_string(str(t))
            else:
                self.write_string("not set")
            return
        try:
            t = float(arg)
            self.api.set_command_timeout(t)
            if t == 0:
                self.write_string("command timeout is unset")
            else:
                self.write_string(f"command timeout is set to {t}")
        except:
            self.write_string("invalid timeout value - must be float number >= 0.0")

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
            traceback.print_exc()
            if self.debug:
                secure_log_traceback()
            self.write_stdout(f"exception occurred: {secure_format_exception(e)}")
        self._close_output_file()

    def _user_input(self, prompt: str) -> str:
        answer = input(prompt)
        self.last_active_time = time.time()
        # remove leading and trailing spaces
        return answer.strip()

    def _do_default(self, line):
        line, args, props = parse_command_line(line)
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
            if self.user_name:
                info = CommandInfo.CONFIRM_USER_NAME
            else:
                info = CommandInfo.CONFIRM_YN

        if info == CommandInfo.CONFIRM_YN:
            answer = self._user_input("Are you sure (y/N): ")
            answer = answer.lower()
            if answer != "y" and answer != "yes":
                return
        elif info == CommandInfo.CONFIRM_USER_NAME:
            answer = self._user_input("Confirm with User Name: ")
            if answer != self.user_name:
                self.write_string("user name mismatch")
                return

        # execute the command!
        start = time.time()
        resp = self.api.do_command(line, props=props)
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

    def preloop(self):
        if readline and os.path.exists(self.cli_history_file):
            try:
                readline.read_history_file(self.cli_history_file)
            except OSError:
                self.stdout.write("Unable to read history file.  No command history loaded\n")
                readline.clear_history()

    def postcmd(self, stop, line):
        if readline:
            readline.write_history_file(self.cli_history_file)
        return stop

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
                            line = self._user_input(self.prompt)
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
            self.api.connect(self.login_timeout)
            self.api.login()
            self.last_active_time = time.time()
            monitor = threading.Thread(target=self._monitor_user, daemon=True)
            monitor.start()
            self.cmdloop(intro='Type ? to list commands; type "? cmdName" to show usage of a command.')
        except _SessionClosed as e:
            # this is intentional session closing (by server or client)
            if self.debug:
                print(f"Session closed: {secure_format_exception(e)}")
        except Exception as e:
            print(f"Exception {secure_format_exception(e)}")
        finally:
            self.stopped = True
            self.api.close()

    def print_resp(self, resp: dict):
        """Prints the server response

        Args:
            resp (dict): The server response.
        """
        if ProtoKey.DETAILS in resp:
            details = resp[ProtoKey.DETAILS]
            if isinstance(details, str):
                self.write_string(details)
            elif isinstance(details, Table):
                self.write_table(details)

        if ProtoKey.DATA in resp:
            for item in resp[ProtoKey.DATA]:
                if not isinstance(item, dict):
                    continue
                item_type = item.get(ProtoKey.TYPE)
                item_data = item.get(ProtoKey.DATA)
                if item_type == ProtoKey.STRING:
                    self.write_string(item_data)
                elif item_type == ProtoKey.TABLE:
                    table = Table(None)
                    table.set_rows(item[ProtoKey.ROWS])
                    self.write_table(table)
                elif item_type == ProtoKey.ERROR:
                    self.write_error(item_data)
                elif item_type == ProtoKey.DICT:
                    self.write_dict(item_data)

        if ProtoKey.DETAILS not in resp and ProtoKey.DATA not in resp:
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
