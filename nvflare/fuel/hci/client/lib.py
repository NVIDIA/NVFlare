# Copyright (c) 2021, NVIDIA CORPORATION.
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
import socket
import ssl
import time
import traceback
from datetime import datetime

from nvflare.fuel.hci.cmd_arg_utils import join_args, split_to_args
from nvflare.fuel.hci.conn import Connection, receive_and_process
from nvflare.fuel.hci.proto import make_error
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandRegister, CommandSpec
from nvflare.fuel.hci.security import get_certificate_common_name, hash_password, verify_password
from nvflare.fuel.hci.table import Table


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


class ReplyProcessor(object):
    def reply_start(self, client, reply_json):
        pass

    def process_string(self, client, item: str):
        pass

    def process_success(self, client, item: str):
        pass

    def process_error(self, client, err: str):
        pass

    def process_table(self, client, table: Table):
        pass

    def process_dict(self, client, data: dict):
        pass

    def process_shutdown(self, client, msg: str):
        pass

    def process_token(self, client, token: str):
        pass

    def protocol_error(self, client, err: str):
        pass

    def reply_done(self, client):
        pass


class _RegularReplyProcessor(ReplyProcessor):
    def process_shutdown(self, client, msg: str):
        client.shutdown_received = True
        client.shutdown_msg = msg

    def process_string(self, client, item: str):
        client.write_string(item)

    def process_error(self, client, err: str):
        client.write_error(err)

    def process_table(self, client, table: Table):
        client.write_table(table)

    def process_dict(self, client, data: dict):
        client.write_dict(data)

    def protocol_error(self, client, err: str):
        client.write_error(err)

    def reply_done(self, client):
        client.flush()


class _LoginReplyProcessor(ReplyProcessor):
    def reply_start(self, client, reply_json):
        client.login_result = "error"

    def process_string(self, client, item: str):
        client.login_result = item

    def process_token(self, client, token: str):
        client.token = token


class _CmdListReplyProcessor(ReplyProcessor):
    def reply_start(self, client, reply_json):
        client.server_cmd_received = False

    def process_table(self, client, table: Table):
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

            if confirm == "auth" and not client.require_login:
                # the user is not authenticated - skip this command
                continue

            client.server_cmd_reg.add_command(
                scope_name=scope,
                cmd_name=cmd_name,
                desc=desc,
                usage=usage,
                handler=None,
                authz_func=None,
                visible=True,
                confirm=confirm,
            )

        client.server_cmd_received = True
        client.server_cmd_reg.finalize(client.register_command)


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
        require_login: whether or not to require login
        credential_type: what type of credential to use
        cmd_modules: command modules to load and register
        debug: Whether or not to print debug messages. False by default.
    """

    def __init__(
        self,
        host,
        port,
        prompt="> ",
        ca_cert=None,
        client_cert=None,
        client_key=None,
        server_cn=None,
        require_login=False,
        credential_type="password",
        cmd_modules=None,
        debug=False,
    ):
        cmd.Cmd.__init__(self)
        self.intro = "Type help or ? to list commands.\n"
        self.prompt = prompt
        self.host = host
        self.port = port
        self.ca_cert = ca_cert
        self.client_cert = client_cert
        self.client_key = client_key
        self.server_cn = server_cn
        self.require_login = require_login
        self.credential_type = credential_type
        self.user_name = None
        self.pwd = None
        self.token = None
        self.login_result = None
        self.server_cmd_reg = CommandRegister(app_ctx=self)
        self.client_cmd_reg = CommandRegister(app_ctx=self)
        self.server_cmd_received = False
        self.debug = debug
        self.shutdown_received = False
        self.shutdown_msg = None
        self.reply_processor = None
        self.all_cmds = []
        self.out_file = None
        self.no_stdout = False

        assert credential_type in ["password", "cert"], "invalid credential_type {}".format(credential_type)

        self.client_cmd_reg.register_module(_BuiltInCmdModule())
        if cmd_modules:
            assert isinstance(cmd_modules, list), "cmd_modules must be a list"
            for m in cmd_modules:
                assert isinstance(m, CommandModule), "cmd_modules must be a list of CommandModule"
                self.client_cmd_reg.register_module(m, include_invisible=False)

        self.client_cmd_reg.finalize(self.register_command)

    def _set_output_file(self, file, no_stdout):
        self._close_output_file()
        self.out_file = file
        self.no_stdout = no_stdout

    def _close_output_file(self):
        if self.out_file:
            self.out_file.close()
            self.out_file = None
        self.no_stdout = False

    def register_command(self, cmd_entry):
        self.all_cmds.append(cmd_entry.name)

    def do_bye(self, arg):
        "exit from the client"
        if self.require_login:
            self.server_execute("_logout")
        return True

    def do_lpwd(self, arg):
        "print local current work dir"
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
            self._show_commands(self.client_cmd_reg)

            self.write_string("\nServer Commands")
            self._show_commands(self.server_cmd_reg)
        else:
            server_cmds = []
            local_cmds = []
            parts = arg.split()
            for p in parts:
                entries = self.client_cmd_reg.get_command_entries(p)
                if len(entries) > 0:
                    local_cmds.append(p)

                entries = self.server_cmd_reg.get_command_entries(p)
                if len(entries) > 0:
                    server_cmds.append(p)

            if len(local_cmds) > 0:
                self.write_string("Client Commands")
                self.write_string("---------------")
                for cmd_name in local_cmds:
                    self._show_one_command(cmd_name, self.client_cmd_reg)

            if len(server_cmds) > 0:
                self.write_string("Server Commands")
                self.write_string("---------------")
                for cmd_name in server_cmds:
                    self._show_one_command(cmd_name, self.server_cmd_reg)

    def complete(self, text, state):
        results = [x + " " for x in self.all_cmds if x.startswith(text)] + [None]
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
        entries = self.client_cmd_reg.get_command_entries(cmd_name)
        if len(entries) > 1:
            self.write_string("Ambiguous client command {} - qualify with scope".format(cmd_name))
            return
        elif len(entries) == 1:
            ent = entries[0]
            return ent.handler(args, self.client_cmd_reg.app_ctx)

        entries = self.server_cmd_reg.get_command_entries(cmd_name)
        if len(entries) <= 0:
            self.write_string("Undefined server command {}".format(cmd_name))
            return
        elif len(entries) > 1:
            self.write_string("Ambiguous server command {} - qualify with scope".format(cmd_name))
            return

        ent = entries[0]
        confirm_method = ent.confirm
        if ent.confirm == "auth":
            if self.credential_type == "password":
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

        self.server_execute(line)
        if self.shutdown_received:
            # exit the client
            self.write_string(self.shutdown_msg)
            return True

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
        if self.debug:
            print("Server Reply: {}".format(resp))

        reply_processor = self.reply_processor
        if reply_processor is None:
            reply_processor = _RegularReplyProcessor()

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
                else:
                    reply_processor.protocol_error(self, "Invalid item type: " + it)
        else:
            reply_processor.protocol_error(self, "Protocol Error")

        reply_processor.reply_done(self)

    def _try_command(self, command, reply_processor):
        self.reply_processor = reply_processor
        process_json_func = self._process_server_reply
        try:
            if self.ca_cert and self.client_cert:
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
                # Non-SSL
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((self.host, self.port))
                    self._send_to_sock(sock, command, process_json_func)
        except Exception as ex:
            if self.debug:
                traceback.print_exc()

            process_json_func(
                make_error("Failed to communicate with Admin Server {} on {}: {}".format(self.host, self.port, ex))
            )

    def server_execute(self, command, reply_processor=None):
        start = time.time()
        self._try_command(command, reply_processor)
        secs = time.time() - start
        usecs = int(secs * 1000000)
        done = "Done [{} usecs] {}".format(usecs, datetime.now())
        self.write_stdout(done)

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
                        except EOFError:
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
        if self.require_login:
            user_name = input("User Name: ")

            if self.credential_type == "password":
                while True:
                    pwd = getpass.getpass("Password: ")
                    self._try_command("_login {} {}".format(user_name, pwd), _LoginReplyProcessor())
                    if self.login_result == "OK":
                        self.user_name = user_name
                        self.pwd = hash_password(pwd)
                        break
                    elif self.login_result == "REJECT":
                        print("Incorrect password - please try again")
                    else:
                        print("Communication Error - please try later")
                        return
            elif self.credential_type == "cert":
                self._try_command("_cert_login {}".format(user_name), _LoginReplyProcessor())
                if self.login_result == "OK":
                    self.user_name = user_name
                elif self.login_result == "REJECT":
                    print("Incorrect user name or certificate")
                    return
                else:
                    print("Communication Error - please try later")
                    return

        # get command list from server
        self._try_command("_commands", _CmdListReplyProcessor())
        if not self.server_cmd_received:
            print("Failed to handshake with server - please try later")
            return

        self.cmdloop(intro='Type ? to list commands; type "? cmdName" to show usage of a command.')

    def write_stdout(self, data: str):
        self.stdout.write(data + "\n")

    def write_string(self, data: str):
        content = data + "\n"
        if not self.no_stdout:
            self.stdout.write(content)
        if self.out_file:
            self.out_file.write(content)

    def write_table(self, table: Table):
        if not self.no_stdout:
            table.write(self.stdout)
        if self.out_file:
            table.write(self.out_file)

    def write_dict(self, data: dict):
        content = json.dumps(data, indent=2) + "\n"
        if not self.no_stdout:
            self.stdout.write(content)
        if self.out_file:
            self.out_file.write(content)

    def write_error(self, err: str):
        content = "Error: " + err + "\n"
        if not self.no_stdout:
            self.stdout.write(content)
        if self.out_file:
            self.out_file.write(content)

    def flush(self):
        if not self.no_stdout:
            self.stdout.flush()
        if self.out_file:
            self.out_file.flush()
