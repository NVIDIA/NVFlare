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

import traceback
from typing import List

from nvflare.fuel.hci.cmd_arg_utils import split_to_args
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandRegister

from .constants import ConnProps


class CommandFilter(object):
    """Base class for filters to run before or after commands."""

    def pre_command(self, conn: Connection, args: List[str]) -> bool:
        """Code to execute before executing a command.

        Returns: True to continue filter chain or False to not
        """
        return True

    def post_command(self, conn: Connection, args: List[str]) -> bool:
        """Code to execute after executing a command."""
        pass

    def close(self):
        pass


class ServerCommandRegister(CommandRegister):
    def __init__(self, app_ctx):
        """Runs filters and executes commands by calling their handler function.

        This is the main command register used by AdminServer.

        Args:
            app_ctx: app context
        """
        CommandRegister.__init__(self, app_ctx)
        self.filters = []
        self.closed = False

    def add_filter(self, cmd_filter: CommandFilter):
        assert isinstance(cmd_filter, CommandFilter), "cmd_filter must be CommandFilter but got {}.".format(
            type(cmd_filter)
        )
        self.filters.append(cmd_filter)

    def _do_command(self, conn: Connection, command: str):
        """Executes command.

        Getting the command from the command registry, invoke filters and call the handler function, passing along conn
        and the args split from the command.
        """
        conn.app_ctx = self.app_ctx
        args = split_to_args(command)
        conn.args = args
        conn.command = command

        cmd_name = args[0]
        entries = self.get_command_entries(cmd_name)
        if len(entries) <= 0:
            conn.append_error('Unknown command "{}"'.format(cmd_name))
            return
        elif len(entries) == 1:
            conn.set_prop(ConnProps.CMD_ENTRY, entries[0])
            handler = entries[0].handler
        else:
            conn.append_error('Command "{}" exists in multiple scopes. Please use full command name'.format(cmd_name))
            return

        if handler is None:
            conn.append_error('Unknown command "{}"'.format(cmd_name))
            return

        # invoke pre filters
        if len(self.filters) > 0:
            for f in self.filters:
                ok = f.pre_command(conn, args)
                if not ok:
                    return

        if handler is not None:
            handler(conn, args)
        else:
            conn.append_error('Unknown command "{}"'.format(command))
            return

        # invoke post filters
        if len(self.filters) > 0:
            for f in self.filters:
                f.post_command(conn, args)

    def process_command(self, conn: Connection, command: str):
        try:
            self._do_command(conn, command)
        except BaseException as e:
            traceback.print_exc()
            conn.append_error(f"Exception Occurred: {e}")

    def close(self):
        if self.closed:
            return

        for f in self.filters:
            f.close()

        for m in self.modules:
            m.close()

        self.closed = True
