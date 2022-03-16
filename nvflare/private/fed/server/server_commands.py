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

"""FL Admin commands."""

from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey, ServerCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class CommandProcessor(object):
    """The CommandProcessor is responsible for processing a command from parent process."""

    def get_command_name(self) -> str:
        """Get command name that this processor will handle.

        Returns: name of the command

        """
        pass

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the specified command.

        Args:
            data: process data
            fl_ctx: FLContext

        Return: reply message

        """
        pass


class AbortCommand(CommandProcessor):
    """To implement the abort command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: AdminCommandNames.ABORT

        """
        return AdminCommandNames.ABORT

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the abort command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: abort command message

        """
        server_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        server_runner.abort(fl_ctx)
        return "Aborted the run"


class GetRunInfoCommand(CommandProcessor):
    """To implement the abort command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: AdminCommandNames.ABORT

        """
        return ServerCommandNames.GET_RUN_INFO

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the abort command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: abort command message

        """
        engine = fl_ctx.get_engine()
        return engine.get_run_info()


class ByeCommand(CommandProcessor):
    """To implement the ShutdownCommand."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: AdminCommandNames.SHUTDOWN

        """
        return AdminCommandNames.SHUTDOWN

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the Shutdown command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: Shutdown command message

        """
        return None


class ServerCommands(object):
    """AdminCommands contains all the commands for processing the commands from the parent process."""

    commands = [
        AbortCommand(),
        ByeCommand(),
        GetRunInfoCommand(),
    ]

    @staticmethod
    def get_command(command_name):
        """Call to return the AdminCommand object.

        Args:
            command_name: AdminCommand name

        Returns: AdminCommand object

        """
        for command in ServerCommands.commands:
            if command_name == command.get_command_name():
                return command
        return None

    @staticmethod
    def register_command(command_processor: CommandProcessor):
        """Call to register the AdminCommand processor.

        Args:
            command_processor: AdminCommand processor

        """
        if not isinstance(command_processor, CommandProcessor):
            raise TypeError(
                "command_processor must be an instance of CommandProcessor, but got {}".format(type(command_processor))
            )

        ServerCommands.commands.append(command_processor)
