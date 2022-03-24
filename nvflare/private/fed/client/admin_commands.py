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

from abc import ABC, abstractmethod

from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.private.fed.client.client_run_manager import ClientRunManager
from nvflare.private.fed.client.client_status import get_status_message
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import WidgetID


class CommandProcessor(ABC):
    """The CommandProcessor is responsible for processing a command from parent process."""

    @abstractmethod
    def get_command_name(self) -> str:
        """Get command name.

        Returns:
            name of the command
        """
        pass

    @abstractmethod
    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the specified command.

        Args:
            data: process data
            fl_ctx: FLContext

        Return:
            result
        """
        pass


class CheckStatusCommand(CommandProcessor):
    """Check status command."""

    def get_command_name(self) -> str:
        return AdminCommandNames.CHECK_STATUS

    def process(self, data: Shareable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        federated_client = engine.client
        return get_status_message(federated_client.status)


class AbortCommand(CommandProcessor):
    """Abort command."""

    def get_command_name(self) -> str:
        return AdminCommandNames.ABORT

    def process(self, data: Shareable, fl_ctx: FLContext):
        client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if client_runner:
            client_runner.abort()


class AbortTaskCommand(CommandProcessor):
    """Implement abort_task command."""

    def get_command_name(self) -> str:
        return AdminCommandNames.ABORT_TASK

    def process(self, data: Shareable, fl_ctx: FLContext):
        client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if client_runner:
            client_runner.abort_task()


class ShowStatsCommand(CommandProcessor):
    """Implement the show_stats command."""

    def get_command_name(self) -> str:
        return AdminCommandNames.SHOW_STATS

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Process the show_stats command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns:
            A dict of show_stats result.
        """
        engine = fl_ctx.get_engine()
        collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        if not collector:
            result = {"error": "no info collector"}
        else:
            if not isinstance(collector, InfoCollector):
                raise TypeError("collector must be an instance of InfoCollector, but got {}".format(type(collector)))
            result = collector.get_run_stats()

        if not result:
            result = "No stats info"
        return result


class ShowErrorsCommand(CommandProcessor):
    """Implement the show_errors command."""

    def get_command_name(self) -> str:
        return AdminCommandNames.SHOW_ERRORS

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the show_errors command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns:
            A dict of show_errors result.
        """
        engine = fl_ctx.get_engine()
        collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        if not collector:
            result = {"error": "no info collector"}
        else:
            if not isinstance(collector, InfoCollector):
                raise TypeError("collector must be an instance of InfoCollector, but got {}".format(type(collector)))

            result = collector.get_errors()

        # CommandAgent is expecting data, could not be None
        if not result:
            result = "No Errors"
        return result


class ResetErrorsCommand(CommandProcessor):
    """Implement the reset_errors command."""

    def get_command_name(self) -> str:
        return AdminCommandNames.RESET_ERRORS

    def process(self, data: Shareable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        engine.reset_errors()


class AuxCommand(CommandProcessor):
    """Implement the Aux communication command."""

    def get_command_name(self) -> str:
        return AdminCommandNames.AUX_COMMAND

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the Aux communication command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: Aux communication command message

        """
        engine = fl_ctx.get_engine()
        if not isinstance(engine, ClientRunManager):
            raise TypeError("engine must be an instance of ClientRunManager, but got {}".format(type(engine)))

        topic = data.get_header(ReservedHeaderKey.TOPIC)
        return engine.dispatch(topic=topic, request=data, fl_ctx=fl_ctx)


class ShutdownCommand(CommandProcessor):
    """Implement the ShutdownCommand."""

    def get_command_name(self) -> str:
        return AdminCommandNames.SHUTDOWN

    def process(self, data: Shareable, fl_ctx: FLContext):
        return None


class AdminCommands(object):
    """AdminCommands contains all the commands for processing the commands from the parent process."""

    commands = [
        CheckStatusCommand(),
        AbortCommand(),
        AbortTaskCommand(),
        ShutdownCommand(),
        ShowStatsCommand(),
        ShowErrorsCommand(),
        ResetErrorsCommand(),
        AuxCommand(),
    ]

    @staticmethod
    def get_command(command_name):
        """Call to return the AdminCommand object.

        Args:
            command_name: AdminCommand name

        Returns: AdminCommand object

        """
        for command in AdminCommands.commands:
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

        AdminCommands.commands.append(command_processor)
