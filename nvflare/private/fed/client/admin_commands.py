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

from nvflare.apis.fl_constant import FLContextKey, AdminCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, ReservedHeaderKey
from nvflare.private.fed.client.client_status import get_status_message
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import WidgetID


class CommandProcessor(object):
    """
    The CommandProcessor is responsible for processing a command from parent process.
    """

    def get_command_name(self) -> str:
        """
        Get command name that this processor will handle
        :return: name of the command
        """
        pass

    def process(self, data: Shareable, fl_ctx: FLContext):
        """
        Called to process the specified command
        :param data:
        :param fl_ctx:
        :return: command processing result
        """
        pass


class CheckStatusCommand(CommandProcessor):
    def get_command_name(self) -> str:
        return AdminCommandNames.CHECK_STATUS

    def process(self, data: Shareable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        federated_client = engine.client
        return get_status_message(federated_client.status)


class AbortCommand(CommandProcessor):
    def get_command_name(self) -> str:
        return AdminCommandNames.ABORT

    def process(self, data: Shareable, fl_ctx: FLContext):
        client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        return client_runner.abort()


class AbortTaskCommand(CommandProcessor):
    def get_command_name(self) -> str:
        return AdminCommandNames.ABORT_TASK

    def process(self, data: Shareable, fl_ctx: FLContext):
        client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if client_runner:
            client_runner.abort_task()
        return None


class ShowStatsCommand(CommandProcessor):
    def get_command_name(self) -> str:
        return AdminCommandNames.SHOW_STATS

    def process(self, data: Shareable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        if not collector:
            result = {"error": "no info collector"}
        else:
            assert isinstance(collector, InfoCollector)

            result = collector.get_run_stats()

        if not result:
            result = "No stats info"
        return result


class ShowErrorsCommand(CommandProcessor):
    def get_command_name(self) -> str:
        return AdminCommandNames.SHOW_ERRORS

    def process(self, data: Shareable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        if not collector:
            result = {"error": "no info collector"}
        else:
            assert isinstance(collector, InfoCollector)

            result = collector.get_errors()

        # CommandAgent is expecting data, could not be None
        if result is None:
            result = "No Errors"
        return result


class ResetErrorsCommand(CommandProcessor):
    def get_command_name(self) -> str:
        return AdminCommandNames.RESET_ERRORS

    def process(self, data: Shareable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        engine.reset_errors()


class AuxCommand(CommandProcessor):
    def get_command_name(self) -> str:
        return AdminCommandNames.AUX_COMMAND

    def process(self, data: Shareable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()

        topic = data.get_header(ReservedHeaderKey.TOPIC)
        return engine.dispatch(topic=topic, request=data, fl_ctx=fl_ctx)


class ByeCommand(CommandProcessor):
    def get_command_name(self) -> str:
        return AdminCommandNames.SHUTDOWN

    def process(self, data: Shareable, fl_ctx: FLContext):
        return None


class AdminCommands(object):
    """
    AdminCommands contains all the commands for processing the commands from the parent process.
    """
    commands = [
        CheckStatusCommand(),
        AbortCommand(),
        AbortTaskCommand(),
        ByeCommand(),
        ShowStatsCommand(),
        ShowErrorsCommand(),
        ResetErrorsCommand(),
        AuxCommand()
    ]

    @staticmethod
    def get_command(command_name):
        for command in AdminCommands.commands:
            if command_name == command.get_command_name():
                return command
        return None

    @staticmethod
    def register_command(command_processor: CommandProcessor):
        assert isinstance(command_processor, CommandProcessor)

        AdminCommands.commands.append(command_processor)
