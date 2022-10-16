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

import copy
import time
from abc import ABC, abstractmethod
from typing import List

from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey, ReservedKey, ServerCommandKey, ServerCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.fuel.utils import fobs
from nvflare.private.defs import SpecialTaskName, TaskConstant
from nvflare.widgets.widget import WidgetID

NO_OP_REPLY = "__no_op_reply"


class CommandProcessor(ABC):
    """The CommandProcessor is responsible for processing a command from parent process."""

    @abstractmethod
    def get_command_name(self) -> str:
        """Gets the command name that this processor will handle.

        Returns:
            name of the command
        """
        pass

    @abstractmethod
    def process(self, data: Shareable, fl_ctx: FLContext):
        """Processes the data.

        Args:
            data: process data
            fl_ctx: FLContext

        Return:
            A reply message
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
        if server_runner:
            server_runner.abort(fl_ctx)
            # wait for the runner process gracefully abort the run.
            time.sleep(3.0)
        return "Aborted the run"


class GetRunInfoCommand(CommandProcessor):
    """Implements the GET_RUN_INFO command."""

    def get_command_name(self) -> str:
        return ServerCommandNames.GET_RUN_INFO

    def process(self, data: Shareable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        run_info = engine.get_run_info()
        if run_info:
            return run_info
        return NO_OP_REPLY


class GetTaskCommand(CommandProcessor):
    """To implement the server GetTask command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: ServerCommandNames.GET_TASK

        """
        return ServerCommandNames.GET_TASK

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the abort command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: task data

        """

        shared_fl_ctx = data.get_header(ServerCommandKey.PEER_FL_CONTEXT)
        client = data.get_header(ServerCommandKey.FL_CLIENT)
        fl_ctx.set_peer_context(shared_fl_ctx)
        server_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if not server_runner:
            # this is possible only when the client request is received before the
            # server_app_runner.start_server_app is called in runner_process.py
            # We ask the client to try again later.
            taskname = SpecialTaskName.TRY_AGAIN
            task_id = ""
            shareable = Shareable()
            shareable.set_header(TaskConstant.WAIT_TIME, 1.0)
        else:
            taskname, task_id, shareable = server_runner.process_task_request(client, fl_ctx)
        data = {
            ServerCommandKey.TASK_NAME: taskname,
            ServerCommandKey.TASK_ID: task_id,
            ServerCommandKey.SHAREABLE: shareable,
            ServerCommandKey.FL_CONTEXT: copy.deepcopy(get_serializable_data(fl_ctx).props),
        }
        return fobs.dumps(data)


class SubmitUpdateCommand(CommandProcessor):
    """To implement the server GetTask command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: ServerCommandNames.SUBMIT_UPDATE

        """
        return ServerCommandNames.SUBMIT_UPDATE

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the abort command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns:

        """
        shareable = data.get(ReservedKey.SHAREABLE)
        shared_fl_ctx = data.get(ReservedKey.SHARED_FL_CONTEXT)
        client = shareable.get_header(ServerCommandKey.FL_CLIENT)
        fl_ctx.set_peer_context(shared_fl_ctx)
        contribution_task_name = shareable.get_header(ServerCommandKey.TASK_NAME)
        task_id = shareable.get_cookie(FLContextKey.TASK_ID)
        server_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        server_runner.process_submission(client, contribution_task_name, task_id, shareable, fl_ctx)

        return None


class AuxCommunicateCommand(CommandProcessor):
    """Server AuxCommunicate command."""

    def get_command_name(self) -> str:
        return ServerCommandNames.AUX_COMMUNICATE

    def process(self, data: Shareable, fl_ctx: FLContext):
        shared_fl_ctx = data.get_header(ServerCommandKey.PEER_FL_CONTEXT)
        topic = data.get_header(ServerCommandKey.TOPIC)
        shareable = data.get_header(ServerCommandKey.SHAREABLE)
        fl_ctx.set_peer_context(shared_fl_ctx)

        engine = fl_ctx.get_engine()
        reply = engine.dispatch(topic=topic, request=shareable, fl_ctx=fl_ctx)

        data = {
            ServerCommandKey.AUX_REPLY: reply,
            ServerCommandKey.FL_CONTEXT: copy.deepcopy(get_serializable_data(fl_ctx).props),
        }
        return data


class ShowStatsCommand(CommandProcessor):
    """To implement the show_stats command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: ServerCommandNames.SHOW_STATS

        """
        return ServerCommandNames.SHOW_STATS

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the abort command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: Engine run_info

        """
        engine = fl_ctx.get_engine()
        collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        return collector.get_run_stats()


class GetErrorsCommand(CommandProcessor):
    """To implement the show_errors command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: ServerCommandNames.GET_ERRORS

        """
        return ServerCommandNames.GET_ERRORS

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the abort command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: Engine run_info

        """
        engine = fl_ctx.get_engine()
        collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        errors = collector.get_errors()
        if not errors:
            errors = "No Error"
        return errors


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

    commands: List[CommandProcessor] = [
        AbortCommand(),
        ByeCommand(),
        GetRunInfoCommand(),
        GetTaskCommand(),
        SubmitUpdateCommand(),
        AuxCommunicateCommand(),
        ShowStatsCommand(),
        GetErrorsCommand(),
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
