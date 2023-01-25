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

import time

from nvflare.apis.fl_constant import (
    AdminCommandNames,
    FLContextKey,
    MachineStatus,
    ServerCommandKey,
    ServerCommandNames,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, simple_shareable
from nvflare.private.fed.cmd_agent import CommandProcessor
from nvflare.widgets.widget import WidgetID

NO_OP_REPLY = "__no_op_reply"


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
            engine = fl_ctx.get_engine()
            start_time = time.time()
            while engine.engine_info.status != MachineStatus.STOPPED:
                time.sleep(1.0)
                if time.time() - start_time > 30.0:
                    break
        return simple_shareable(data="Aborted the run")


class GetRunInfoCommand(CommandProcessor):
    """Implements the GET_RUN_INFO command."""

    def get_command_name(self) -> str:
        return ServerCommandNames.GET_RUN_INFO

    def process(self, data: Shareable, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        run_info = engine.get_run_info()
        if run_info:
            return simple_shareable(run_info)
        return simple_shareable(NO_OP_REPLY)


class HandleDeadJobCommand(CommandProcessor):
    """To implement the server HandleDeadJob command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: ServerCommandNames.SUBMIT_UPDATE

        """
        return ServerCommandNames.HANDLE_DEAD_JOB

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the HandleDeadJob command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns:

        """
        client_name = data.get_header(ServerCommandKey.FL_CLIENT)
        server_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        server_runner.handle_dead_job(client_name, fl_ctx)
        return ""


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
        return simple_shareable(collector.get_run_stats())


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
            errors = {}
        return simple_shareable(errors)
