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

from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.private.fed.client.client_status import get_status_message
from nvflare.private.fed.cmd_agent import CommandProcessor
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import WidgetID


class CheckStatusCommand(CommandProcessor):
    """To implement the check_status command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: AdminCommandNames.CHECK_STATUSv

        """
        return AdminCommandNames.CHECK_STATUS

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the check_status command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: status message

        """
        engine = fl_ctx.get_engine()
        federated_client = engine.client
        return get_status_message(federated_client.status)


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
        client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        return client_runner.abort()


class AbortTaskCommand(CommandProcessor):
    """To implement the abort_task command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: AdminCommandNames.ABORT_TASK

        """
        return AdminCommandNames.ABORT_TASK

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the abort_task command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: abort_task command message

        """
        client_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if client_runner:
            client_runner.abort_task()
        return None


class ShowStatsCommand(CommandProcessor):
    """To implement the show_stats command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: AdminCommandNames.SHOW_STATS

        """
        return AdminCommandNames.SHOW_STATS

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the abort_task command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: show_stats command message

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
    """To implement the show_errors command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: AdminCommandNames.SHOW_ERRORS

        """
        return AdminCommandNames.SHOW_ERRORS

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the show_errors command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: show_errors command message

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
        if result is None:
            result = "No Errors"
        return result


class ResetErrorsCommand(CommandProcessor):
    """To implement the reset_errors command."""

    def get_command_name(self) -> str:
        """To get the command name.

        Returns: AdminCommandNames.RESET_ERRORS

        """
        return AdminCommandNames.RESET_ERRORS

    def process(self, data: Shareable, fl_ctx: FLContext):
        """Called to process the reset_errors command.

        Args:
            data: process data
            fl_ctx: FLContext

        Returns: reset_errors command message

        """
        engine = fl_ctx.get_engine()
        engine.reset_errors()
