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

import json
from typing import List

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.private.defs import InfoCollectorTopic
from nvflare.private.fed.server.admin import new_message
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import WidgetID
from .cmd_utils import CommandUtil


class InfoCollectorCommandModule(CommandModule, CommandUtil):
    """
    NOTE: we only support Server side info collector commands for now,
    due to the complexity of client-side process/child-process architecture.
    """

    CONN_KEY_COLLECTOR = "collector"

    def get_spec(self):
        return CommandModuleSpec(
            name="info",
            cmd_specs=[
                CommandSpec(
                    name=AdminCommandNames.SHOW_STATS,
                    description="show current system stats",
                    usage="show_stats",
                    handler_func=self.show_stats,
                    authz_func=self.authorize_info_collection,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.SHOW_ERRORS,
                    description="show latest errors",
                    usage="show_errors",
                    handler_func=self.show_errors,
                    authz_func=self.authorize_info_collection,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.RESET_ERRORS,
                    description="reset errors",
                    usage="reset_errors",
                    handler_func=self.reset_errors,
                    authz_func=self.authorize_info_collection,
                    visible=True,
                ),
            ],
        )

    def authorize_info_collection(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        assert isinstance(engine, ServerEngineInternalSpec)

        collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        if not collector:
            conn.append_error("info collector not available")
            return False, None

        if not isinstance(collector, InfoCollector):
            conn.append_error("system error: info collector not right object")
            return False, None

        conn.set_prop(self.CONN_KEY_COLLECTOR, collector)

        run_info = engine.get_run_info()
        if not run_info or run_info.run_number < 0:
            conn.append_string("App is not running")
            return False, None

        # return True, FLAuthzContext.new_authz_context(
        #     site_names=['server'],
        #     actions=[Action.VIEW])
        return self.authorize_view(conn, args)

    def show_stats(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        assert isinstance(engine, ServerEngineInternalSpec)

        target_type = args[1]
        if target_type == self.TARGET_TYPE_SERVER:
            collector = conn.get_prop(self.CONN_KEY_COLLECTOR)
            result = collector.get_run_stats()
            conn.append_any(result)
        elif target_type == self.TARGET_TYPE_CLIENT:
            message = new_message(conn, topic=InfoCollectorTopic.SHOW_STATS, body="")
            replies = self.send_request_to_clients(conn, message)
            self._process_stats_replies(conn, replies)

        # collector = conn.get_prop(self.CONN_KEY_COLLECTOR)
        # result = collector.get_run_stats()
        # conn.append_any(result)

    def show_errors(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        assert isinstance(engine, ServerEngineInternalSpec)

        target_type = args[1]
        if target_type == self.TARGET_TYPE_SERVER:
            collector = conn.get_prop(self.CONN_KEY_COLLECTOR)
            result = collector.get_errors()
            conn.append_any(result)
        elif target_type == self.TARGET_TYPE_CLIENT:
            message = new_message(conn, topic=InfoCollectorTopic.SHOW_ERRORS, body="")
            replies = self.send_request_to_clients(conn, message)
            self._process_stats_replies(conn, replies)

    def reset_errors(self, conn: Connection, args: List[str]):
        collector = conn.get_prop(self.CONN_KEY_COLLECTOR)
        collector.reset_errors()
        conn.append_string("errors reset")

    def _process_stats_replies(self, conn, replies):
        if not replies:
            conn.append_error("no responses from clients")
            return

        engine = conn.app_ctx
        for r in replies:
            client_name = engine.get_client_name_from_token(r.client_token)

            conn.append_string(f"--- Client ---: {client_name}")
            try:
                body = json.loads(r.reply.body)
                conn.append_any(body)
            except BaseException:
                conn.append_string("Bad responses from clients")
