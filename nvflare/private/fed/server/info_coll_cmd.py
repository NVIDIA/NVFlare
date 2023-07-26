# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.fuel.hci.proto import MetaStatusValue, make_meta
from nvflare.fuel.hci.reg import CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.defs import InfoCollectorTopic, RequestHeader
from nvflare.private.fed.server.admin import new_message
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec
from nvflare.widgets.info_collector import InfoCollector
from nvflare.widgets.widget import WidgetID

from .cmd_utils import CommandUtil
from .job_cmds import JobCommandModule


class InfoCollectorCommandModule(JobCommandModule, CommandUtil):
    """This class is for server side info collector commands.

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
                    description="show current system stats for an actively running job",
                    usage="show_stats job_id server|client [clients]",
                    handler_func=self.show_stats,
                    authz_func=self.authorize_info_collection,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.SHOW_ERRORS,
                    description="show latest errors in an actively running job",
                    usage="show_errors job_id server|client [clients]",
                    handler_func=self.show_errors,
                    authz_func=self.authorize_info_collection,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.RESET_ERRORS,
                    description="reset error stats for an actively running job",
                    usage="reset_errors job_id server|client [clients]",
                    handler_func=self.reset_errors,
                    authz_func=self.authorize_info_collection,
                    visible=True,
                ),
            ],
        )

    def authorize_info_collection(self, conn: Connection, args: List[str]):
        if len(args) < 3:
            cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
            conn.append_error(f"Usage: {cmd_entry.usage}", meta=make_meta(MetaStatusValue.SYNTAX_ERROR))
            return PreAuthzReturnCode.ERROR

        rt = self.authorize_job(conn, args)
        if rt == PreAuthzReturnCode.ERROR:
            return rt

        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        if not collector:
            msg = "info collector not available"
            conn.append_error(msg, meta=make_meta(MetaStatusValue.INTERNAL_ERROR, msg))
            return PreAuthzReturnCode.ERROR

        if not isinstance(collector, InfoCollector):
            msg = "info collector not right object"
            conn.append_error(msg, meta=make_meta(MetaStatusValue.INTERNAL_ERROR, msg))
            return PreAuthzReturnCode.ERROR

        conn.set_prop(self.CONN_KEY_COLLECTOR, collector)

        job_id = conn.get_prop(self.JOB_ID)
        if job_id not in engine.run_processes:
            conn.append_error(
                f"Job_id: {job_id} is not running.", meta=make_meta(MetaStatusValue.JOB_NOT_RUNNING, job_id)
            )
            return PreAuthzReturnCode.ERROR

        run_info = engine.get_app_run_info(job_id)
        if not run_info:
            conn.append_string(
                f"Cannot find job: {job_id}. Please make sure the first arg following the command is a valid job_id.",
                meta=make_meta(MetaStatusValue.INVALID_JOB_ID, job_id),
            )
            return PreAuthzReturnCode.ERROR
        return rt

    def show_stats(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        self._collect_stats(conn, args, stats_func=engine.show_stats, msg_topic=InfoCollectorTopic.SHOW_STATS)

    def _collect_stats(self, conn: Connection, args: List[str], stats_func, msg_topic):
        job_id = conn.get_prop(self.JOB_ID)
        target_type = args[2]
        result = {}
        if target_type in [self.TARGET_TYPE_SERVER, self.TARGET_TYPE_ALL]:
            server_stats = stats_func(job_id)
            result["server"] = server_stats

        if target_type in [self.TARGET_TYPE_CLIENT, self.TARGET_TYPE_ALL]:
            message = new_message(conn, topic=msg_topic, body="", require_authz=True)
            message.set_header(RequestHeader.JOB_ID, job_id)
            replies = self.send_request_to_clients(conn, message)
            self._process_stats_replies(conn, replies, result)
        conn.append_any(result)

    def show_errors(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        self._collect_stats(conn, args, stats_func=engine.get_errors, msg_topic=InfoCollectorTopic.SHOW_ERRORS)

    def reset_errors(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        self._collect_stats(conn, args, stats_func=engine.reset_errors, msg_topic=InfoCollectorTopic.RESET_ERRORS)

    @staticmethod
    def _process_stats_replies(conn, replies, result: dict):
        if not replies:
            return

        for r in replies:
            client_name = r.client_name
            try:
                body = json.loads(r.reply.body)
                result[client_name] = body
            except Exception:
                result[client_name] = "invalid_reply"
                return
