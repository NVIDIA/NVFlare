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
from nvflare.fuel.hci.reg import CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
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
                    usage="show_stats job_id server|client",
                    handler_func=self.show_stats,
                    authz_func=self.authorize_info_collection,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.SHOW_ERRORS,
                    description="show latest errors in an actively running job",
                    usage="show_errors job_id server|client",
                    handler_func=self.show_errors,
                    authz_func=self.authorize_info_collection,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.RESET_ERRORS,
                    description="reset error stats for an actively running job",
                    usage="reset_errors",
                    handler_func=self.reset_errors,
                    authz_func=self.authorize_info_collection,
                    visible=True,
                ),
            ],
        )

    def authorize_info_collection(self, conn: Connection, args: List[str]):
        if len(args) != 3:
            conn.append_error("syntax error: missing job_id and target")
            return PreAuthzReturnCode.ERROR

        rt = self.authorize_job(conn, args)
        if rt == PreAuthzReturnCode.ERROR:
            return rt

        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        collector = engine.get_widget(WidgetID.INFO_COLLECTOR)
        if not collector:
            conn.append_error("info collector not available")
            return PreAuthzReturnCode.ERROR

        if not isinstance(collector, InfoCollector):
            conn.append_error("system error: info collector not right object")
            return PreAuthzReturnCode.ERROR

        conn.set_prop(self.CONN_KEY_COLLECTOR, collector)

        job_id = conn.get_prop(self.JOB_ID)
        if job_id not in engine.run_processes:
            conn.append_error(f"Job_id: {job_id} is not running.")
            return PreAuthzReturnCode.ERROR

        run_info = engine.get_app_run_info(job_id)
        if not run_info:
            conn.append_string(
                "Cannot find job: {}. Please make sure the first arg following the command is a valid job_id.".format(
                    job_id
                )
            )
            return PreAuthzReturnCode.ERROR
        return rt

    def show_stats(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        job_id = conn.get_prop(self.JOB_ID)
        target_type = args[2]
        if target_type == self.TARGET_TYPE_SERVER:
            result = engine.show_stats(job_id)
            conn.append_any(result)
        elif target_type == self.TARGET_TYPE_CLIENT:
            message = new_message(conn, topic=InfoCollectorTopic.SHOW_STATS, body="", require_authz=True)
            message.set_header(RequestHeader.JOB_ID, job_id)
            replies = self.send_request_to_clients(conn, message)
            self._process_stats_replies(conn, replies)

    def show_errors(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        job_id = conn.get_prop(self.JOB_ID)
        target_type = args[2]
        if target_type == self.TARGET_TYPE_SERVER:
            result = engine.get_errors(job_id)
            conn.append_any(result)
        elif target_type == self.TARGET_TYPE_CLIENT:
            message = new_message(conn, topic=InfoCollectorTopic.SHOW_ERRORS, body="", require_authz=True)
            message.set_header(RequestHeader.JOB_ID, job_id)
            replies = self.send_request_to_clients(conn, message)
            self._process_stats_replies(conn, replies)

    def reset_errors(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        job_id = conn.get_prop(self.JOB_ID)
        target_type = args[2]
        if target_type == self.TARGET_TYPE_SERVER:
            result = engine.reset_errors(job_id)
            conn.append_any(result)
        elif target_type == self.TARGET_TYPE_CLIENT:
            message = new_message(conn, topic=InfoCollectorTopic.RESET_ERRORS, body="", require_authz=True)
            message.set_header(RequestHeader.JOB_ID, job_id)
            replies = self.send_request_to_clients(conn, message)
            self._process_stats_replies(conn, replies)

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
            except Exception:
                conn.append_string("Bad responses from clients")
