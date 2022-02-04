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

import json

import psutil

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.private.defs import SysCommandTopic
from nvflare.private.fed.server.admin import new_message
from nvflare.private.fed.server.cmd_utils import CommandUtil


class SystemCommandModule(CommandModule, CommandUtil):
    def get_spec(self):
        return CommandModuleSpec(
            name="sys",
            cmd_specs=[
                CommandSpec(
                    name="sys_info",
                    description="get the system info",
                    usage="sys_info server|client <client-name> ...",
                    handler_func=self.sys_info,
                    authz_func=self.authorize_operate,
                    visible=True,
                ),
            ],
        )

    def sys_info(self, conn: Connection, args: [str]):
        if len(args) < 2:
            conn.append_error("syntax error: missing site names")
            return

        target_type = args[1]
        if target_type == self.TARGET_TYPE_SERVER:
            infos = dict(psutil.virtual_memory()._asdict())

            table = conn.append_table(["Metrics", "Value"])

            for k, v in infos.items():
                table.add_row([str(k), str(v)])
            table.add_row(
                [
                    "available_percent",
                    "%.1f" % (psutil.virtual_memory().available * 100 / psutil.virtual_memory().total),
                ]
            )
            return

        if target_type == self.TARGET_TYPE_CLIENT:
            message = new_message(conn, topic=SysCommandTopic.SYS_INFO, body="")
            replies = self.send_request_to_clients(conn, message)
            self._process_replies(conn, replies)
            return

        conn.append_string("invalid target type {}. Usage: sys_info server|client <client-name>".format(target_type))

    def _process_replies(self, conn, replies):
        if not replies:
            conn.append_error("no responses from clients")
            return

        engine = conn.app_ctx
        for r in replies:
            client_name = engine.get_client_name_from_token(r.client_token)
            conn.append_string("Client: " + client_name)

            if r.reply:
                try:
                    infos = json.loads(r.reply.body)
                    table = conn.append_table(["Metrics", "Value"])

                    for k, v in infos.items():
                        table.add_row([str(k), str(v)])
                    table.add_row(
                        [
                            "available_percent",
                            "%.1f" % (psutil.virtual_memory().available * 100 / psutil.virtual_memory().total),
                        ]
                    )
                except BaseException:
                    conn.append_string(": Bad replies")
            else:
                conn.append_string(": No replies")
