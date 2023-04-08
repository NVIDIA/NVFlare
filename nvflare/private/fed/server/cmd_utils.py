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

from typing import List

from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaStatusValue, make_meta
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server.admin import FedAdminServer


class CommandUtil(object):

    TARGET_CLIENTS = "target_clients"
    TARGET_CLIENT_TOKENS = "target_client_tokens"
    TARGET_CLIENT_NAMES = "target_client_names"
    TARGET_TYPE = "target_type"

    TARGET_TYPE_CLIENT = "client"
    TARGET_TYPE_SERVER = "server"
    TARGET_TYPE_ALL = "all"

    JOB_ID = "job_id"
    JOB = "job"

    def command_authz_required(self, conn: Connection, args: List[str]) -> PreAuthzReturnCode:
        return PreAuthzReturnCode.REQUIRE_AUTHZ

    def authorize_client_operation(self, conn: Connection, args: List[str]) -> PreAuthzReturnCode:
        auth_args = [args[0], self.TARGET_TYPE_CLIENT]
        auth_args.extend(args[1:])

        err = self.validate_command_targets(conn, auth_args[1:])
        if err:
            conn.append_error(err)
            return PreAuthzReturnCode.ERROR

        return PreAuthzReturnCode.REQUIRE_AUTHZ

    def authorize_abort_client_task(self, conn: Connection, args: List[str]) -> PreAuthzReturnCode:
        if len(args) < 2:
            conn.append_error("missing job_id (syntax is: abort_task job_id <client-name>)")
            return PreAuthzReturnCode.ERROR

        auth_args = [args[0], self.TARGET_TYPE_CLIENT]
        auth_args.extend(args[2:])

        job_id = args[1].lower()
        conn.set_prop(self.JOB_ID, job_id)

        err = self.validate_command_targets(conn, auth_args[1:])
        if err:
            conn.append_error(err)
            return PreAuthzReturnCode.ERROR

        return PreAuthzReturnCode.REQUIRE_AUTHZ

    def validate_command_targets(self, conn: Connection, args: List[str]) -> str:
        """Validate specified args and determine and set target type and target names in the Connection.

        The args must be like this:

            target_type client_names ...

        where target_type is one of 'all', 'client', 'server'

        Args:
            conn: A Connection object.
            args: Specified arguments.

        Returns:
            An error message. It is empty "" if no error found.
        """
        # return target type and a list of target names
        if len(args) < 1:
            return "missing target type (server or client)"

        target_type = args[0]
        conn.set_prop(self.TARGET_TYPE, target_type)

        if target_type == self.TARGET_TYPE_SERVER:
            return ""

        if target_type == self.TARGET_TYPE_CLIENT:
            client_names = args[1:]
        elif target_type == self.TARGET_TYPE_ALL:
            client_names = []
        else:
            return "unknown target type {}".format(target_type)

        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineSpec):
            raise TypeError("engine must be ServerEngineSpec but got {}".format(type(engine)))
        if len(client_names) == 0:
            # get all clients
            clients = engine.get_clients()
        else:
            clients, invalid_inputs = engine.validate_targets(client_names)
            if invalid_inputs:
                return "invalid client(s): {}".format(" ".join(invalid_inputs))

        if target_type == self.TARGET_TYPE_CLIENT and not clients:
            return "no clients available"

        valid_tokens = []
        client_names = []
        all_clients = {}
        for c in clients:
            valid_tokens.append(c.token)
            client_names.append(c.name)
            all_clients[c.token] = c.name
        conn.set_prop(self.TARGET_CLIENT_TOKENS, valid_tokens)
        # if clients:
        #     client_names = [c.name for c in clients]
        # else:
        #     client_names = []
        conn.set_prop(self.TARGET_CLIENT_NAMES, client_names)
        conn.set_prop(self.TARGET_CLIENTS, all_clients)
        return ""

    def must_be_project_admin(self, conn: Connection, args: List[str]):
        if args[0] == AdminCommandNames.ADMIN_CHECK_STATUS and len(args) == 1:
            args.extend(["server"])
        role = conn.get_prop(ConnProps.USER_ROLE, "")
        if role != "project_admin":
            conn.append_error("Not authorized as project_admin.", meta=make_meta(MetaStatusValue.NOT_AUTHORIZED))
            return PreAuthzReturnCode.ERROR
        else:
            return PreAuthzReturnCode.OK

    def authorize_server_operation(self, conn: Connection, args: List[str]):
        err = self.validate_command_targets(conn, args[1:])
        if err:
            conn.append_error(err)
            return PreAuthzReturnCode.ERROR

        target_type = conn.get_prop(self.TARGET_TYPE)
        if target_type == self.TARGET_TYPE_SERVER or target_type == self.TARGET_TYPE_ALL:
            return PreAuthzReturnCode.REQUIRE_AUTHZ
        else:
            return PreAuthzReturnCode.OK

    def send_request_to_clients(self, conn, message):
        client_tokens = conn.get_prop(self.TARGET_CLIENT_TOKENS)

        if not client_tokens:
            return None

        requests = {}
        for token in client_tokens:
            requests.update({token: message})

        admin_server: FedAdminServer = conn.server
        replies = admin_server.send_requests(requests, timeout_secs=admin_server.timeout)

        return replies

    @staticmethod
    def get_job_name(meta: dict) -> str:
        """Gets job name from job meta."""

        name = meta.get(JobMetaKey.JOB_NAME)
        if not name:
            name = meta.get(JobMetaKey.JOB_FOLDER_NAME, "No name")

        return name

    def process_replies_to_table(self, conn: Connection, replies):
        """Process the clients' replies and put in a table format.

        Args:
            conn: A Connection object.
            replies: replies from clients
        """
        if not replies:
            conn.append_string("no responses from clients")

        engine = conn.app_ctx
        table = conn.append_table(["Client", "Response"])
        for r in replies:
            if r.reply:
                resp = r.reply.body
            else:
                resp = ""
            client_name = engine.get_client_name_from_token(r.client_token)
            if not client_name:
                clients = conn.get_prop(self.TARGET_CLIENTS)
                client_name = clients.get(r.client_token, "")

            table.add_row([client_name, resp])

    def _process_replies_to_string(self, conn: Connection, replies) -> str:
        """Process the clients replies and put in a string format.

        Args:
            conn: A Connection object.
            replies: replies from clients

        Returns:
            A string response.
        """
        engine = conn.app_ctx
        response = "no responses from clients"
        if replies:
            response = ""
            for r in replies:
                client_name = engine.get_client_name_from_token(r.client_token)
                response += "client:" + client_name
                if r.reply:
                    response += " : " + r.reply.body + "\n"
                else:
                    response += " : No replies\n"
        return response
