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

from typing import List

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.fuel.hci.conn import Connection
from nvflare.security.security import Action, FLAuthzContext


class CommandUtil(object):

    TARGET_CLIENTS = "target_clients"
    TARGET_CLIENT_TOKENS = "target_client_tokens"
    TARGET_CLIENT_NAMES = "target_client_names"
    TARGET_TYPE = "target_type"

    TARGET_TYPE_CLIENT = "client"
    TARGET_TYPE_SERVER = "server"
    TARGET_TYPE_ALL = "all"

    SITE_SERVER = "server"
    ALL_SITES = "@ALL"
    JOB_ID = "job_id"

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
            clients, invalid_inputs = engine.validate_clients(client_names)
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

    def _authorize_actions(self, conn: Connection, args: List[str], actions):
        err = self.validate_command_targets(conn, args)
        if err:
            conn.append_error(err)
            return False, None

        target_type = conn.get_prop(self.TARGET_TYPE)
        authorize_server = False
        authorize_clients = False

        if target_type == self.TARGET_TYPE_SERVER:
            authorize_server = True
        elif target_type == self.TARGET_TYPE_CLIENT:
            authorize_clients = True
        else:
            # all
            authorize_server = True
            authorize_clients = True

        sites = []
        if authorize_clients:
            client_names = conn.get_prop(self.TARGET_CLIENT_NAMES)

            if client_names:
                sites.extend(client_names)

        if authorize_server:
            sites.append(self.SITE_SERVER)

        authz_ctx = FLAuthzContext.new_authz_context(site_names=sites, actions=actions)
        return True, authz_ctx

    def authorize_view(self, conn: Connection, args: List[str]):
        return self._authorize_actions(conn, args[1:], [Action.VIEW])

    def authorize_train(self, conn: Connection, args: List[str]):
        if len(args) != 3:
            conn.append_error("syntax error: missing job_id and target")
            return False, None

        job_id = args[1].lower()

        destination = job_id[len(WorkspaceConstants.WORKSPACE_PREFIX) :]
        conn.set_prop(self.JOB_ID, destination)

        return self._authorize_actions(conn, args[2:], [Action.TRAIN])

    def authorize_job_meta(self, conn: Connection, meta: dict, actions: List[str]):

        deploy_map = meta.get("deploy_map")
        if not deploy_map:
            conn.append_error(f"deploy_map missing for job {self.get_job_name(meta)}")
            return False, None

        sites = set()
        for app, site_list in deploy_map.items():
            sites.update(site_list)

        # Run-time might be a better spot for this
        if self.ALL_SITES.casefold() in (site.casefold() for site in sites):
            sites.add(self.SITE_SERVER)
            engine = conn.app_ctx
            clients = engine.get_clients()
            sites.update([client.name for client in clients])

        authz_ctx = FLAuthzContext.new_authz_context(site_names=list(sites), actions=actions)
        return True, authz_ctx

    def authorize_operate(self, conn: Connection, args: List[str]):
        return self._authorize_actions(conn, args[1:], [Action.OPERATE])

    def send_request_to_clients(self, conn, message, process_client_replies=None):
        client_tokens = conn.get_prop(self.TARGET_CLIENT_TOKENS)

        # for client in clients:
        #     requests.update({client.strip(): message})

        # client_names = conn.get_prop(self.TARGET_CLIENT_NAMES, None)
        if not client_tokens:
            return None

        requests = {}
        for token in client_tokens:
            requests.update({token: message})

        admin_server = conn.server
        replies = admin_server.send_requests(requests, timeout_secs=admin_server.timeout)

        if process_client_replies:
            return process_client_replies(replies)
        else:
            return replies

    @staticmethod
    def get_job_name(meta: dict) -> str:
        """Get job name from meta.json"""

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
