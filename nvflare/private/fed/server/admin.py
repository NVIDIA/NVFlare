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

import threading
import time
from typing import List, Optional

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.cellnet.net_manager import NetManager
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule
from nvflare.fuel.hci.server.audit import CommandAudit
from nvflare.fuel.hci.server.authz import AuthzFilter
from nvflare.fuel.hci.server.builtin import new_command_register_with_builtin_module
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.hci.server.login import LoginModule, SessionManager, SimpleAuthenticator
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import Message
from nvflare.private.defs import ERROR_MSG_PREFIX, RequestHeader
from nvflare.private.fed.server.message_send import ClientReply, send_requests


def new_message(conn: Connection, topic, body, require_authz: bool) -> Message:
    msg = Message(topic=topic, body=body)

    cmd_entry = conn.get_prop(ConnProps.CMD_ENTRY)
    if cmd_entry:
        msg.set_header(RequestHeader.ADMIN_COMMAND, cmd_entry.name)
        msg.set_header(RequestHeader.REQUIRE_AUTHZ, str(require_authz).lower())

    props_to_copy = [
        ConnProps.EVENT_ID,
        ConnProps.USER_NAME,
        ConnProps.USER_ROLE,
        ConnProps.USER_ORG,
        ConnProps.SUBMITTER_NAME,
        ConnProps.SUBMITTER_ORG,
        ConnProps.SUBMITTER_ROLE,
    ]

    for p in props_to_copy:
        prop = conn.get_prop(p, default=None)
        if prop:
            msg.set_header(p, prop)

    return msg


class _Client(object):
    def __init__(self, token, name):
        self.token = token
        self.name = name
        self.last_heard_time = None


class _ClientReq(object):
    def __init__(self, client, req: Message):
        self.client = client
        self.req = req


def check_client_replies(replies: List[ClientReply], client_sites: List[str], command: str):
    display_sites = ", ".join(client_sites)
    if not replies:
        raise RuntimeError(f"Failed to {command} to the clients {display_sites}: no replies.")
    if len(replies) != len(client_sites):
        raise RuntimeError(f"Failed to {command} to the clients {display_sites}: not enough replies.")

    error_msg = ""
    for r, client_name in zip(replies, client_sites):
        if r.reply and ERROR_MSG_PREFIX in r.reply.body:
            error_msg += f"\t{client_name}: {r.reply.body}\n"
    if error_msg != "":
        raise RuntimeError(f"Failed to {command} to the following clients: \n{error_msg}")


class FedAdminServer(AdminServer):
    def __init__(
        self,
        cell: Cell,
        fed_admin_interface,
        users,
        cmd_modules,
        file_upload_dir,
        file_download_dir,
        host,
        port,
        ca_cert_file_name,
        server_cert_file_name,
        server_key_file_name,
        accepted_client_cns=None,
        download_job_url="",
    ):
        """The FedAdminServer is the framework for developing admin commands.

        Args:
            fed_admin_interface: the server's federated admin interface
            users: a dict of {username: pwd hash}
            cmd_modules: a list of CommandModules
            file_upload_dir: the directory for uploaded files
            file_download_dir: the directory for files to be downloaded
            host: the IP address of the admin server
            port: port number of admin server
            ca_cert_file_name: the root CA's cert file name
            server_cert_file_name: server's cert, signed by the CA
            server_key_file_name: server's private key file
            accepted_client_cns: list of accepted Common Names from client, if specified
            download_job_url: download job url
        """
        cmd_reg = new_command_register_with_builtin_module(app_ctx=fed_admin_interface)
        self.sai = fed_admin_interface
        self.cell = cell
        self.client_lock = threading.Lock()

        authenticator = SimpleAuthenticator(users)
        sess_mgr = SessionManager()
        login_module = LoginModule(authenticator, sess_mgr)
        cmd_reg.register_module(login_module)

        # register filters - order is important!
        # login_module is also a filter that determines if user is authenticated
        cmd_reg.add_filter(login_module)

        # next is the authorization filter and command module
        authz_filter = AuthzFilter()
        cmd_reg.add_filter(authz_filter)

        # audit filter records commands to audit trail
        auditor = AuditService.get_auditor()
        # TODO:: clean this up
        if not isinstance(auditor, Auditor):
            raise TypeError("auditor must be Auditor but got {}".format(type(auditor)))
        audit_filter = CommandAudit(auditor)
        cmd_reg.add_filter(audit_filter)

        self.file_upload_dir = file_upload_dir
        self.file_download_dir = file_download_dir

        cmd_reg.register_module(sess_mgr)
        # mpm.add_cleanup_cb(sess_mgr.shutdown)

        agent = NetAgent(self.cell)
        net_mgr = NetManager(agent)
        cmd_reg.register_module(net_mgr)

        mpm.add_cleanup_cb(net_mgr.close)
        mpm.add_cleanup_cb(agent.close)

        if cmd_modules:
            if not isinstance(cmd_modules, list):
                raise TypeError("cmd_modules must be list but got {}".format(type(cmd_modules)))

            for m in cmd_modules:
                if not isinstance(m, CommandModule):
                    raise TypeError("cmd_modules must contain CommandModule but got element of type {}".format(type(m)))
                cmd_reg.register_module(m)

        AdminServer.__init__(
            self,
            cmd_reg=cmd_reg,
            host=host,
            port=port,
            ca_cert=ca_cert_file_name,
            server_cert=server_cert_file_name,
            server_key=server_key_file_name,
            accepted_client_cns=accepted_client_cns,
            extra_conn_props={
                ConnProps.DOWNLOAD_DIR: file_download_dir,
                ConnProps.UPLOAD_DIR: file_upload_dir,
                ConnProps.DOWNLOAD_JOB_URL: download_job_url,
            },
        )

        self.clients = {}  # token => _Client
        self.timeout = 10.0

    def client_heartbeat(self, token, name: str):
        """Receive client heartbeat.

        Args:
            token: the session token of the client
            name: client name

        Returns:
            Client.
        """
        with self.client_lock:
            client = self.clients.get(token)
            if not client:
                client = _Client(token, name)
                self.clients[token] = client
            client.last_heard_time = time.time()
            return client

    def client_dead(self, token):
        """Remove dead client.

        Args:
            token: the session token of the client
        """
        with self.client_lock:
            self.clients.pop(token, None)

    def get_client_tokens(self) -> []:
        """Get tokens of existing clients."""
        result = []
        with self.client_lock:
            for token in self.clients.keys():
                result.append(token)
        return result

    def send_request_to_client(self, req: Message, client_token: str, timeout_secs=2.0) -> Optional[ClientReply]:
        if not isinstance(req, Message):
            raise TypeError("request must be Message but got {}".format(type(req)))
        reqs = {client_token: req}
        replies = self.send_requests(reqs, timeout_secs=timeout_secs)
        if replies is None or len(replies) <= 0:
            return None
        else:
            return replies[0]

    def send_requests_and_get_reply_dict(self, requests: dict, timeout_secs=2.0) -> dict:
        """Send requests to clients

        Args:
            requests: A dict of requests: {client token: Message}
            timeout_secs: how long to wait for reply before timeout

        Returns:
            A dict of {client token: reply}, where reply is a Message or None (no reply received)
        """
        result = {}
        if requests:
            for token, _ in requests.items():
                result[token] = None

            replies = self.send_requests(requests, timeout_secs=timeout_secs)
            for r in replies:
                result[r.client_token] = r.reply
        return result

    def send_requests(self, requests: dict, timeout_secs=2.0, optional=False) -> [ClientReply]:
        """Send requests to clients.

        NOTE::

            This method is to be used by a Command Handler to send requests to Clients.
            Hence, it is run in the Command Handler's handling thread.
            This is a blocking call - returned only after all responses are received or timeout.

        Args:
            requests: A dict of requests: {client token: request or list of requests}
            timeout_secs: how long to wait for reply before timeout
            optional: whether the requests are optional

        Returns:
            A list of ClientReply
        """

        return send_requests(
            cell=self.cell,
            command="admin",
            requests=requests,
            clients=self.clients,
            timeout_secs=timeout_secs,
            optional=optional,
        )

    def stop(self):
        super().stop()
        self.sai.close()
