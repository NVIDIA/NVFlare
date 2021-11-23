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

import threading
import time

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule
from nvflare.fuel.hci.server.audit import CommandAudit
from nvflare.fuel.hci.server.authz import AuthorizationService, AuthzCommandModule, AuthzFilter
from nvflare.fuel.hci.server.builtin import new_command_register_with_builtin_module
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.file_transfer import FileTransferModule
from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.hci.server.login import LoginModule, SessionManager, SimpleAuthenticator
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.admin_defs import Message

from .app_authz import AppAuthzService


def new_message(conn: Connection, topic, body) -> Message:
    msg = Message(topic=topic, body=body)
    event_id = conn.get_prop(ConnProps.EVENT_ID, default=None)
    if event_id:
        msg.set_header(ConnProps.EVENT_ID, event_id)

    user_name = conn.get_prop(ConnProps.USER_NAME, default=None)
    if user_name:
        msg.set_header(ConnProps.USER_NAME, user_name)

    return msg


class _Waiter(object):
    def __init__(self, req: Message):
        self.req = req
        self.reply = None
        self.reply_time = None


class _Client(object):
    def __init__(self, token):
        self.token = token
        self.last_heard_time = None
        self.outgoing_reqs = []
        self.fnf_reqs = []   # fire-and-forget requests
        self.waiters = {}  # ref => waiter
        self.req_lock = threading.Lock()
        self.waiter_lock = threading.Lock()

    def send(self, req: Message):
        waiter = _Waiter(req)
        with self.req_lock:
            self.outgoing_reqs.append(req)

        with self.waiter_lock:
            self.waiters[req.id] = waiter

        return waiter

    def fire_and_forget(self, reqs: [Message]):
        with self.req_lock:
            for r in reqs:
                self.fnf_reqs.append(r)

    def get_outgoing_requests(self, max_num_reqs=0) -> [Message]:
        result = []
        with self.req_lock:
            # reqs in outgoing_reqs have higher priority
            q = self.outgoing_reqs
            if len(self.outgoing_reqs) <= 0:
                # no regular reqs - take fire-and-forget reqs
                q = self.fnf_reqs

            if max_num_reqs <= 0:
                num_reqs = len(q)
            else:
                num_reqs = min(max_num_reqs, len(q))

            for i in range(num_reqs):
                result.append(q.pop(0))

        return result

    def accept_reply(self, reply: Message):
        self.last_heard_time = time.time()
        ref_id = reply.get_ref_id()
        with self.waiter_lock:
            # print("accept_reply waiter_lock ID: {}".format(ref_id))
            w = self.waiters.pop(ref_id, None)
            # print("accept_reply waiter_lock ID Done.....")

        if w:
            assert isinstance(w, _Waiter)
            w.reply = reply
            w.reply_time = time.time()

    def cancel_waiter(self, waiter: _Waiter):
        req = waiter.req
        with self.waiter_lock:
            # print("cancel waiter_lock ID: {}".format(req.id))
            waiter = self.waiters.pop(req.id, None)
            # print("cancel waiter_lock ID done .....")

        if waiter:
            with self.req_lock:
                if req in self.outgoing_reqs:
                    self.outgoing_reqs.remove(req)


class ClientReply(object):
    def __init__(self, client_token: str, req: Message, reply: Message):
        self.client_token = client_token
        self.request = req
        self.reply = reply


class _ClientReq(object):
    def __init__(self, client, req: Message):
        self.client = client
        self.req = req
        self.waiter = None


class FedAdminServer(AdminServer):
    """
    The FedAdminServer is the framework for developing admin commands.

    Args:
        fed_admin_interface: the Fed Server's admin interface
        users: a dict of user name => pwd hash
        cmd_modules: a list of CommandModules
        file_upload_dir: the directory for uploaded files
        file_download_dir: the directory for files to be downloaded
        allowed_shell_cmds: list of shell commands allowed. If not specified, all allowed.
        host: the IP address of the admin server
        port: port number of admin server
        ca_cert_file_name: the root CA's cert file name
        server_cert_file_name: server's cert, signed by the CA
        server_key_file_name: server's private key file
        accepted_client_cns: list of accepted Common Names from client, if specified
    """

    def __init__(
        self,
        fed_admin_interface,
        users,
        cmd_modules,
        file_upload_dir,
        file_download_dir,
        allowed_shell_cmds,
        host,
        port,
        ca_cert_file_name,
        server_cert_file_name,
        server_key_file_name,
        accepted_client_cns=None,
        app_validator=None,
    ):
        cmd_reg = new_command_register_with_builtin_module(app_ctx=fed_admin_interface)
        self.sai = fed_admin_interface
        self.allowed_shell_cmds = allowed_shell_cmds

        authenticator = SimpleAuthenticator(users)
        sess_mgr = SessionManager()
        login_module = LoginModule(authenticator, sess_mgr)
        cmd_reg.register_module(login_module)

        # register filters - order is important!
        # login_module is also a filter that determines user name or authenticated
        cmd_reg.add_filter(login_module)

        # next is the authorization filter and command module
        authorizer = AuthorizationService.get_authorizer()
        authz_filter = AuthzFilter(authorizer=authorizer)
        cmd_reg.add_filter(authz_filter)
        authz_cmd_module = AuthzCommandModule(authorizer=authorizer)
        cmd_reg.register_module(authz_cmd_module)

        # audit filter records commands to audit trail
        auditor = AuditService.get_auditor()
        assert isinstance(auditor, Auditor), "auditor must be Auditor but got {}".format(type(auditor))
        audit_filter = CommandAudit(auditor)
        cmd_reg.add_filter(audit_filter)

        self.file_upload_dir = file_upload_dir
        self.file_download_dir = file_download_dir

        AppAuthzService.initialize(app_validator)
        cmd_reg.register_module(
            FileTransferModule(
                upload_dir=file_upload_dir,
                download_dir=file_download_dir,
                upload_folder_authz_func=AppAuthzService.authorize_upload,
            )
        )

        if cmd_modules:
            assert isinstance(cmd_modules, list), "cmd_modules must be list"

            for m in cmd_modules:
                assert isinstance(m, CommandModule), "cmd_modules must contain CommandModule"
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
        )

        self.clients = {}  # token => _Client
        self.client_lock = threading.Lock()
        self.timeout = 10.0  # default admin command timeout

    def client_heartbeat(self, token):
        """
        This method is called by the Fed Engine to indicate the client is alive.
        Args:
            token:

        Returns:

        """
        with self.client_lock:
            client = self.clients.get(token)
            if not client:
                client = _Client(token)
                self.clients[token] = client
            client.last_heard_time = time.time()
            return client

    def client_dead(self, client_token):
        """
        This method is called by the Fed Engine to indicate the client is dead.

        Args:
            client_token: the session token of the client

        Returns:

        """
        with self.client_lock:
            self.clients.pop(client_token, None)

    def get_client_tokens(self) -> []:
        """
        Get tokens of existing clients.
        """
        result = []
        with self.client_lock:
            for token in self.clients.keys():
                result.append(token)
        return result

    def send_request_to_client(self, req: Message, client_token: str, timeout_secs=2.0) -> ClientReply:
        assert isinstance(req, Message), "request must be Message"
        reqs = {client_token: req}
        replies = self.send_requests(reqs, timeout_secs)
        if replies is None or len(replies) <= 0:
            return None
        else:
            return replies[0]

    def send_request_to_clients(self, req: Message, client_tokens: [str], timeout_secs=2.0) -> [ClientReply]:
        assert isinstance(req, Message), "request must be Message"
        reqs = {}
        for token in client_tokens:
            reqs[token] = req

        return self.send_requests(reqs, timeout_secs)

    def send_requests(self, requests: dict, timeout_secs=2.0) -> [ClientReply]:
        """
        This method is to be used by a Command Handler to send requests to Clients.
        Hence it is run in the Command Handler's handling thread.
        This is a blocking call - returned only after all responses are received or timeout.

        Args:
            requests: a dict of requests: client token => req message or list of msgs
            timeout_secs: how long to wait for reply before timeout

        Returns: a list of ClientReply

        """

        assert isinstance(requests, dict), "requests must be a dict"

        if len(requests) <= 0:
            return []

        if timeout_secs <= 0.0:
            # this is fire-and-forget!
            for token, r in requests.items():
                client = self.clients.get(token)
                if not client:
                    continue

                assert isinstance(client, _Client)
                if isinstance(r, list):
                    reqs = r
                else:
                    reqs = [r]

                client.fire_and_forget(reqs)
            # No replies
            return []

        # Regular requests
        client_reqs = []
        with self.client_lock:
            for token, r in requests.items():
                client = self.clients.get(token)
                if not client:
                    continue

                if isinstance(r, list):
                    reqs = r
                else:
                    reqs = [r]

                for req in reqs:
                    assert isinstance(req, Message), "request must be a Message"
                    client_reqs.append(_ClientReq(client, req))

        return self._send_client_reqs(client_reqs, timeout_secs)

    def _send_client_reqs(self, client_reqs, timeout_secs) -> [ClientReply]:
        result = []
        if len(client_reqs) <= 0:
            return result

        for cr in client_reqs:
            cr.waiter = cr.client.send(cr.req)

        start_time = time.time()
        while True:
            all_received = True
            for cr in client_reqs:
                if cr.waiter.reply_time is None:
                    all_received = False
                    break

            if all_received:
                break

            if time.time() - start_time > timeout_secs:
                # timeout
                break

            time.sleep(0.1)

        for cr in client_reqs:
            result.append(ClientReply(client_token=cr.client.token,
                                      req=cr.waiter.req,
                                      reply=cr.waiter.reply))

            if cr.waiter.reply_time is None:
                # this client timed out
                cr.client.cancel_waiter(cr.waiter)

        return result

    def send_to_all_clients(self, req: Message, timeout_secs=2.0) -> [ClientReply]:
        """
        This method is to be used by a Command Handler to send a request to all Clients.
        Hence it is run in the Command Handler's handling thread.
        This is a blocking call - returned only after all responses are received or timeout.

        Args:
            req: the request to be sent
            timeout_secs: how long to wait for reply before timeout

        Returns: a list of ClientReply

        """
        assert isinstance(req, Message), "req must be a Message"
        return self.send_requests_to_all_clients([req], timeout_secs)

    def send_requests_to_all_clients(self, reqs: [Message], timeout_secs=2.0) -> [ClientReply]:
        """
        Send multiple request messages to all clients and wait for replies

        Args:
            reqs: requests to be sent
            timeout_secs: how long to wait for reply before timeout

        Returns: a list of ClientReply

        """
        if len(reqs) <= 0:
            return []

        client_reqs = []
        with self.client_lock:
            for _, client in self.clients.items():
                for req in reqs:
                    client_reqs.append(_ClientReq(client, req))

        return self._send_client_reqs(client_reqs, timeout_secs)

    def accept_reply(self, client_token, reply: Message):
        """
        This method is to be called by the FL Engine after a client response is received.
        Hence it is called from the FL Engine's message processing thread.

        Args:
            client_token: session token of the client
            reply: the reply message

        Returns:
        """
        client = self.client_heartbeat(client_token)

        ref_id = reply.get_ref_id()
        assert ref_id is not None, "protocol error: missing ref_id in reply from client {}".format(client_token)

        client.accept_reply(reply)

    def get_outgoing_requests(self, client_token, max_reqs=0):
        """
        This method is called by FL Engine to get outgoing messages to the client, so it
        can send them to the client.

        Args:
            client_token: session token of the client
            max_reqs: max number of requests. 0 means unlimited.

        Returns:

        """
        with self.client_lock:
            client = self.clients.get(client_token)

        if client:
            return client.get_outgoing_requests(max_reqs)
        else:
            return []

    def stop(self):
        super().stop()
        self.sai.close()
