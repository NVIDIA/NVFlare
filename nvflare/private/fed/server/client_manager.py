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
import uuid
from typing import Optional

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, SecureTrainConst
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegSession, ClientType, InternalFLContextKey
from nvflare.private.fed.utils.identity_utils import IdentityVerifier, load_crt_bytes
from nvflare.security.logging import secure_format_exception


class ClientManager:
    def __init__(self, project_name=None, min_num_clients=2, max_num_clients=10):
        """Manages client adding and removing.

        Args:
            project_name: project name
            min_num_clients: minimum number of clients allowed.
            max_num_clients: maximum number of clients allowed.
        """
        self.project_name = project_name
        # TODO:: remove min num clients
        self.min_num_clients = min_num_clients
        self.max_num_clients = max_num_clients
        self.clients = dict()  # token => Client
        self.name_to_clients = dict()  # name => Client
        self.id_verifier = None
        self.lock = threading.Lock()

        self.logger = get_obj_logger(self)

    def set_clients(self, clients: dict):
        self.clients = clients
        self.name_to_clients = {}
        for c in clients.values():
            self.name_to_clients[c.name] = c

    def authenticate(self, request, fl_ctx: FLContext) -> Optional[Client]:
        client = self.login_client(request, fl_ctx)
        if not client:
            return None

        # client_ip = context.peer().split(":")[1]
        client_ip = request.get_header(CellMessageHeaderKeys.CLIENT_IP)

        # new client join
        with self.lock:
            client_type = request.get_header(CellMessageHeaderKeys.CLIENT_TYPE)
            if client_type == ClientType.REGULAR:
                self.name_to_clients[client.name] = client
                self.clients.update({client.token: client})
                client_kind = "client"
            else:
                # do not update self.clients for non-regular clients
                client_kind = client_type

            self.logger.info(
                "Client: New {} {} joined. Sent token: {}.  Total clients: {}".format(
                    client_kind, client.name + "@" + client_ip, client.token, len(self.clients)
                )
            )
        return client

    def remove_client(self, token):
        """Remove a registered client.

        Args:
            token: client token

        Returns:
            The removed Client object
        """
        with self.lock:
            client = self.clients.pop(token, None)
            if client:
                self.name_to_clients.pop(client.name, None)
            self.logger.info(
                "Client Name:{} \tToken: {} left.  Total clients: {}".format(client.name, token, len(self.clients))
            )
            return client

    def login_client(self, client_login, fl_ctx: FLContext):
        proj_name = client_login.get_header(CellMessageHeaderKeys.PROJECT_NAME)
        if not self.is_valid_task(proj_name):
            fl_ctx.set_prop(
                FLContextKey.UNAUTHENTICATED, "Requested task does not match the current server task", sticky=False
            )
            self.logger.error(f"login_client failed: {proj_name}")
            return None
        return self.authenticated_client(client_login, fl_ctx)

    def validate_client(self, request, fl_ctx: FLContext, allow_new=False):
        """Validate the client state message.

        Args:
            request: A request from client.
            fl_ctx: FLContext
            allow_new: whether to allow new client. Note that its task should still match server's.

        Returns:
             client id if it's a valid client
        """
        # token = client_state.token
        token = request.get_header(CellMessageHeaderKeys.TOKEN)
        if not token:
            fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, "Could not read client uid from the payload", sticky=False)
            client = None
        elif not self.is_valid_task(request.get_header(CellMessageHeaderKeys.PROJECT_NAME)):
            fl_ctx.set_prop(
                FLContextKey.UNAUTHENTICATED, "Requested task does not match the current server task", sticky=False
            )
            client = None
        elif not (allow_new or self.is_from_authorized_client(token)):
            fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, "Unknown client identity", sticky=False)
            client = None
        else:
            client = self.clients.get(token)
        return client

    def _get_id_verifier(self, fl_ctx: FLContext):
        if not self.id_verifier:
            server_config = fl_ctx.get_prop(FLContextKey.SERVER_CONFIG)
            if not server_config:
                self.logger.error(f"missing {FLContextKey.SERVER_CONFIG} in FL context")
                return None

            if not isinstance(server_config, list):
                self.logger.error(f"expect server_config to be list but got {type(server_config)}")
                return None

            server1 = server_config[0]
            if not isinstance(server1, dict):
                self.logger.error(f"expect server config data to be dict but got {type(server1)}")
                return None

            root_cert_file = server1.get(SecureTrainConst.SSL_ROOT_CERT)
            if not root_cert_file:
                self.logger.error(f"missing {SecureTrainConst.SSL_ROOT_CERT} in server config")
                return None

            self.id_verifier = IdentityVerifier(root_cert_file=root_cert_file)
        return self.id_verifier

    def authenticated_client(self, request, fl_ctx: FLContext) -> Optional[Client]:
        """Use SSL certificate for authenticate the client.

        Args:
            request: client login request Message
            fl_ctx: FL_Context

        Returns:
            Client object.
        """
        client_name = request.get_header(CellMessageHeaderKeys.CLIENT_NAME)
        shareable = request.payload
        if not isinstance(shareable, Shareable):
            self.logger.error(f"payload must be Shareable but got {type(shareable)}")
            return None

        secure_mode = fl_ctx.get_prop(FLContextKey.SECURE_MODE, False)
        if secure_mode:
            # verify client identity
            asserter_cert_data = shareable.get(IdentityChallengeKey.CERT)
            if not asserter_cert_data:
                self.logger.error("missing client cert in register request")
                return None

            signature = shareable.get(IdentityChallengeKey.SIGNATURE)
            if not signature:
                self.logger.error("missing signature in register request")
                return None

            asserter_cert = load_crt_bytes(asserter_cert_data)
            id_verifier = self._get_id_verifier(fl_ctx)
            reg = fl_ctx.get_prop(InternalFLContextKey.CLIENT_REG_SESSION)
            if not reg:
                self.logger.error(f"missing {InternalFLContextKey.CLIENT_REG_SESSION} in FLContext!")
                return None

            if not isinstance(reg, ClientRegSession):
                self.logger.error(f"reg should be ClientRegSession but got {type(reg)}")
                return None

            try:
                id_verifier.verify_common_name(
                    asserted_cn=client_name,
                    asserter_cert=asserter_cert,
                    signature=signature,
                    nonce=reg.nonce,
                )
            except Exception as ex:
                self.logger.error(f"failed to verify client identity: {secure_format_exception(ex)}")
                return None

            self.logger.info(f"identity verified for client '{client_name}'")

        with self.lock:
            clients_to_be_removed = [token for token, client in self.clients.items() if client.name == client_name]
            for item in clients_to_be_removed:
                client = self.clients.pop(item, None)
                if client:
                    self.name_to_clients.pop(client.name, None)
                self.logger.info(f"Client: {client_name} already registered. Re-login the client with a new token.")

        client = Client(client_name, str(uuid.uuid4()))
        client_fqcn = request.get_header(MessageHeaderKey.ORIGIN)
        self._set_client_props(client, client_fqcn, fl_ctx)
        self.logger.info(f"authenticated client {client_name}: {client_fqcn=}")

        if len(self.clients) >= self.max_num_clients:
            fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, "Maximum number of clients reached", sticky=False)
            self.logger.info(f"Maximum number of clients reached. Reject client: {client_name} login.")
            return None

        return client

    def is_from_authorized_client(self, token):
        """Check if a client is authorized.

        Args:
            token: client token

        Returns:
            True if it is a recognised client
        """
        return token in self.clients

    def is_valid_task(self, task):
        """Check whether the requested task matches the server's project_name.

        Returns:
            True if task name is the same as server's project name.
        """
        # TODO: change the name of this method
        return task == self.project_name

    def heartbeat(self, token, client_name, client_fqcn, fl_ctx: FLContext):
        """Update the heartbeat of the client.

        Args:
            token: client token
            client_name: client name
            client_fqcn: FQCN of the client
            fl_ctx: FLContext

        Returns:
            If a new client needs to be created.
        """
        with self.lock:
            client = self.clients.get(token)
            if client:
                client.last_connect_time = time.time()
                self.logger.debug(f"Receive heartbeat from Client:{token}")
                return False
            else:
                for _token, _client in self.clients.items():
                    if _client.name == client_name:
                        fl_ctx.set_prop(
                            FLContextKey.COMMUNICATION_ERROR,
                            "Client ID already registered as a client: {}".format(client_name),
                            sticky=False,
                        )
                        self.logger.info(
                            f"Failed to re-activate the client:{client_name} with token: {token}. "
                            f"Client already exist with token: {_token}."
                        )
                        return False

                client = Client(client_name, token)
                self._set_client_props(client, client_fqcn, fl_ctx)
                self.clients.update({token: client})
                self.name_to_clients[client.name] = client
                self.logger.info(f"Re-activate the client: {client_name} at {client_fqcn} with token: {token}")
                return True

    @staticmethod
    def _set_client_props(client: Client, fqcn: str, fl_ctx: FLContext):
        client.set_fqcn(fqcn)
        client.last_connect_time = time.time()
        peer_ctx = fl_ctx.get_peer_context()
        if peer_ctx:
            client.set_fqsn(peer_ctx.get_prop(ReservedKey.FQSN, "?"))
            client.set_is_leaf(peer_ctx.get_prop(ReservedKey.IS_LEAF, "?"))

    def get_clients(self):
        """Get the list of registered clients.

        Returns:
            A dict of {client_token: client}
        """
        return self.clients

    def get_min_clients(self):
        return self.min_num_clients

    def get_max_clients(self):
        return self.max_num_clients

    def get_all_clients_from_inputs(self, inputs):
        clients = []
        invalid_inputs = []
        for item in inputs:
            client = self.clients.get(item)
            # if item in self.get_all_clients():
            if client:
                clients.append(client)
            else:
                client = self.get_client_from_name(item)
                if client:
                    clients.append(client)
                else:
                    invalid_inputs.append(item)
        return clients, invalid_inputs

    def get_client_from_name(self, client_name):
        result = self.name_to_clients.get(client_name)
        if not result:
            self.logger.error(
                f"no client for {client_name}: I have {self.name_to_clients.keys()} {self.clients.keys()}"
            )
        return result
