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
import os
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from typing import Optional

from nvflare.apis.client import Client, ClientPropKey
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.fuel.sec.authn import ADMIN_AUTH_TYPE_OIDC, AuthError, get_admin_auth_type
from nvflare.fuel.utils.admin_name_utils import is_valid_admin_client_name
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.log_utils import get_module_logger, get_obj_logger
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegSession, ClientType, InternalFLContextKey
from nvflare.private.fed.server.cred_keeper import CredKeeper
from nvflare.private.fed.utils.identity_utils import get_org_from_cert, load_crt_bytes
from nvflare.security.logging import secure_format_exception

_module_logger = get_module_logger()

# How long an admin client's token->origin binding stays usable without being used, in seconds.
# 24h comfortably exceeds the 8h max admin session lifetime. Expired entries are NOT removed:
# resolve_admin_client_fqcn returns a fail-closed binding for them, because dropping the entry
# would make the token "unknown" and skip origin validation entirely (fail open). Records are
# tiny, one per admin registration, and the map is cleared on server restart.
ADMIN_CLIENT_FQCN_TTL = 24 * 60 * 60


class _AdminClientRecord:
    """Origin binding for a registered admin client token (see resolve_admin_client_fqcn)."""

    def __init__(self, name: str, fqcn: str):
        self.name = name
        self.fqcn = fqcn
        self.last_used = time.time()


def _is_oidc_admin_auth_enabled() -> bool:
    """Check whether admin auth type is configured as OIDC, using the shared fail-closed parser.

    A malformed auth config disables the OIDC exemption (cert verification stays required),
    so a misconfiguration can never widen the cert-less registration path.
    """
    startup_config = ConfigService.get_section(SystemConfigs.STARTUP_CONF) or {}
    try:
        return get_admin_auth_type(startup_config) == ADMIN_AUTH_TYPE_OIDC
    except AuthError as ex:
        _module_logger.error(f"invalid admin auth config; disabling OIDC admin auth exemption: {ex}")
        return False


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
        self.admin_clients = dict()  # token => _AdminClientRecord (admin cellnet origin bindings)
        self.admin_client_fqcn_ttl = ADMIN_CLIENT_FQCN_TTL
        self.disabled_clients = set()
        self.disabled_clients_file = None
        self.cred_keeper = CredKeeper()
        self.lock = threading.Lock()
        self.num_relays = 0

        self.logger = get_obj_logger(self)

    def set_disabled_clients_file(self, file_path: str):
        self.disabled_clients_file = file_path
        self._load_disabled_clients()

    def _load_disabled_clients(self):
        if not self.disabled_clients_file or not os.path.exists(self.disabled_clients_file):
            return
        try:
            with open(self.disabled_clients_file) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("disabled clients file must be a JSON object")
            clients = data.get("disabled_clients")
            if not isinstance(clients, list):
                raise ValueError("disabled_clients must be a list")
            with self.lock:
                self.disabled_clients = {str(client_name) for client_name in clients if client_name}
        except Exception as ex:
            self.logger.critical(
                f"failed to load disabled clients from {self.disabled_clients_file}: {ex}; "
                "refusing to start to preserve disable-client policy"
            )
            raise

    def _save_disabled_clients(self, disabled_clients=None):
        if not self.disabled_clients_file:
            return
        if disabled_clients is None:
            with self.lock:
                disabled_clients = set(self.disabled_clients)
        dirname = os.path.dirname(self.disabled_clients_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {"disabled_clients": sorted(disabled_clients)}
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f"{os.path.basename(self.disabled_clients_file)}.",
                suffix=".tmp",
                dir=dirname or ".",
                text=True,
            )
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.disabled_clients_file)
        except Exception:
            if tmp_path:
                with suppress(OSError):
                    os.unlink(tmp_path)
            raise

    def is_client_disabled(self, client_name: str) -> bool:
        with self.lock:
            return client_name in self.disabled_clients

    def disable_client(self, client_name: str) -> list:
        with self.lock:
            already_disabled = client_name in self.disabled_clients
            self.disabled_clients.add(client_name)
            removed_clients = []
            for token, client in list(self.clients.items()):
                if client.name == client_name:
                    removed_clients.append((token, client))
                    self.clients.pop(token, None)
            self.name_to_clients.pop(client_name, None)
            disabled_snapshot = set(self.disabled_clients)
            try:
                self._save_disabled_clients(disabled_snapshot)
            except Exception as ex:
                if not already_disabled:
                    self.disabled_clients.discard(client_name)
                for token, client in removed_clients:
                    self.clients[token] = client
                    self.name_to_clients[client.name] = client
                self.logger.error(f"failed to persist disabled-client state for {client_name}: {ex}")
                raise
        removed_tokens = [token for token, _client in removed_clients]
        self.logger.info(f"Client {client_name} disabled. Removed active tokens: {removed_tokens}")
        return removed_tokens

    def enable_client(self, client_name: str) -> bool:
        with self.lock:
            was_disabled = client_name in self.disabled_clients
            if was_disabled:
                self.disabled_clients.remove(client_name)
                disabled_snapshot = set(self.disabled_clients)
            else:
                disabled_snapshot = None
            if was_disabled:
                try:
                    self._save_disabled_clients(disabled_snapshot)
                except Exception as ex:
                    self.disabled_clients.add(client_name)
                    self.logger.error(f"failed to persist enabled-client state for {client_name}: {ex}")
                    raise
        self.logger.info(f"Client {client_name} enabled. Was disabled: {was_disabled}")
        return was_disabled

    def set_clients(self, clients: dict):
        self.clients = clients
        self.name_to_clients = {}
        for c in clients.values():
            self.name_to_clients[c.name] = c

    def authenticate(self, request, fl_ctx: FLContext) -> Optional[Client]:
        client_type = request.get_header(CellMessageHeaderKeys.CLIENT_TYPE)
        client = self.login_client(request, fl_ctx, client_type)
        if not client:
            return None

        # client_ip = context.peer().split(":")[1]
        client_ip = request.get_header(CellMessageHeaderKeys.CLIENT_IP)

        # new client join
        with self.lock:
            if client_type == ClientType.REGULAR:
                self.name_to_clients[client.name] = client
                self.clients.update({client.token: client})
                client_kind = "client"
            else:
                # do not update self.clients for non-regular clients
                client_kind = client_type
                if client_type == ClientType.ADMIN:
                    # Bind the admin token to the cell ORIGIN it registered from, so that
                    # validate_auth_headers can reject admin messages with a spoofed ORIGIN.
                    self.admin_clients[client.token] = _AdminClientRecord(client.name, client.get_fqcn())

            self.logger.info(
                "Client: New {} {} joined. Sent token: {}.  Total clients: {}".format(
                    client_kind, client.name + "@" + client_ip, client.token, len(self.clients)
                )
            )
        return client

    def remove_client(self, token):
        """Remove a registered client's active token entry.

        Args:
            token: client token

        Returns:
            The removed Client object, if the token was active
        """
        with self.lock:
            client = self.clients.pop(token, None)
            if client:
                self.name_to_clients.pop(client.name, None)
                self.logger.info(
                    "Client Name:{} \tToken: {} left.  Total clients: {}".format(client.name, token, len(self.clients))
                )
            else:
                self.logger.warning("remove_client: unknown token %s", token)
            return client

    def resolve_admin_client_fqcn(self, client_name: str, token: str) -> Optional[str]:
        """Resolve a registered admin client token to the cell origin FQCN it registered from.

        Args:
            client_name: client name claimed by the message
            token: auth token of the message

        Returns:
            None if the token is not a known admin registration; an empty string when the
            registration is known but must not be trusted (claimed name mismatch, no recorded
            origin, or unused beyond the TTL) — the empty string can never match a message
            origin, so validation fails CLOSED rather than skipping the origin check;
            otherwise the FQCN recorded at registration.
        """
        now = time.time()
        with self.lock:
            record = self.admin_clients.get(token)
            if record is None:
                return None
            if record.name != client_name or record.last_used < now - self.admin_client_fqcn_ttl:
                return ""
            record.last_used = now
            return record.fqcn or ""

    def login_client(self, client_login, fl_ctx: FLContext, client_type):
        proj_name = client_login.get_header(CellMessageHeaderKeys.PROJECT_NAME)
        if not self.is_valid_task(proj_name):
            fl_ctx.set_prop(
                FLContextKey.UNAUTHENTICATED, "Requested task does not match the current server task", sticky=False
            )
            self.logger.error(f"login_client failed: {proj_name}")
            return None
        return self.authenticated_client(client_login, fl_ctx, client_type)

    def has_relays(self):
        return self.num_relays > 0

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
        return self.cred_keeper.get_id_verifier(fl_ctx)

    def _verify_client_identity(self, client_name: str, shareable: Shareable, fl_ctx: FLContext) -> Optional[str]:
        """Verify the client's certificate-based identity assertion.

        Args:
            client_name: name claimed by the registering client
            shareable: registration payload carrying the cert assertion and signature
            fl_ctx: FLContext

        Returns:
            The org extracted from the verified client cert, or None if verification failed.
        """
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

        self.logger.debug(f"identity verified for client '{client_name}'")
        return get_org_from_cert(asserter_cert)

    def authenticated_client(self, request, fl_ctx: FLContext, client_type) -> Optional[Client]:
        """Use SSL certificate for authenticate the client.

        Args:
            request: client login request Message
            fl_ctx: FL_Context
            client_type: type of the client

        Returns:
            Client object.
        """
        client_name = request.get_header(CellMessageHeaderKeys.CLIENT_NAME)
        if self.is_client_disabled(client_name):
            fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, f"Client '{client_name}' is disabled", sticky=False)
            self.logger.warning(f"Reject disabled client registration: {client_name}")
            return None

        shareable = request.payload
        if not isinstance(shareable, Shareable):
            self.logger.error(f"payload must be Shareable but got {type(shareable)}")
            return None

        secure_mode = fl_ctx.get_prop(FLContextKey.SECURE_MODE, False)
        client_org = ""
        asserter_cert_data = shareable.get(IdentityChallengeKey.CERT)
        if secure_mode and not asserter_cert_data and client_type == ClientType.ADMIN and _is_oidc_admin_auth_enabled():
            # Cert-less OIDC admin exemption: OIDC admin kits are provisioned without client
            # certs/private keys, so an admin client cannot present a PKI identity assertion here.
            # Admin authority is still established later by the OIDC HCI login; the token issued
            # from this registration is a cellnet transport credential only and grants no admin
            # privileges by itself. Log at INFO so cert-less registrations remain auditable.
            self.logger.info(f"admin client '{client_name}' registered without client cert assertion (OIDC admin auth)")
        elif secure_mode:
            # Regular clients (and cert-based admins) must prove their identity with PKI.
            client_org = self._verify_client_identity(client_name, shareable, fl_ctx)
            if client_org is None:
                return None
        elif asserter_cert_data:
            try:
                asserter_cert = load_crt_bytes(asserter_cert_data)
                client_org = get_org_from_cert(asserter_cert)
            except Exception:
                pass

        with self.lock:
            # Recheck under lock so disable_client cannot race with registration after the fast-path checks above.
            if client_name in self.disabled_clients:
                fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, f"Client '{client_name}' is disabled", sticky=False)
                self.logger.warning(f"Reject disabled client registration: {client_name}")
                return None

            clients_to_be_removed = [token for token, client in self.clients.items() if client.name == client_name]
            for item in clients_to_be_removed:
                client = self.clients.pop(item, None)
                if client:
                    self.name_to_clients.pop(client.name, None)
                self.logger.info(f"Client: {client_name} already registered. Re-login the client with a new token.")

        client = Client(client_name, str(uuid.uuid4()))
        client.set_prop(ClientPropKey.ORG, client_org)
        client_fqcn = request.get_header(MessageHeaderKey.ORIGIN)
        self._set_client_props(client, client_fqcn, fl_ctx)
        self.logger.debug(f"authenticated client {client_name}: {client_fqcn=}")

        if client_type == ClientType.REGULAR and len(self.clients) >= self.max_num_clients:
            # only impose the limit to REGULAR clients
            fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, "Maximum number of clients reached", sticky=False)
            self.logger.info(f"Maximum number of clients reached. Reject client: {client_name} login.")
            return None

        if client_type == ClientType.RELAY:
            self.num_relays += 1

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
            if client_name in self.disabled_clients:
                fl_ctx.set_prop(FLContextKey.UNAUTHENTICATED, f"Client '{client_name}' is disabled", sticky=False)
                self.logger.warning(f"Reject disabled client heartbeat: {client_name}")
                return False

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
        site_config = fl_ctx.get_prop(FLContextKey.CLIENT_SITE_CONFIG)
        if site_config is not None:
            client.set_site_config(site_config)

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
            # Check whether this is a valid admin client.
            # Note that since admin clients are not kept in name_to_clients, we assume that the admin client
            # is valid and dynamically create the Client object as the result.
            if is_valid_admin_client_name(client_name):
                result = Client(client_name, None)
                result.set_fqcn(client_name)
            else:
                self.logger.debug(
                    f"no client for {client_name}: I have {self.name_to_clients.keys()} {self.clients.keys()}"
                )
        return result
