# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Mapping

from nvflare.apis.fl_constant import CellMessageAuthHeaderKey
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.validation_utils import check_object_type, check_str

ADMIN_AUTH_TYPE_CERT = "cert"
ADMIN_AUTH_TYPE_OIDC = "oidc"
SUPPORTED_ADMIN_AUTH_TYPES = (ADMIN_AUTH_TYPE_CERT, ADMIN_AUTH_TYPE_OIDC)

# Machine-readable AuthError codes: reject-code selection must not depend on error message text.
AUTH_ERROR_CODE_CONFIG_INVALID = "auth_config_invalid"
AUTH_ERROR_CODE_NOT_CONFIGURED = "auth_not_configured"
AUTH_ERROR_CODE_UNAVAILABLE = "auth_unavailable"
AUTH_ERROR_CODE_INVALID_TOKEN = "auth_invalid_token"

MAX_SESSION_LIFETIME_KEY = "max_session_lifetime"


def parse_max_session_lifetime(value) -> float:
    """Validate the auth.admin.max_session_lifetime config value (seconds).

    The single definition of validity, shared by provisioning (which fails the provision)
    and the server runtime (which warns and falls back to the default).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(
            f"invalid auth.admin.{MAX_SESSION_LIFETIME_KEY} {value!r}: must be a positive number of seconds"
        )
    return float(value)


def _is_server_fqcn(fqcn: str) -> bool:
    return bool(fqcn) and FqcnInfo(fqcn).is_on_server


class AuthError(Exception):
    """Authentication failed or returned unusable identity data."""

    def __init__(self, message: str = "", code: str = ""):
        super().__init__(message)
        self.code = code


def get_admin_auth_config(server_startup_conf) -> Mapping:
    """Return the admin auth config section of the server startup config.

    Supports both the provisioned layout {"auth": {"admin": {...}}} and the flat
    layout {"auth": {"type": ..., ...}}. Fails closed: an 'auth' or 'auth.admin'
    section that is present but not a mapping raises AuthError instead of silently
    falling back to cert auth.
    """
    if not isinstance(server_startup_conf, Mapping):
        return {}
    auth_config = server_startup_conf.get("auth")
    if auth_config is None:
        return {}
    if not isinstance(auth_config, Mapping):
        raise AuthError(
            "invalid server config: 'auth' must be a mapping",
            code=AUTH_ERROR_CODE_CONFIG_INVALID,
        )
    if "admin" in auth_config:
        admin_config = auth_config.get("admin")
        if not isinstance(admin_config, Mapping):
            raise AuthError(
                "invalid server config: 'auth.admin' must be a mapping",
                code=AUTH_ERROR_CODE_CONFIG_INVALID,
            )
        return admin_config
    return auth_config


def get_admin_auth_type(server_startup_conf) -> str:
    """Return the normalized admin authentication type ('cert' by default).

    Raises AuthError for a malformed auth section or an unsupported type, so that
    every reader of this config fails closed in the same way.
    """
    admin_auth_config = get_admin_auth_config(server_startup_conf)
    auth_type = str(admin_auth_config.get("type", ADMIN_AUTH_TYPE_CERT) or ADMIN_AUTH_TYPE_CERT).strip().lower()
    if auth_type not in SUPPORTED_ADMIN_AUTH_TYPES:
        raise AuthError(
            f"unsupported admin authentication type '{auth_type}'",
            code=AUTH_ERROR_CODE_CONFIG_INVALID,
        )
    return auth_type


def add_authentication_headers(msg: Message, client_name: str, auth_token, token_signature, ssid=None):
    """Add authentication headers to the specified message.

    Args:
        msg: the message that the headers are added to
        client_name: name of the client
        auth_token: authentication token
        token_signature: token signature
        ssid: optional SSID

    Returns:

    """
    if client_name:
        msg.set_header(CellMessageAuthHeaderKey.CLIENT_NAME, client_name)

    if ssid:
        msg.set_header(CellMessageAuthHeaderKey.SSID, ssid)

    msg.set_header(CellMessageAuthHeaderKey.TOKEN, auth_token if auth_token else "NA")
    msg.set_header(CellMessageAuthHeaderKey.TOKEN_SIGNATURE, token_signature if token_signature else "NA")


def add_server_path_reply_authentication_headers(
    msg: Message, client_name: str, auth_token, token_signature, ssid=None
):
    origin = msg.get_header(MessageHeaderKey.ORIGIN)
    destination = msg.get_header(MessageHeaderKey.DESTINATION)
    to_cell = msg.get_header(MessageHeaderKey.TO_CELL)
    # Auth headers are for the server trust boundary, not for peer clients. A peer reply routed through the server
    # must authenticate to the server on the next hop, and the server strips these headers before forwarding to peer.
    # Server-owned job/transfer cells also keep auth on replies from or to server paths.
    if _is_server_fqcn(origin) or _is_server_fqcn(destination) or _is_server_fqcn(to_cell):
        add_authentication_headers(msg, client_name, auth_token, token_signature, ssid)


def set_add_auth_headers_filters(cell: Cell, client_name: str, auth_token: str, token_signature: str, ssid=None):
    """Set filters for adding auth headers to outgoing requests and server-path replies.

    Args:
        cell: the cell to add the filters to.
        client_name: name of the client
        auth_token: authentication token
        token_signature: token signature
        ssid: SSID, optional

    Returns: None

    """
    check_object_type("cell", cell, Cell)

    if client_name:
        check_str("client_name", client_name)

    check_str("auth_token", auth_token)
    check_str("token_signature", token_signature)

    if ssid:
        check_str("ssid", ssid)

    cell.core_cell.add_outgoing_reply_filter(
        channel="*",
        topic="*",
        cb=add_server_path_reply_authentication_headers,
        client_name=client_name,
        auth_token=auth_token,
        token_signature=token_signature,
        ssid=ssid,
    )
    cell.core_cell.add_outgoing_request_filter(
        channel="*",
        topic="*",
        cb=add_authentication_headers,
        client_name=client_name,
        auth_token=auth_token,
        token_signature=token_signature,
        ssid=ssid,
    )
