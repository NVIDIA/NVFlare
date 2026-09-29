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
from typing import Optional

from nvflare.apis.fl_constant import CellMessageAuthHeaderKey
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.fqcn import FQCN, FqcnInfo
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.validation_utils import check_object_type, check_str


def _is_server_fqcn(fqcn: str) -> bool:
    return bool(fqcn) and FqcnInfo(fqcn).is_on_server


def _get_client_family_fqcn(origin: str, client_name: Optional[str]) -> Optional[str]:
    if not origin or not client_name:
        return None
    origin_path = FQCN.split(origin)
    try:
        client_index = origin_path.index(client_name)
    except ValueError:
        return None
    else:
        return FQCN.join(origin_path[: client_index + 1])


def _is_client_family_member(fqcn: str, family_fqcn: Optional[str]) -> bool:
    if not fqcn:
        return False
    if family_fqcn and (fqcn == family_fqcn or FQCN.is_ancestor(family_fqcn, fqcn)):
        return True
    return False


def is_cross_client_family(origin: str, destination: str, client_name: Optional[str] = None) -> bool:
    if not origin or not destination:
        return False
    if origin == destination:
        return False
    origin_info = FqcnInfo(origin)
    destination_info = FqcnInfo(destination)
    if origin_info.is_on_server or destination_info.is_on_server:
        return False

    family_fqcn = _get_client_family_fqcn(origin, client_name)
    if family_fqcn:
        return not _is_client_family_member(destination, family_fqcn)

    # An authentication identity that cannot be mapped to the origin's FQCN
    # cannot prove that a non-server destination belongs to the same client.
    # Fail closed and require the server trust boundary.
    return True


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
    if is_cross_client_family(
        msg.get_header(MessageHeaderKey.ORIGIN), msg.get_header(MessageHeaderKey.DESTINATION), client_name
    ):
        msg.set_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED, True)


def add_server_path_reply_authentication_headers(
    msg: Message, client_name: str, auth_token, token_signature, ssid=None
):
    origin = msg.get_header(MessageHeaderKey.ORIGIN)
    destination = msg.get_header(MessageHeaderKey.DESTINATION)
    to_cell = msg.get_header(MessageHeaderKey.TO_CELL)
    # Auth headers are for the server trust boundary, not for peer clients. All
    # cross-client replies must be marked even when a cached direct peer endpoint
    # supplied the initial return path. The transport reroutes the marked reply to
    # the server, which validates and strips these headers before forwarding it.
    if (
        _is_server_fqcn(origin)
        or _is_server_fqcn(destination)
        or _is_server_fqcn(to_cell)
        or is_cross_client_family(origin, destination, client_name)
    ):
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
