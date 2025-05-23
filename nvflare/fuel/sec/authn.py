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
from nvflare.apis.fl_constant import CellMessageAuthHeaderKey
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.validation_utils import check_object_type, check_str


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


def set_add_auth_headers_filters(cell: Cell, client_name: str, auth_token: str, token_signature: str, ssid=None):
    """Set filters for adding auth headers.

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
        cb=add_authentication_headers,
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
