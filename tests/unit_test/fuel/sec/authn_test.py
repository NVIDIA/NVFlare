# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import uuid

import pytest

from nvflare.apis.fl_constant import CellMessageAuthHeaderKey
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.sec.authn import set_add_auth_headers_filters

AUTH_HEADERS = [
    CellMessageAuthHeaderKey.CLIENT_NAME,
    CellMessageAuthHeaderKey.TOKEN,
    CellMessageAuthHeaderKey.TOKEN_SIGNATURE,
    CellMessageAuthHeaderKey.SSID,
]


@pytest.fixture(autouse=True)
def clean_core_cells():
    original_cells = dict(CoreCell.ALL_CELLS)
    CoreCell.ALL_CELLS.clear()
    yield
    for cell in CoreCell.ALL_CELLS.values():
        cell.running = False
    CoreCell.ALL_CELLS.clear()
    CoreCell.ALL_CELLS.update(original_cells)


def _make_running_cell(fqcn: str):
    cell = Cell(fqcn=fqcn, root_url="tcp://127.0.0.1:8002", secure=False, credentials={})
    cell.core_cell.running = True
    return cell


def _unique_fqcn(prefix: str):
    return f"{prefix}_{uuid.uuid4().hex}"


def _auth_header_values(message):
    return {k: message.get_header(k) for k in AUTH_HEADERS}


def test_auth_filter_does_not_add_client_credentials_to_peer_replies():
    victim = _make_running_cell(_unique_fqcn("victim"))
    peer = _make_running_cell(_unique_fqcn("peer"))
    set_add_auth_headers_filters(victim, "victim", "tok-victim", "sig-victim", "ssid-victim")

    victim.core_cell.register_request_cb("probe", "ping", lambda _request: Message(payload="pong"))

    reply = peer.send_request("probe", "ping", victim.get_fqcn(), Message(payload="hello"), timeout=1.0)

    assert reply.payload == "pong"
    assert _auth_header_values(reply) == {
        CellMessageAuthHeaderKey.CLIENT_NAME: None,
        CellMessageAuthHeaderKey.TOKEN: None,
        CellMessageAuthHeaderKey.TOKEN_SIGNATURE: None,
        CellMessageAuthHeaderKey.SSID: None,
    }


def test_auth_filter_keeps_auth_on_outgoing_requests():
    victim = _make_running_cell(_unique_fqcn("victim"))
    peer = _make_running_cell(_unique_fqcn("peer"))
    set_add_auth_headers_filters(victim, "victim", "tok-victim", "sig-victim", "ssid-victim")
    peer.core_cell.register_request_cb("probe", "echo", lambda request: Message(payload=_auth_header_values(request)))

    reply = victim.send_request("probe", "echo", peer.get_fqcn(), Message(payload="hello"), timeout=1.0)

    assert reply.payload == {
        CellMessageAuthHeaderKey.CLIENT_NAME: "victim",
        CellMessageAuthHeaderKey.TOKEN: "tok-victim",
        CellMessageAuthHeaderKey.TOKEN_SIGNATURE: "sig-victim",
        CellMessageAuthHeaderKey.SSID: "ssid-victim",
    }


def test_auth_filter_keeps_client_reply_auth_on_server_path():
    victim = _make_running_cell(_unique_fqcn("victim"))
    server = _make_running_cell(f"server.{_unique_fqcn('authn')}")
    set_add_auth_headers_filters(victim, "victim", "tok-victim", "sig-victim", "ssid-victim")
    victim.core_cell.register_request_cb("probe", "ping", lambda _request: Message(payload="pong"))

    reply = server.send_request("probe", "ping", victim.get_fqcn(), Message(payload="hello"), timeout=1.0)

    assert reply.payload == "pong"
    assert _auth_header_values(reply) == {
        CellMessageAuthHeaderKey.CLIENT_NAME: "victim",
        CellMessageAuthHeaderKey.TOKEN: "tok-victim",
        CellMessageAuthHeaderKey.TOKEN_SIGNATURE: "sig-victim",
        CellMessageAuthHeaderKey.SSID: "ssid-victim",
    }
