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

import logging
import uuid
from threading import Lock
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import CellMessageAuthHeaderKey
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessageType, ReturnCode
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.sec.authn import set_add_auth_headers_filters
from nvflare.private.fed.server.fed_server import FederatedServer

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


class _TokenVerifier:
    def verify(self, _client_name, _token, _signature):
        return True


class _Cert:
    def public_key(self):
        return None


def _make_server_auth_filter(monkeypatch):
    server_auth = FederatedServer.__new__(FederatedServer)
    server_auth.logger = logging.getLogger(__name__)
    server_auth._peer_transit_reply_auth_lock = Lock()
    server_auth._peer_transit_reply_auth_keys = {}
    server_auth._get_id_asserter = lambda: SimpleNamespace(cert=_Cert())
    server_auth._resolve_client_fqcn_for_auth = lambda client_name, _token: client_name
    monkeypatch.setattr("nvflare.private.fed.server.fed_server.TokenVerifier", lambda _cert: _TokenVerifier())
    return server_auth._validate_auth_headers


def _make_routed_message(
    destination: str, msg_type: str, origin: str = "site-a", req_id: str = "req-1", with_auth: bool = False
):
    headers = {
        MessageHeaderKey.CHANNEL: "peer",
        MessageHeaderKey.TOPIC: "ping",
        MessageHeaderKey.ORIGIN: origin,
        MessageHeaderKey.DESTINATION: destination,
        MessageHeaderKey.MSG_TYPE: msg_type,
        MessageHeaderKey.REQ_ID: req_id,
    }
    if with_auth:
        headers.update(
            {
                CellMessageAuthHeaderKey.CLIENT_NAME: origin,
                CellMessageAuthHeaderKey.TOKEN: f"token-{origin}",
                CellMessageAuthHeaderKey.TOKEN_SIGNATURE: f"sig-{origin}",
            }
        )
    return Message(headers=headers)


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


def test_client_to_client_reply_routed_through_server_is_not_blocked_by_server_auth(monkeypatch):
    server = _make_running_cell("server")
    site_a = _make_running_cell(_unique_fqcn("site_a"))
    site_b = _make_running_cell(_unique_fqcn("site_b"))
    server.core_cell.add_incoming_filter(channel="*", topic="*", cb=_make_server_auth_filter(monkeypatch))
    set_add_auth_headers_filters(site_a, site_a.get_fqcn(), "tok-a", "sig-a")
    set_add_auth_headers_filters(site_b, site_b.get_fqcn(), "tok-b", "sig-b")
    site_b.core_cell.register_request_cb("peer", "ping", lambda _request: Message(payload="pong"))

    original_find_ep = site_a.core_cell._try_find_ep

    def _route_site_b_via_server(target_fqcn, for_msg):
        if target_fqcn == site_b.get_fqcn():
            return Endpoint(server.get_fqcn())
        return original_find_ep(target_fqcn, for_msg)

    monkeypatch.setattr(site_a.core_cell, "_try_find_ep", _route_site_b_via_server)

    reply = site_a.send_request("peer", "ping", site_b.get_fqcn(), Message(payload="hello"), timeout=0.2)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload == "pong"


def test_server_auth_filter_allows_matching_unauthenticated_client_reply_transit(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)

    assert auth_filter(_make_routed_message("site-b", MessageType.REQ, with_auth=True)) is None
    assert auth_filter(_make_routed_message("site-a", MessageType.REPLY, origin="site-b")) is None


def test_server_auth_filter_rejects_untracked_unauthenticated_client_reply_transit(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)

    reply = auth_filter(_make_routed_message("site-a", MessageType.REPLY, origin="site-b"))

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_server_auth_filter_consumes_tracked_client_reply_transit(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)
    transit_reply = _make_routed_message("site-a", MessageType.REPLY, origin="site-b")

    assert auth_filter(_make_routed_message("site-b", MessageType.REQ, with_auth=True)) is None
    assert auth_filter(transit_reply) is None
    rejected_reply = auth_filter(transit_reply)

    assert rejected_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_server_auth_filter_still_rejects_unauthenticated_client_request_transit(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)

    reply = auth_filter(_make_routed_message("site-b", MessageType.REQ))

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_server_auth_filter_still_rejects_unauthenticated_server_destination(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)

    reply = auth_filter(_make_routed_message("server.job-1", MessageType.REPLY))

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


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


def test_auth_filter_keeps_auth_on_replies_from_server_path_origin():
    server = _make_running_cell(f"server.{_unique_fqcn('job')}")
    client = _make_running_cell(_unique_fqcn("client"))
    set_add_auth_headers_filters(server, "server-job", "tok-server", "sig-server", "ssid-server")

    def _reply(_request):
        return Message(payload="pong")

    server.core_cell.register_request_cb("probe", "ping", _reply)

    reply = client.send_request("probe", "ping", server.get_fqcn(), Message(payload="hello"), timeout=1.0)

    assert reply.payload == "pong"
    assert reply.get_header(MessageHeaderKey.ORIGIN) == server.get_fqcn()
    assert _auth_header_values(reply) == {
        CellMessageAuthHeaderKey.CLIENT_NAME: "server-job",
        CellMessageAuthHeaderKey.TOKEN: "tok-server",
        CellMessageAuthHeaderKey.TOKEN_SIGNATURE: "sig-server",
        CellMessageAuthHeaderKey.SSID: "ssid-server",
    }
