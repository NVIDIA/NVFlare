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
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import CellMessageAuthHeaderKey
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessageType, ReturnCode
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.sec.authn import is_cross_client_family, set_add_auth_headers_filters
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
    original_pools = dict(StatsPoolManager.pools)
    CoreCell.ALL_CELLS.clear()
    StatsPoolManager.pools.clear()
    yield
    for cell in CoreCell.ALL_CELLS.values():
        cell.running = False
    CoreCell.ALL_CELLS.clear()
    CoreCell.ALL_CELLS.update(original_cells)
    StatsPoolManager.pools.clear()
    StatsPoolManager.pools.update(original_pools)


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


def _make_server_auth_filter(monkeypatch, client_fqcn_resolver=None):
    server_auth = FederatedServer.__new__(FederatedServer)
    server_auth.logger = logging.getLogger(__name__)
    server_auth._get_id_asserter = lambda: SimpleNamespace(cert=_Cert())
    server_auth._resolve_client_fqcn_for_auth = client_fqcn_resolver or (lambda client_name, _token: client_name)
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
    victim_name = _unique_fqcn("victim")
    victim = _make_running_cell(victim_name)
    peer = _make_running_cell(f"{victim_name}.peer")
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


def test_cross_client_auth_forces_server_transit_and_strips_credentials_from_peer(monkeypatch):
    server = _make_running_cell("server")
    site_a = _make_running_cell(_unique_fqcn("site_a"))
    site_b = _make_running_cell(_unique_fqcn("site_b"))
    server.core_cell.add_incoming_filter(channel="*", topic="*", cb=_make_server_auth_filter(monkeypatch))
    set_add_auth_headers_filters(site_a, site_a.get_fqcn(), "tok-a", "sig-a")
    set_add_auth_headers_filters(site_b, site_b.get_fqcn(), "tok-b", "sig-b")
    site_b.core_cell.register_request_cb(
        "peer",
        "ping",
        lambda request: Message(
            payload={
                "auth": _auth_header_values(request),
                "transit_required": request.get_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED),
                "from_cell": request.get_header(MessageHeaderKey.FROM_CELL),
            }
        ),
    )

    reply = site_a.send_request("peer", "ping", site_b.get_fqcn(), Message(payload="hello"), timeout=1.0)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload == {
        "auth": {key: None for key in AUTH_HEADERS},
        "transit_required": None,
        "from_cell": "server",
    }
    assert _auth_header_values(reply) == {key: None for key in AUTH_HEADERS}
    assert reply.get_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED) is None


def test_cross_client_auth_under_same_relay_transits_server_and_strips_credentials(monkeypatch):
    relay_name = _unique_fqcn("relay")
    site_a_name = _unique_fqcn("site_a")
    site_b_name = _unique_fqcn("site_b")
    server = _make_running_cell("server")
    _make_running_cell(relay_name)
    _make_running_cell(f"{relay_name}.{site_a_name}")
    _make_running_cell(f"{relay_name}.{site_b_name}")
    site_a_job = _make_running_cell(f"{relay_name}.{site_a_name}.job")
    site_b_job = _make_running_cell(f"{relay_name}.{site_b_name}.job")
    server.core_cell.add_incoming_filter(
        channel="*",
        topic="*",
        cb=_make_server_auth_filter(
            monkeypatch,
            client_fqcn_resolver=lambda client_name, _token: f"{relay_name}.{client_name}",
        ),
    )
    set_add_auth_headers_filters(site_a_job, site_a_name, "tok-a", "sig-a")
    set_add_auth_headers_filters(site_b_job, site_b_name, "tok-b", "sig-b")
    site_b_job.core_cell.register_request_cb(
        "peer",
        "ping",
        lambda request: Message(
            payload={
                "auth": _auth_header_values(request),
                "transit_required": request.get_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED),
                "from_cell": request.get_header(MessageHeaderKey.FROM_CELL),
            }
        ),
    )

    reply = site_a_job.send_request("peer", "ping", site_b_job.get_fqcn(), Message(payload="hello"), timeout=1.0)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload == {
        "auth": {key: None for key in AUTH_HEADERS},
        "transit_required": None,
        "from_cell": "server",
    }
    assert _auth_header_values(reply) == {key: None for key in AUTH_HEADERS}
    assert reply.get_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED) is None


def test_client_job_reply_routed_via_local_parents_authenticates_at_server(monkeypatch):
    site_a_name = _unique_fqcn("site_a")
    site_b_name = _unique_fqcn("site_b")
    site_a = _make_running_cell(site_a_name)
    site_b = _make_running_cell(site_b_name)
    site_a_job = _make_running_cell(f"{site_a_name}.job")
    site_b_job = _make_running_cell(f"{site_b_name}.job")
    server = _make_running_cell("server")

    server.core_cell.add_incoming_filter(channel="*", topic="*", cb=_make_server_auth_filter(monkeypatch))
    set_add_auth_headers_filters(site_a_job, site_a_name, "tok-a", "sig-a")
    set_add_auth_headers_filters(site_b_job, site_b_name, "tok-b", "sig-b")
    site_b_job.core_cell.register_request_cb("peer", "ping", lambda _request: Message(payload="pong"))

    routes = {
        site_a_job: {site_b_job.get_fqcn(): site_a},
        site_a: {site_b_job.get_fqcn(): server, site_a_job.get_fqcn(): site_a_job},
        server: {site_b_job.get_fqcn(): site_b, site_a_job.get_fqcn(): site_a},
        site_b: {site_b_job.get_fqcn(): site_b_job, site_a_job.get_fqcn(): server},
        site_b_job: {site_a_job.get_fqcn(): site_b},
    }
    for cell, target_routes in routes.items():
        original_find_ep = cell.core_cell._try_find_ep

        def _route(target_fqcn, for_msg, *, mapping=target_routes, fallback=original_find_ep):
            next_cell = mapping.get(target_fqcn)
            return Endpoint(next_cell.get_fqcn()) if next_cell else fallback(target_fqcn, for_msg)

        monkeypatch.setattr(cell.core_cell, "_try_find_ep", _route)

    reply = site_a_job.send_request("peer", "ping", site_b_job.get_fqcn(), Message(payload="hello"), timeout=1.0)

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload == "pong"
    assert _auth_header_values(reply) == {key: None for key in AUTH_HEADERS}


def test_server_auth_filter_strips_validated_client_reply_transit_auth(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)
    reply_msg = _make_routed_message("site-a", MessageType.REPLY, origin="site-b", with_auth=True)

    assert auth_filter(reply_msg) is None
    assert _auth_header_values(reply_msg) == {
        CellMessageAuthHeaderKey.CLIENT_NAME: None,
        CellMessageAuthHeaderKey.TOKEN: None,
        CellMessageAuthHeaderKey.TOKEN_SIGNATURE: None,
        CellMessageAuthHeaderKey.SSID: None,
    }


def test_server_auth_filter_strips_validated_client_request_transit_auth(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)
    request_msg = _make_routed_message("site-b", MessageType.REQ, with_auth=True)
    request_msg.set_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED, True)

    assert auth_filter(request_msg) is None
    assert _auth_header_values(request_msg) == {key: None for key in AUTH_HEADERS}
    assert request_msg.get_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED) is None


def test_server_auth_filter_strips_transit_auth_between_sites_under_same_relay(monkeypatch):
    relay = _unique_fqcn("relay")
    site_a = _unique_fqcn("site_a")
    site_b = _unique_fqcn("site_b")
    origin = f"{relay}.{site_a}.job"
    destination = f"{relay}.{site_b}.job"
    auth_filter = _make_server_auth_filter(
        monkeypatch,
        client_fqcn_resolver=lambda client_name, _token: f"{relay}.{client_name}",
    )
    request_msg = _make_routed_message(destination, MessageType.REQ, origin=origin)
    request_msg.add_headers(
        {
            CellMessageAuthHeaderKey.CLIENT_NAME: site_a,
            CellMessageAuthHeaderKey.TOKEN: "token-site-a",
            CellMessageAuthHeaderKey.TOKEN_SIGNATURE: "sig-site-a",
            MessageHeaderKey.SERVER_TRANSIT_REQUIRED: True,
        }
    )

    assert auth_filter(request_msg) is None
    assert _auth_header_values(request_msg) == {key: None for key in AUTH_HEADERS}
    assert request_msg.get_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED) is None


def test_server_auth_filter_rejects_untracked_unauthenticated_client_reply_transit(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)

    reply = auth_filter(_make_routed_message("site-a", MessageType.REPLY, origin="site-b"))

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_server_auth_filter_keeps_auth_on_validated_server_destination_reply(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)
    reply_msg = _make_routed_message("server.job-1", MessageType.REPLY, origin="site-b", with_auth=True)

    assert auth_filter(reply_msg) is None
    assert _auth_header_values(reply_msg) == {
        CellMessageAuthHeaderKey.CLIENT_NAME: "site-b",
        CellMessageAuthHeaderKey.TOKEN: "token-site-b",
        CellMessageAuthHeaderKey.TOKEN_SIGNATURE: "sig-site-b",
        CellMessageAuthHeaderKey.SSID: None,
    }


def test_server_auth_filter_still_rejects_unauthenticated_client_request_transit(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)

    reply = auth_filter(_make_routed_message("site-b", MessageType.REQ))

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_server_auth_filter_still_rejects_unauthenticated_server_destination(monkeypatch):
    auth_filter = _make_server_auth_filter(monkeypatch)

    reply = auth_filter(_make_routed_message("server.job-1", MessageType.REPLY))

    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.UNAUTHENTICATED


def test_auth_filter_keeps_auth_on_outgoing_requests():
    victim_name = _unique_fqcn("victim")
    victim = _make_running_cell(victim_name)
    peer = _make_running_cell(f"{victim_name}.peer")
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


def test_auth_filter_keeps_cross_site_reply_auth_when_routed_via_local_parent():
    origin = f"site-a.{_unique_fqcn('job')}"
    victim = _make_running_cell(origin)
    set_add_auth_headers_filters(victim, "site-a", "tok-a", "sig-a", "ssid-a")
    reply = Message(
        headers={
            MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
            MessageHeaderKey.ORIGIN: origin,
            MessageHeaderKey.DESTINATION: "site-b.job-1",
            MessageHeaderKey.TO_CELL: "site-a",
        }
    )

    for callback in victim.core_cell.out_reply_filter_reg.find("peer", "reply"):
        callback.cb(reply, *callback.args, **callback.kwargs)

    assert _auth_header_values(reply) == {
        CellMessageAuthHeaderKey.CLIENT_NAME: "site-a",
        CellMessageAuthHeaderKey.TOKEN: "tok-a",
        CellMessageAuthHeaderKey.TOKEN_SIGNATURE: "sig-a",
        CellMessageAuthHeaderKey.SSID: "ssid-a",
    }


def test_auth_filter_keeps_cross_site_reply_auth_under_same_relay():
    relay = _unique_fqcn("relay")
    origin = f"{relay}.site-a.job"
    victim = _make_running_cell(origin)
    set_add_auth_headers_filters(victim, "site-a", "tok-a", "sig-a", "ssid-a")
    reply = Message(
        headers={
            MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
            MessageHeaderKey.ORIGIN: origin,
            MessageHeaderKey.DESTINATION: f"{relay}.site-b.job",
            MessageHeaderKey.TO_CELL: f"{relay}.site-a",
        }
    )

    for callback in victim.core_cell.out_reply_filter_reg.find("peer", "reply"):
        callback.cb(reply, *callback.args, **callback.kwargs)

    assert _auth_header_values(reply) == {
        CellMessageAuthHeaderKey.CLIENT_NAME: "site-a",
        CellMessageAuthHeaderKey.TOKEN: "tok-a",
        CellMessageAuthHeaderKey.TOKEN_SIGNATURE: "sig-a",
        CellMessageAuthHeaderKey.SSID: "ssid-a",
    }
    assert reply.get_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED) is True


def test_auth_filter_does_not_add_auth_to_same_site_reply_via_parent():
    origin = f"site-a.{_unique_fqcn('job')}"
    victim = _make_running_cell(origin)
    set_add_auth_headers_filters(victim, "site-a", "tok-a", "sig-a", "ssid-a")
    reply = Message(
        headers={
            MessageHeaderKey.MSG_TYPE: MessageType.REPLY,
            MessageHeaderKey.ORIGIN: origin,
            MessageHeaderKey.DESTINATION: "site-a.job-2",
            MessageHeaderKey.TO_CELL: "site-a",
        }
    )

    for callback in victim.core_cell.out_reply_filter_reg.find("peer", "reply"):
        callback.cb(reply, *callback.args, **callback.kwargs)

    assert _auth_header_values(reply) == {key: None for key in AUTH_HEADERS}


@pytest.mark.parametrize(
    ("origin", "destination", "client_name", "expected"),
    [
        ("relay-1.site-a.job-1", "relay-1.site-b.job-2", "site-a", True),
        ("relay-1.site-a.job-1", "relay-1.site-a.job-2", "site-a", False),
    ],
)
def test_cross_client_family_uses_authenticated_site_identity(origin, destination, client_name, expected):
    assert is_cross_client_family(origin, destination, client_name) is expected
