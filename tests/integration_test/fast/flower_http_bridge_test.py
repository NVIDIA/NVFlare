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
"""The Flower HTTP bridge over real sockets, with a fake SuperLink and the FLARE hop replaced by a direct call.

Needs no Flower installation: the bridge forwards opaque bytes. The unit tests cover the proxy and connector
logic with fakes; these cases prove that the real HTTP server and client behave as the SuperNode needs.
"""

import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.flower.connectors.http_client_connector import HttpClientConnector
from nvflare.app_opt.flower.connectors.http_server_connector import HttpServerConnector
from nvflare.fuel.utils.network_utils import get_open_ports

_WAIT = 10.0


class _FakeSuperLink:
    """HTTP server standing in for the Flower SuperLink. Records requests and can hold /v1/fleet/pull-messages."""

    def __init__(self):
        self.requests = []
        self.pull_entered = threading.Event()
        self.release_pull = threading.Event()
        self.next_response = None  # (status, headers, body) override
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append((self.path, dict(self.headers.items()), body))
                if self.path == "/v1/fleet/pull-messages":
                    outer.pull_entered.set()
                    outer.release_pull.wait(_WAIT)
                status, headers, payload = outer.next_response or (200, {"content-type": "application/protobuf"}, body)
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        port = get_open_ports(1)[0]
        self.addr = f"127.0.0.1:{port}"
        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.release_pull.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(_WAIT)


@pytest.fixture
def bridge():
    """Yield (proxy_addr, superlink, client_connector); everything is stopped in the finally block."""
    superlink = _FakeSuperLink()
    server_connector = HttpServerConnector()
    server_connector.superlink_addr = superlink.addr
    server_connector.is_applet_stopped = MagicMock(return_value=(False, 0))

    client_connector = HttpClientConnector()
    client_connector._send_flower_request = lambda request: server_connector.send_request_to_flower(
        request, FLContext()
    )
    client_connector.stop_applet = MagicMock()
    client_connector.is_applet_stopped = MagicMock(return_value=(False, 0))
    proxy_addr = f"127.0.0.1:{get_open_ports(1)[0]}"
    try:
        client_connector._start_proxy(proxy_addr)
        yield proxy_addr, superlink, client_connector
    finally:
        superlink.close()
        client_connector.stop(None)


def _request(addr, method, path, body=b"", headers=None, conn=None):
    own = conn is None
    conn = conn or http.client.HTTPConnection(*addr.split(":"), timeout=_WAIT)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    finally:
        if own:
            conn.close()


def test_request_reaches_superlink_with_headers_and_response_returns_intact(bridge):
    proxy_addr, superlink, _ = bridge
    superlink.next_response = (200, {"content-type": "application/protobuf", "x-flwr-extra": "yes"}, b"response-bytes")

    status, headers, body = _request(
        proxy_addr,
        "POST",
        "/v1/fleet/send-node-heartbeat",
        b"\x00\x01request",
        {"content-type": "application/protobuf", "flwr-component-name": "SuperNode", "flwr-signature": "sig"},
    )

    assert (status, body) == (200, b"response-bytes")
    assert headers["content-type"] == "application/protobuf"
    assert headers["x-flwr-extra"] == "yes"
    path, seen_headers, seen_body = superlink.requests[0]
    assert (path, seen_body) == ("/v1/fleet/send-node-heartbeat", b"\x00\x01request")
    seen = {k.lower(): v for k, v in seen_headers.items()}
    assert seen["flwr-component-name"] == "SuperNode" and seen["flwr-signature"] == "sig"
    assert seen["content-type"] == "application/protobuf"
    # framing is each hop's own: the SuperLink sees the length of the body it actually received
    assert seen["content-length"] == str(len(b"\x00\x01request"))


def test_large_body_survives_the_round_trip(bridge):
    proxy_addr, superlink, _ = bridge
    payload = bytes(range(256)) * (8 * 1024 * 1024 // 256)

    status, _, body = _request(proxy_addr, "POST", "/v1/fleet/push-object", payload)

    assert status == 200 and body == payload
    assert superlink.requests[0][2] == payload


@pytest.mark.parametrize("status", [401, 404, 500])
def test_superlink_errors_reach_the_supernode_unchanged(bridge, status):
    proxy_addr, superlink, connector = bridge
    superlink.next_response = (status, {"content-type": "application/json"}, b'{"detail":"x"}')

    got_status, headers, body = _request(proxy_addr, "POST", "/v1/fleet/get-run", b"r")

    assert (got_status, body) == (status, b'{"detail":"x"}')
    assert not connector._stopping  # an application-level error does not stop the bridge


def test_slow_pull_does_not_block_concurrent_requests_and_connection_is_reused(bridge):
    proxy_addr, superlink, _ = bridge
    results = {}

    def pull():
        results["pull"] = _request(proxy_addr, "POST", "/v1/fleet/pull-messages", b"pull")

    worker = threading.Thread(target=pull, daemon=True)
    try:
        worker.start()
        assert superlink.pull_entered.wait(_WAIT), "pull never reached the SuperLink"

        # a heartbeat on its own connection completes while the pull is still in flight
        conn = http.client.HTTPConnection(*proxy_addr.split(":"), timeout=_WAIT)
        try:
            assert _request(proxy_addr, "POST", "/v1/fleet/send-node-heartbeat", b"a", conn=conn)[2] == b"a"
            # and the same connection serves a second request (keep-alive)
            assert _request(proxy_addr, "POST", "/v1/fleet/send-node-heartbeat", b"b", conn=conn)[2] == b"b"
        finally:
            conn.close()
        assert "pull" not in results
    finally:
        superlink.release_pull.set()
        worker.join(_WAIT)

    assert not worker.is_alive()
    assert results["pull"][2] == b"pull"


@pytest.mark.parametrize(
    "method,path",
    [("GET", "/health"), ("POST", "/v1/control/stop-run"), ("GET", "/v1/fleet/pull-messages")],
)
def test_requests_outside_the_fleet_api_are_refused_locally(bridge, method, path):
    proxy_addr, superlink, _ = bridge

    status, _, _ = _request(proxy_addr, method, path, b"x" if method == "POST" else b"")

    assert status == 404
    assert superlink.requests == []


def test_stopped_bridge_answers_503_and_begins_stopping(bridge):
    proxy_addr, superlink, connector = bridge
    connector.stopped = True

    status, _, _ = _request(proxy_addr, "POST", "/v1/fleet/pull-messages", b"x")

    assert status == 503
    assert connector._stopping
    assert superlink.requests == []


def test_superlink_that_is_gone_answers_503_and_begins_stopping(bridge):
    proxy_addr, superlink, connector = bridge
    superlink.close()

    status, _, _ = _request(proxy_addr, "POST", "/v1/fleet/pull-messages", b"x")

    assert status == 503
    assert connector._stopping
