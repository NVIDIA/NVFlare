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
import io
from http.client import HTTPResponse
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.app_opt.flower.connectors.http_client_connector import HttpClientConnector, _ProxyHandler
from nvflare.app_opt.flower.utils import response_to_shareable, shareable_to_request


class _FakeSocket:
    """A connected socket whose inbound bytes are scripted and whose outbound bytes are captured."""

    def __init__(self, inbound: bytes):
        self.inbound = io.BytesIO(inbound)
        self.outbound = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self.inbound if "r" in mode else self.outbound

    def sendall(self, data):
        self.outbound.write(data)


class _NoCloseBytesIO(io.BytesIO):
    """HTTPResponse closes its file after each response; keep it readable for the next one."""

    def close(self):
        pass


class _FakeConnector:
    """Records what the proxy forwards and replies with a scripted (status, headers, body)."""

    def __init__(self, reply=(200, {"content-type": "application/protobuf"}, b"ok")):
        self.reply = reply
        self.calls = []
        self.logger = MagicMock()

    def forward(self, path, headers, body):
        self.calls.append((path, dict(headers.items()), body))
        return self.reply


def _post(path, body=b"", extra_headers=""):
    return (
        f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/protobuf\r\n"
        f"Content-Length: {len(body)}\r\n{extra_headers}\r\n"
    ).encode() + body


def _serve(raw: bytes, connector):
    """Run the proxy handler over the raw bytes and return the parsed HTTP responses."""
    handler_cls = type("TestHandler", (_ProxyHandler,), {"connector": connector})
    sock = _FakeSocket(raw)
    handler_cls(sock, ("127.0.0.1", 1234), MagicMock())

    out = _NoCloseBytesIO(sock.outbound.getvalue())

    class _Reader:
        def makefile(self, *args, **kwargs):
            return out

    responses = []
    while out.tell() < len(out.getvalue()):
        r = HTTPResponse(_Reader())
        r.begin()
        responses.append((r.status, dict((k.lower(), v) for k, v in r.getheaders()), r.read()))
    return responses


def test_fleet_post_is_forwarded_and_reply_written_back():
    connector = _FakeConnector()

    responses = _serve(_post("/v1/fleet/pull-messages", b"abc", "flwr-timestamp: 5\r\n"), connector)

    assert len(connector.calls) == 1
    path, headers, body = connector.calls[0]
    assert (path, body) == ("/v1/fleet/pull-messages", b"abc")
    assert headers["flwr-timestamp"] == "5"  # request headers reach the connector
    assert headers["Content-Type"] == "application/protobuf"
    status, out_headers, out_body = responses[0]
    assert (status, out_body) == (200, b"ok")
    assert out_headers["content-type"] == "application/protobuf"
    assert out_headers["content-length"] == "2"


def test_connection_is_kept_alive_for_consecutive_requests():
    """The SuperNode issues heartbeats and pulls on a shared connection; HTTP/1.1 keep-alive must work."""
    connector = _FakeConnector()
    raw = _post("/v1/fleet/pull-messages", b"1") + _post("/v1/fleet/send-node-heartbeat", b"22")

    responses = _serve(raw, connector)

    assert [c[0] for c in connector.calls] == ["/v1/fleet/pull-messages", "/v1/fleet/send-node-heartbeat"]
    assert [c[2] for c in connector.calls] == [b"1", b"22"]
    assert [r[0] for r in responses] == [200, 200]


@pytest.mark.parametrize("status", [401, 404, 500])
def test_error_status_from_superlink_is_written_as_is(status):
    connector = _FakeConnector((status, {"content-type": "application/json"}, b'{"detail":"no"}'))

    responses = _serve(_post("/v1/fleet/get-run"), connector)

    assert responses[0][0] == status
    assert responses[0][2] == b'{"detail":"no"}'


@pytest.mark.parametrize("path", ["/health", "/v1/control/stop-run", "/v1/runtime/x", "/"])
def test_post_outside_fleet_api_gets_404_and_is_not_forwarded(path):
    connector = _FakeConnector()

    responses = _serve(_post(path, b"x"), connector)

    assert connector.calls == []
    assert responses[0][0] == 404


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "HEAD"])
def test_non_post_methods_get_404_and_are_not_forwarded(method):
    connector = _FakeConnector()

    responses = _serve(f"{method} /v1/fleet/pull-messages HTTP/1.1\r\nHost: x\r\n\r\n".encode(), connector)

    assert connector.calls == []
    assert responses[0][0] == 404


def test_chunked_request_is_refused():
    connector = _FakeConnector()
    raw = b"POST /v1/fleet/x HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n"

    responses = _serve(raw, connector)

    assert connector.calls == []
    assert responses[0][0] == 411


def test_bad_content_length_is_refused():
    connector = _FakeConnector()

    responses = _serve(b"POST /v1/fleet/x HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n", connector)

    assert connector.calls == []
    assert responses[0][0] == 400


def _client_connector(reply: Shareable = None, error: Exception = None):
    connector = HttpClientConnector()
    sent = []

    def send(request):
        sent.append(request)
        if error:
            raise error
        return reply

    connector._send_flower_request = send
    connector.abort_signal = MagicMock()
    connector.engine = MagicMock()
    connector.system_panic = MagicMock()
    return connector, sent


def test_forward_sends_request_to_server_and_returns_http_reply():
    connector, sent = _client_connector(response_to_shareable(200, {"x-a": "1"}, b"body"))

    status, headers, body = connector.forward("/v1/fleet/get-fab", {"flwr-timestamp": "9", "Host": "h"}, b"req")

    assert (status, headers, body) == (200, {"x-a": "1"}, b"body")
    assert shareable_to_request(sent[0]) == ("/v1/fleet/get-fab", {"flwr-timestamp": "9"}, b"req")
    assert not connector._stopping


def test_forward_returns_503_and_starts_stopping_when_server_side_has_ended():
    connector, _ = _client_connector(make_reply(ReturnCode.SERVICE_UNAVAILABLE))

    status, _, _ = connector.forward("/v1/fleet/pull-messages", {}, b"")

    assert status == 503
    assert connector._stopping
    # _is_stopped drives the shutdown of the connector (and with it the SuperNode)
    connector.stop = MagicMock()
    connector.is_applet_stopped = MagicMock(return_value=(False, 0))
    assert connector._is_stopped() == (True, 0)
    connector.stop.assert_called_once()


def test_forward_returns_503_without_contacting_server_once_stopped():
    connector, sent = _client_connector(response_to_shareable(200, {}, b""))
    connector.stopped = True

    status, _, _ = connector.forward("/v1/fleet/pull-messages", {}, b"")

    assert status == 503
    assert sent == []
    assert connector._stopping


def test_forward_aborts_and_returns_503_when_sending_fails():
    connector, _ = _client_connector(error=RuntimeError("boom"))

    status, _, _ = connector.forward("/v1/fleet/pull-messages", {}, b"")

    assert status == 503
    assert connector._stopping
    connector.abort_signal.trigger.assert_called_once_with(True)
    connector.system_panic.assert_called_once()


def test_stop_terminates_the_applet_without_waiting_for_an_exit_signal(monkeypatch):
    """A 1.43 SuperNode retries forever and has no should-exit channel, so the applet must terminate it."""
    connector = HttpClientConnector(client_shutdown_timeout=0.5)
    connector.stop_applet = MagicMock()
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    connector.stop(FLContext())

    connector.stop_applet.assert_called_once_with(0.5)
    assert sleeps == []
    assert connector.stopped
