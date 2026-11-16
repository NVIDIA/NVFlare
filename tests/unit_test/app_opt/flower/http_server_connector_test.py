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
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode
from nvflare.app_opt.flower.connectors.http_server_connector import HttpServerConnector
from nvflare.app_opt.flower.utils import request_to_shareable, shareable_to_response

_HTTP_CONNECTION = "nvflare.app_opt.flower.connectors.http_server_connector.http.client.HTTPConnection"


class _FakeUpstreamResponse:
    def __init__(self, status, headers, body):
        self.status = status
        self._headers = headers
        self._body = body

    def getheaders(self):
        return self._headers

    def read(self):
        return self._body


class _FakeUpstream:
    """Stands in for http.client.HTTPConnection: records the request and replays a scripted response."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.target = None
        self.requests = []
        self.closed = 0

    def __call__(self, host, port, timeout=None):
        self.target = (host, port)
        return self

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, body, headers))
        if self.error:
            raise self.error

    def getresponse(self):
        return self.response

    def close(self):
        self.closed += 1


def _connector(applet_stopped=False, addr="127.0.0.1:9091"):
    connector = HttpServerConnector()
    connector.superlink_addr = addr
    connector.is_applet_stopped = MagicMock(return_value=(applet_stopped, 0))
    return connector


def test_request_is_replayed_as_post_and_response_is_preserved():
    upstream = _FakeUpstream(_FakeUpstreamResponse(200, [("content-type", "application/protobuf")], b"reply"))
    request = request_to_shareable(
        "/v1/fleet/pull-messages", {"flwr-timestamp": "7", "Content-Length": "5", "Host": "proxy"}, b"hello"
    )

    with patch(_HTTP_CONNECTION, upstream):
        reply = _connector().send_request_to_flower(request, FLContext())

    assert upstream.target == ("127.0.0.1", 9091)
    assert upstream.requests == [("POST", "/v1/fleet/pull-messages", b"hello", {"flwr-timestamp": "7"})]
    assert upstream.closed == 1
    assert shareable_to_response(reply) == (200, {"content-type": "application/protobuf"}, b"reply")


@pytest.mark.parametrize("status", [400, 401, 404, 500])
def test_superlink_error_status_and_body_are_preserved(status):
    """An error from the SuperLink is a normal reply: the SuperNode must see the original status and body."""
    upstream = _FakeUpstream(_FakeUpstreamResponse(status, [("content-type", "application/json")], b"{}"))

    with patch(_HTTP_CONNECTION, upstream):
        reply = _connector().send_request_to_flower(request_to_shareable("/v1/fleet/get-run", {}, b""), FLContext())

    assert reply.get_return_code() == ReturnCode.OK
    assert shareable_to_response(reply) == (status, {"content-type": "application/json"}, b"{}")


@pytest.mark.parametrize("path", ["/v1/control/stop-run", "/v1/runtime/x", "/health", "/v1/fleet/../control/x", None])
def test_non_fleet_paths_are_never_sent_to_the_superlink(path):
    upstream = _FakeUpstream(_FakeUpstreamResponse(200, [], b""))
    request = request_to_shareable("/v1/fleet/x", {}, b"")
    request["flower.name"] = path

    with patch(_HTTP_CONNECTION, upstream):
        reply = _connector().send_request_to_flower(request, FLContext())

    assert reply.get_return_code() == ReturnCode.BAD_REQUEST_DATA
    assert upstream.requests == []
    assert upstream.target is None


def test_unreachable_superlink_gives_service_unavailable():
    upstream = _FakeUpstream(error=ConnectionRefusedError("down"))

    with patch(_HTTP_CONNECTION, upstream):
        reply = _connector().send_request_to_flower(request_to_shareable("/v1/fleet/x", {}, b""), FLContext())

    assert reply.get_return_code() == ReturnCode.SERVICE_UNAVAILABLE
    assert upstream.closed == 1


def test_stopped_applet_gives_service_unavailable_without_contacting_superlink():
    upstream = _FakeUpstream(_FakeUpstreamResponse(200, [], b""))

    with patch(_HTTP_CONNECTION, upstream):
        reply = _connector(applet_stopped=True).send_request_to_flower(
            request_to_shareable("/v1/fleet/x", {}, b""), FLContext()
        )

    assert reply.get_return_code() == ReturnCode.SERVICE_UNAVAILABLE
    assert upstream.target is None


def test_start_allocates_one_address_and_passes_it_to_the_applet():
    connector = HttpServerConnector()
    connector.num_rounds = 2
    connector.start_applet = MagicMock()

    with patch(
        "nvflare.app_opt.flower.connectors.http_server_connector.get_local_addresses", return_value=["127.0.0.1:5555"]
    ) as get_addresses:
        connector.start(FLContext())

    get_addresses.assert_called_once_with(1)
    app_ctx = connector.start_applet.call_args.args[0]
    assert app_ctx["flower_superlink_api_addr"] == "127.0.0.1:5555"
    assert connector.superlink_addr == "127.0.0.1:5555"
