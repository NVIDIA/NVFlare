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
import pytest

from nvflare.apis.shareable import Shareable
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.utils import (
    filter_headers,
    is_fleet_path,
    request_to_shareable,
    response_to_shareable,
    shareable_to_request,
    shareable_to_response,
)


def test_filter_headers_drops_hop_by_hop_and_framing_headers_only():
    headers = {
        "Content-Type": "application/protobuf",
        "flwr-public-key": "abc",
        "Connection": "keep-alive",
        "Keep-Alive": "timeout=5",
        "Transfer-Encoding": "chunked",
        "Content-Length": "12",
        "Host": "127.0.0.1:9092",
        "Upgrade": "h2c",
    }

    assert filter_headers(headers) == {"Content-Type": "application/protobuf", "flwr-public-key": "abc"}


def test_filter_headers_accepts_pairs_and_none():
    assert filter_headers([("a", "1"), ("Content-Length", "2")]) == {"a": "1"}
    assert filter_headers(None) == {}


def test_request_round_trip_preserves_path_headers_and_body():
    body = bytes(range(256))
    s = request_to_shareable("/v1/fleet/pull-messages", {"flwr-timestamp": "1", "Host": "x"}, body)

    assert isinstance(s, Shareable)
    assert shareable_to_request(s) == ("/v1/fleet/pull-messages", {"flwr-timestamp": "1"}, body)


def test_response_round_trip_preserves_status_headers_and_body():
    s = response_to_shareable(401, [("content-type", "application/json"), ("content-length", "5")], b"error")

    assert s[Constant.PARAM_KEY_STATUS] == 401
    assert shareable_to_response(s) == (401, {"content-type": "application/json"}, b"error")


@pytest.mark.parametrize("status", [None, "200", 99, 600, True])
def test_shareable_to_response_rejects_invalid_status(status):
    s = Shareable()
    if status is not None:
        s[Constant.PARAM_KEY_STATUS] = status

    with pytest.raises(ValueError, match="invalid HTTP status"):
        shareable_to_response(s)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/fleet/register-node",
        "/v1/fleet/pull-messages",
        "/v1/fleet/send-node-heartbeat",
        "/v1/fleet/push-object",
    ],
)
def test_is_fleet_path_accepts_fleet_operations(path):
    assert is_fleet_path(path)


@pytest.mark.parametrize(
    "path",
    [
        None,
        "",
        "/health",
        "/v1/control/start-run",
        "/v1/runtime/pull-serverapp-inputs",
        "/v1/fleet/",
        "/v1/fleet",
        "/v1/fleet/../control/stop-run",
        "/v1/fleet/%2e%2e/control",
        "/v1/fleet/pull-messages?x=1",
        "/v1/fleet/a b",
        "http://evil/v1/fleet/pull-messages",
        "//v1/fleet/pull-messages",
        "/v1/fleet/pull-messages\r\nHost: evil",
    ],
)
def test_is_fleet_path_rejects_everything_else(path):
    """The SuperLink port also serves the Control and Runtime APIs; only Fleet paths may be tunneled."""
    assert not is_fleet_path(path)
