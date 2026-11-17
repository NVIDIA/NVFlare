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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.flower.connectors.http_server_connector import HttpServerConnector
from nvflare.app_opt.flower.http_transport import http_response


def context(authenticated_site, claimed_site="site-1"):
    ctx = FLContext()
    if authenticated_site is not None:
        ctx.set_prop(FLContextKey.CLIENT_NAME, authenticated_site, private=True, sticky=False)
    if claimed_site is not None:
        peer = FLContext()
        peer.set_public_props({ReservedKey.IDENTITY_NAME: claimed_site, FLContextKey.CLIENT_NAME: claimed_site})
        ctx.set_peer_context(peer)
    return ctx


def request(path="/v1/fleet/register-node"):
    return Shareable(
        {
            "method": "POST",
            "path": path,
            "body": b"protobuf",
            "headers": {
                "content-type": "application/protobuf",
                "flwr-public-key": "node-key",
                "flwr-signature": "signature",
                "flwr-timestamp": "timestamp",
            },
        }
    )


@pytest.fixture
def connector():
    connector = HttpServerConnector()
    connector.http_client = Mock()
    connector.http_client.send.return_value = http_response(200)
    return connector


@pytest.mark.parametrize("path", ["/v1/fleet/register-node", "/v1/fleet/deactivate-node"])
def test_claimed_owner_cannot_override_authenticated_site(connector, path):
    assert connector.send_request_to_flower(request(), context("site-1"))["status"] == 200
    connector.http_client.send.reset_mock()

    assert connector.send_request_to_flower(request(path), context("site-2", "site-1"))["status"] == 403
    connector.http_client.send.assert_not_called()
    assert connector._key_owners == {"node-key": "site-1"}
    assert connector.send_request_to_flower(request(path), context("site-1"))["status"] == 200


@pytest.mark.parametrize("authenticated_site", [None, "", True, 7, ["site-1"], {"name": "site-1"}])
@pytest.mark.parametrize("registered", [False, True])
def test_missing_or_invalid_authenticated_site_fails_closed(connector, authenticated_site, registered):
    if registered:
        assert connector.send_request_to_flower(request(), context("site-1"))["status"] == 200
    owners = connector._key_owners.copy()
    connector.http_client.send.reset_mock()
    path = "/v1/fleet/deactivate-node" if registered else "/v1/fleet/register-node"

    assert connector.send_request_to_flower(request(path), context(authenticated_site))["status"] == 403
    connector.http_client.send.assert_not_called()
    assert connector._key_owners == owners


@pytest.mark.parametrize("claimed_site", ["site-2", None])
def test_registration_and_access_use_only_authenticated_site(connector, claimed_site):
    assert connector.send_request_to_flower(request(), context("site-1", claimed_site))["status"] == 200
    assert connector._key_owners == {"node-key": "site-1"}
    assert connector.send_request_to_flower(request("/v1/fleet/deactivate-node"), context("site-1"))["status"] == 200
    assert connector.http_client.send.call_count == 2
