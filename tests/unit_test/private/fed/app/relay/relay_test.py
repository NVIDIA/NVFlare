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

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.private.fed.app.relay.relay import _validate_auth_headers


def _relay_core_cell(fqcn="relay-1"):
    core_cell = CoreCell.__new__(CoreCell)
    core_cell.my_info = FqcnInfo(fqcn)
    return core_cell


def _server_transit_message(incoming_endpoint):
    message = Message(
        headers={
            MessageHeaderKey.SERVER_TRANSIT_REQUIRED: True,
            MessageHeaderKey.ROUTE: [("site-1", 0.0), ("server", 1.0)],
        }
    )
    message.set_prop(MessagePropKey.ENDPOINT, Endpoint(incoming_endpoint))
    return message


@pytest.mark.parametrize(("relay_fqcn", "upstream_fqcn"), [("relay-1", "server"), ("relay-1.relay-2", "relay-1")])
def test_relay_accepts_sanitized_server_transit_from_configured_upstream(relay_fqcn, upstream_fqcn):
    core_cell = _relay_core_cell(relay_fqcn)
    message = _server_transit_message(upstream_fqcn)

    with patch("nvflare.private.fed.app.relay.relay.validate_auth_headers") as validate:
        result = _validate_auth_headers(
            message,
            token_verifier=MagicMock(),
            logger=MagicMock(),
            local_cell_fqcn=relay_fqcn,
            core_cell=core_cell,
        )

    assert result is None
    validate.assert_not_called()


def test_relay_does_not_trust_forged_server_route_from_child():
    core_cell = _relay_core_cell()
    message = _server_transit_message("relay-1.site-1")
    expected = Message(payload="rejected")

    with patch("nvflare.private.fed.app.relay.relay.validate_auth_headers", return_value=expected) as validate:
        result = _validate_auth_headers(
            message,
            token_verifier=MagicMock(),
            logger=MagicMock(),
            local_cell_fqcn="relay-1",
            core_cell=core_cell,
        )

    assert result is expected
    validate.assert_called_once()
