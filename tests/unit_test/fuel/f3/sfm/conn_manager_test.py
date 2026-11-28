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

from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint, EndpointState
from nvflare.fuel.f3.sfm.conn_manager import ConnManager


@pytest.mark.parametrize("binding", ["certificate", "connection"])
def test_pinned_reply_is_only_sent_to_the_authenticated_peer(binding):
    def selected_connection(cert):
        props = {DriverParams.PEER_CERT.value: cert} if binding == "certificate" else {}
        return SimpleNamespace(
            conn=SimpleNamespace(
                get_conn_properties=lambda: props,
                connector=SimpleNamespace(driver=Mock()),
            ),
            send_data=Mock(),
        )

    original = selected_connection(b"original")
    replacement = selected_connection(b"replacement")
    peer = sha256(b"original").digest() if binding == "certificate" else original.conn
    endpoint = Endpoint("admin")
    endpoint.state = EndpointState.READY
    route = Mock(endpoint=endpoint)
    route.get_connection.return_value = original
    manager = object.__new__(ConnManager)
    manager.local_endpoint = Endpoint("server")
    manager.sfm_endpoints = {endpoint.name: route}
    manager.send_frame_stats = Mock()

    manager.send_message(endpoint, 1, {}, b"reply", target_peer=peer)
    original.send_data.assert_called_once()

    route.get_connection.return_value = replacement
    with pytest.raises(CommError, match="Authenticated peer changed"):
        manager.send_message(endpoint, 1, {}, b"reply", target_peer=peer)
    replacement.send_data.assert_not_called()
