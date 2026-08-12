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

from unittest.mock import MagicMock

import pytest

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.cellnet.connector_manager import ConnectorManager


def _config(resources):
    config = MagicMock()
    config.get_backbone_connection_generation.return_value = 2
    config.get_internal_connection_scheme.return_value = "tcp"
    config.allow_adhoc_connections.return_value = False
    config.get_adhoc_connection_scheme.return_value = "tcp"
    config.get_config.return_value = {
        "internal": {"scheme": "tcp", "resources": resources},
    }
    return config


@pytest.mark.parametrize(
    "listen_host",
    [
        "localhost",
        "LOCALHOST.",
        "127.0.0.1",
        "127.255.255.254",
        "::1",
        "0:0:0:0:0:0:0:1",
        "[::1]",
        "::ffff:127.0.0.1",
        "::ffff:7f00:1",
    ],
)
def test_equivalent_ipv4_and_ipv6_loopback_listeners_allow_clear_transport(listen_host):
    manager = ConnectorManager(
        communicator=MagicMock(),
        secure=False,
        comm_configurator=_config(
            {
                "host": "parent-service",
                "listen_host": listen_host,
                "connection_security": "clear",
            }
        ),
    )

    assert manager.int_resources["listen_host"] == listen_host


@pytest.mark.parametrize("listen_host", ["0.0.0.0", "::", "::ffff:192.0.2.1", "parent-service"])
@pytest.mark.parametrize("connection_security", ["clear", "tls"])
def test_remote_internal_listener_requires_mtls(listen_host, connection_security):
    with pytest.raises(ConfigError, match="require connection_security='mtls'"):
        ConnectorManager(
            communicator=MagicMock(),
            secure=False,
            comm_configurator=_config(
                {
                    "host": "parent-service",
                    "listen_host": listen_host,
                    "connection_security": connection_security,
                }
            ),
        )


def test_remote_internal_listener_is_allowed_with_mtls():
    manager = ConnectorManager(
        communicator=MagicMock(),
        secure=True,
        comm_configurator=_config({"host": "parent-service", "listen_host": "0.0.0.0", "connection_security": "mtls"}),
    )

    assert manager.int_resources["connection_security"] == "mtls"
