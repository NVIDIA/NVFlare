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

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.cellnet.connector_manager import ConnectorManager


def _config(resources, scheme="tcp"):
    config = MagicMock()
    config.get_backbone_connection_generation.return_value = 2
    config.get_internal_connection_scheme.return_value = scheme
    config.allow_adhoc_connections.return_value = False
    config.get_adhoc_connection_scheme.return_value = "tcp"
    config.get_config.return_value = {
        "internal": {"scheme": scheme, "resources": resources},
    }
    return config


def test_default_internal_listener_uses_ipv4_loopback():
    config = MagicMock()
    config.get_backbone_connection_generation.return_value = 2
    config.get_internal_connection_scheme.return_value = "tcp"
    config.allow_adhoc_connections.return_value = False
    config.get_adhoc_connection_scheme.return_value = "tcp"
    config.get_config.return_value = None

    manager = ConnectorManager(communicator=MagicMock(), secure=False, comm_configurator=config)

    assert manager.int_resources["host"] == "127.0.0.1"


def test_secure_internal_network_listener_defaults_to_mtls_on_loopback():
    config = MagicMock()
    config.get_backbone_connection_generation.return_value = 2
    config.get_internal_connection_scheme.return_value = "tcp"
    config.allow_adhoc_connections.return_value = False
    config.get_adhoc_connection_scheme.return_value = "tcp"
    config.get_config.return_value = None

    manager = ConnectorManager(communicator=MagicMock(), secure=True, comm_configurator=config)

    assert manager.int_resources["connection_security"] == ConnectionSecurity.MTLS


@pytest.mark.parametrize("connection_security", ["clear", "tls"])
def test_secure_internal_network_listener_rejects_unauthenticated_loopback(connection_security):
    with pytest.raises(ConfigError, match="secure-mode internal CellNet listeners require"):
        ConnectorManager(
            communicator=MagicMock(),
            secure=True,
            comm_configurator=_config(
                {"host": "127.0.0.1", "listen_host": "127.0.0.1", "connection_security": connection_security}
            ),
        )


def test_secure_shared_file_internal_listener_does_not_require_mtls():
    manager = ConnectorManager(
        communicator=MagicMock(),
        secure=True,
        comm_configurator=_config({"root_dir": "/tmp/cellnet", "connection_security": "clear"}, scheme="shared-file"),
    )

    assert manager.int_resources["connection_security"] == ConnectionSecurity.CLEAR


@pytest.mark.parametrize(
    "listen_host",
    [
        "localhost",
        "LOCALHOST.",
        "127.0.0.1",
        "127.255.255.254",
    ],
)
def test_equivalent_ipv4_loopback_listeners_allow_clear_transport(listen_host):
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


@pytest.mark.parametrize("listen_host", ["::1", "0:0:0:0:0:0:0:1", "[::1]", "::ffff:127.0.0.1"])
def test_ipv6_loopback_listener_requires_mtls_until_tcp_drivers_support_ipv6(listen_host):
    with pytest.raises(ConfigError, match="require connection_security='mtls'"):
        ConnectorManager(
            communicator=MagicMock(),
            secure=False,
            comm_configurator=_config(
                {
                    "host": listen_host,
                    "listen_host": listen_host,
                    "connection_security": "clear",
                }
            ),
        )


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
