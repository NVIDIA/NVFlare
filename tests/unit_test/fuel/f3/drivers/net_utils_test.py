# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from unittest.mock import patch

import pytest

from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import encode_url, get_ssl_context, get_tcp_urls, parse_url
from nvflare.fuel.f3.drivers.tcp_driver import TcpDriver


class TestNetUtils:
    @pytest.mark.parametrize(("ssl_server", "source_role"), [(True, "client"), (False, "server")])
    @patch("nvflare.fuel.f3.drivers.net_utils.ssl.create_default_context")
    def test_mtls_context_reuses_complete_participant_pair(self, mock_context, ssl_server, source_role):
        cert = f"{source_role}.crt"
        key = f"{source_role}.key"
        params = {
            DriverParams.SCHEME.value: "stcp",
            DriverParams.CA_CERT.value: "rootCA.pem",
            f"{source_role}_cert": cert,
            f"{source_role}_key": key,
        }

        get_ssl_context(params, ssl_server=ssl_server)

        mock_context.return_value.load_cert_chain.assert_called_once_with(certfile=cert, keyfile=key)

    def test_encode_url(self):

        params = {
            DriverParams.SCHEME.value: "tcp",
            DriverParams.HOST.value: "flare.test.com",
            DriverParams.PORT.value: 1234,
            "b": "test value",
            "a": 123,
            "r": False,
        }

        url = encode_url(params)
        assert url == "tcp://flare.test.com:1234?b=test+value&a=123&r=False"

    def test_parse_url(self):
        url = "grpc://test.com:8002?a=123&b=test"
        params = parse_url(url)
        assert params.get(DriverParams.URL) == url
        assert int(params.get(DriverParams.PORT)) == 8002
        assert params.get("a") == "123"
        assert params.get("b") == "test"

    @patch("nvflare.fuel.f3.drivers.net_utils.get_open_tcp_port", return_value=1234)
    def test_tcp_listener_uses_explicit_listening_host(self, _):
        resources = {
            DriverParams.HOST.value: "127.0.0.1",
            DriverParams.LISTEN_HOST.value: "127.0.0.1",
        }

        assert get_tcp_urls("tcp", resources) == ("tcp://127.0.0.1:1234", "tcp://127.0.0.1:1234")

    @patch("nvflare.fuel.f3.drivers.net_utils.get_open_tcp_port", return_value=1234)
    def test_tcp_listener_default_remains_wildcard(self, _):
        assert get_tcp_urls("tcp", {DriverParams.HOST.value: "server.example"}) == (
            "tcp://server.example:1234",
            "tcp://0:1234",
        )

    @patch("nvflare.fuel.f3.drivers.net_utils.get_open_tcp_port", return_value=1234)
    def test_mtls_tcp_listener_advertises_stcp_end_to_end(self, _):
        resources = {
            DriverParams.HOST.value: "site-1",
            DriverParams.CONNECTION_SECURITY.value: "mtls",
        }

        assert TcpDriver.get_urls("tcp", resources) == ("stcp://site-1:1234", "stcp://0:1234")
