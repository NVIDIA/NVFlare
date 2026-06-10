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

from unittest.mock import Mock, patch

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.hci.client.api import AdminAPI
from nvflare.fuel.hci.client.api_spec import AdminConfigKey


def test_admin_api_passes_server_identity_to_cell_auth_identity_map():
    cell_kwargs = {}
    authenticator_kwargs = {}

    class _FakeCell:
        def __init__(self, **kwargs):
            cell_kwargs.update(kwargs)
            self.core_cell = Mock()

        def register_request_cb(self, **kwargs):
            pass

        def start(self):
            pass

    class _FakeTokenVerifier:
        pass

    class _FakeAuthenticator:
        def __init__(self, **kwargs):
            authenticator_kwargs.update(kwargs)

        def authenticate(self, shared_fl_ctx, abort_signal):
            return "token", "signature", "ssid", _FakeTokenVerifier()

    admin_config = {
        AdminConfigKey.CONNECTION_SECURITY: ConnectionSecurity.MTLS,
        AdminConfigKey.PROJECT_NAME: "project",
        AdminConfigKey.SERVER_IDENTITY: "gcp-server",
        AdminConfigKey.CONNECTION_SCHEME: "tcp",
        AdminConfigKey.CA_CERT: "rootCA.pem",
        AdminConfigKey.CLIENT_CERT: "client.crt",
        AdminConfigKey.CLIENT_KEY: "client.key",
        AdminConfigKey.HOST: "35.192.226.99",
        AdminConfigKey.PORT: 8003,
    }
    api = AdminAPI(user_name="admin@nvidia.com", admin_config=admin_config, cmd_modules=[])

    with (
        patch("nvflare.fuel.hci.client.api.Authenticator", _FakeAuthenticator),
        patch("nvflare.fuel.hci.client.api.AuxRunner", Mock()),
        patch("nvflare.fuel.hci.client.api.Cell", _FakeCell),
        patch("nvflare.fuel.hci.client.api.NetAgent", Mock()),
        patch("nvflare.fuel.hci.client.api.ObjectStreamer", Mock()),
        patch("nvflare.fuel.hci.client.api.TokenVerifier", _FakeTokenVerifier),
        patch("nvflare.fuel.hci.client.api.set_add_auth_headers_filters"),
    ):
        api.connect()

    assert cell_kwargs["auth_identity_map"] == {FQCN.ROOT_SERVER: "gcp-server"}
    assert authenticator_kwargs["expected_sp_identity"] == "gcp-server"
