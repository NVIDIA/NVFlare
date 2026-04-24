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

from nvflare.apis.client import ClientPropKey
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegSession, ClientType, InternalFLContextKey
from nvflare.private.fed.server.client_manager import ClientManager


def _make_request(client_name: str) -> MagicMock:
    shareable = Shareable()
    shareable[IdentityChallengeKey.CERT] = b"fake-cert"
    shareable[IdentityChallengeKey.SIGNATURE] = b"fake-signature"

    request = MagicMock()
    request.payload = shareable
    headers = {
        CellMessageHeaderKeys.CLIENT_NAME: client_name,
        MessageHeaderKey.ORIGIN: f"{client_name}@site",
    }
    request.get_header.side_effect = lambda key: headers.get(key)
    return request


def _make_fl_ctx(secure_mode: bool, client_name: str) -> MagicMock:
    reg = ClientRegSession(client_name)
    fl_ctx = MagicMock()

    def _get_prop(key, default=None):
        if key == FLContextKey.SECURE_MODE:
            return secure_mode
        if key == InternalFLContextKey.CLIENT_REG_SESSION:
            return reg
        return default

    fl_ctx.get_prop.side_effect = _get_prop
    return fl_ctx


def test_authenticated_client_stores_org_extracted_from_cert():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    request = _make_request("site-a")
    fl_ctx = _make_fl_ctx(secure_mode=True, client_name="site-a")
    verifier = MagicMock()

    with (
        patch.object(manager, "_get_id_verifier", return_value=verifier),
        patch("nvflare.private.fed.server.client_manager.load_crt_bytes", return_value=object()),
        patch("nvflare.private.fed.server.client_manager.get_org_from_cert", return_value="org_a"),
        patch.object(manager, "_set_client_props"),
    ):
        client = manager.authenticated_client(request, fl_ctx, ClientType.REGULAR)

    assert client is not None
    assert client.get_prop(ClientPropKey.ORG) == "org_a"
    verifier.verify_common_name.assert_called_once()


def test_authenticated_client_sets_empty_org_when_secure_mode_is_disabled():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    request = _make_request("site-a")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="site-a")

    with patch.object(manager, "_set_client_props"):
        client = manager.authenticated_client(request, fl_ctx, ClientType.REGULAR)

    assert client is not None
    assert client.get_prop(ClientPropKey.ORG, "") == ""
