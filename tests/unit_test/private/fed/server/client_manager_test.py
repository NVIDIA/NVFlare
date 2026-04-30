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

import json
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.client import Client, ClientPropKey
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


def test_disable_client_persists_and_removes_active_client(tmp_path):
    disabled_file = tmp_path / "disabled_clients.json"
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    manager.set_disabled_clients_file(str(disabled_file))
    client = Client("site-a", "token-a")
    manager.clients[client.token] = client
    manager.name_to_clients[client.name] = client

    removed_tokens = manager.disable_client("site-a")

    assert removed_tokens == ["token-a"]
    assert "token-a" not in manager.clients
    assert "site-a" not in manager.name_to_clients
    assert manager.is_client_disabled("site-a")
    assert json.loads(disabled_file.read_text()) == {"disabled_clients": ["site-a"]}

    reloaded = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    reloaded.set_disabled_clients_file(str(disabled_file))
    assert reloaded.is_client_disabled("site-a")


def test_disable_client_restores_active_client_when_persist_fails():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    client = Client("site-a", "token-a")
    manager.clients[client.token] = client
    manager.name_to_clients[client.name] = client
    manager._save_disabled_clients = MagicMock(side_effect=OSError("disk full"))

    with pytest.raises(OSError):
        manager.disable_client("site-a")

    assert not manager.is_client_disabled("site-a")
    assert manager.clients["token-a"] is client
    assert manager.name_to_clients["site-a"] is client


def test_enable_client_persists_and_allows_client(tmp_path):
    disabled_file = tmp_path / "disabled_clients.json"
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    manager.set_disabled_clients_file(str(disabled_file))
    manager.disable_client("site-a")

    assert manager.enable_client("site-a") is True

    assert not manager.is_client_disabled("site-a")
    assert json.loads(disabled_file.read_text()) == {"disabled_clients": []}


def test_disabled_client_save_runs_outside_manager_lock():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    calls = []

    def assert_unlocked_save(_disabled_clients):
        acquired = manager.lock.acquire(blocking=False)
        if acquired:
            manager.lock.release()
        calls.append(acquired)

    manager._save_disabled_clients = assert_unlocked_save

    manager.disable_client("site-a")
    manager.enable_client("site-a")

    assert calls == [True, True]


def test_disabled_client_registration_is_rejected():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    manager.disable_client("site-a")
    request = _make_request("site-a")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="site-a")

    client = manager.authenticated_client(request, fl_ctx, ClientType.REGULAR)

    assert client is None
    fl_ctx.set_prop.assert_called_with(FLContextKey.UNAUTHENTICATED, "Client 'site-a' is disabled", sticky=False)


def test_disabled_client_heartbeat_does_not_reactivate():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    manager.disable_client("site-a")
    fl_ctx = MagicMock()

    reactivated = manager.heartbeat("token-a", "site-a", "site-a@server", fl_ctx)

    assert reactivated is False
    assert "token-a" not in manager.clients
    fl_ctx.set_prop.assert_called_with(FLContextKey.UNAUTHENTICATED, "Client 'site-a' is disabled", sticky=False)
