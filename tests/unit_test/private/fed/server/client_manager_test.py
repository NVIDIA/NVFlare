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
import time
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.client import Client, ClientPropKey
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegSession, ClientType, InternalFLContextKey
from nvflare.private.fed.server.client_manager import ADMIN_CLIENT_FQCN_TTL, ClientManager, _is_oidc_admin_auth_enabled


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


def _make_certless_request(client_name: str) -> MagicMock:
    request = _make_request(client_name)
    request.payload.pop(IdentityChallengeKey.CERT, None)
    request.payload.pop(IdentityChallengeKey.SIGNATURE, None)
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


def test_secure_admin_without_cert_is_rejected_when_oidc_admin_auth_is_disabled():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    request = _make_certless_request("admin-client-1")
    fl_ctx = _make_fl_ctx(secure_mode=True, client_name="admin-client-1")

    with patch("nvflare.private.fed.server.client_manager._is_oidc_admin_auth_enabled", return_value=False):
        client = manager.authenticated_client(request, fl_ctx, ClientType.ADMIN)

    assert client is None


def test_secure_admin_without_cert_is_allowed_only_for_oidc_admin_auth():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    request = _make_certless_request("admin-client-1")
    fl_ctx = _make_fl_ctx(secure_mode=True, client_name="admin-client-1")

    with (
        patch("nvflare.private.fed.server.client_manager._is_oidc_admin_auth_enabled", return_value=True),
        patch.object(manager, "_set_client_props"),
    ):
        client = manager.authenticated_client(request, fl_ctx, ClientType.ADMIN)

    assert client is not None
    assert client.get_prop(ClientPropKey.ORG, "") == ""


@pytest.mark.parametrize(
    "startup_conf,expected",
    [
        ({"auth": {"admin": {"type": "oidc"}}}, True),
        ({"auth": {"type": "oidc"}}, True),
        ({"auth": {"admin": {"type": "cert"}}}, False),
        ({}, False),
        (None, False),
        # Malformed configs fail closed: the OIDC exemption stays disabled so certs remain required.
        ({"auth": "oidc"}, False),
        ({"auth": {"admin": "oidc"}}, False),
        ({"auth": {"admin": {"type": "bogus"}}}, False),
    ],
)
def test_is_oidc_admin_auth_enabled(startup_conf, expected):
    with patch(
        "nvflare.private.fed.server.client_manager.ConfigService.get_section",
        return_value=startup_conf,
    ):
        assert _is_oidc_admin_auth_enabled() is expected


def test_secure_admin_without_cert_is_rejected_when_auth_config_is_malformed():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    request = _make_certless_request("admin-client-1")
    fl_ctx = _make_fl_ctx(secure_mode=True, client_name="admin-client-1")

    with patch(
        "nvflare.private.fed.server.client_manager.ConfigService.get_section",
        return_value={"auth": {"admin": "oidc"}},
    ):
        client = manager.authenticated_client(request, fl_ctx, ClientType.ADMIN)

    assert client is None


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


def test_disabled_clients_file_load_failure_fails_closed(tmp_path):
    disabled_file = tmp_path / "disabled_clients.json"
    disabled_file.write_text("{broken-json", encoding="utf-8")
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)

    with pytest.raises(json.JSONDecodeError):
        manager.set_disabled_clients_file(str(disabled_file))


def test_disabled_clients_file_rejects_bare_list_schema(tmp_path):
    disabled_file = tmp_path / "disabled_clients.json"
    disabled_file.write_text(json.dumps(["site-a"]), encoding="utf-8")
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)

    with pytest.raises(ValueError, match="JSON object"):
        manager.set_disabled_clients_file(str(disabled_file))


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


def test_disable_enable_persist_while_holding_client_manager_lock():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    lock_states = []

    def fake_save(_snapshot):
        lock_states.append(manager.lock.locked())

    manager._save_disabled_clients = fake_save

    manager.disable_client("site-a")
    manager.enable_client("site-a")

    assert lock_states == [True, True]


def test_save_disabled_clients_removes_tmp_on_replace_failure(tmp_path):
    disabled_file = tmp_path / "disabled_clients.json"
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    manager.set_disabled_clients_file(str(disabled_file))

    with patch("nvflare.private.fed.server.client_manager.os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError):
            manager._save_disabled_clients({"site-a"})

    assert list(tmp_path.glob("disabled_clients.json.*.tmp")) == []


def test_remove_client_unknown_token_returns_none():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)

    assert manager.remove_client("unknown-token") is None


def test_enable_client_persists_and_allows_client(tmp_path):
    disabled_file = tmp_path / "disabled_clients.json"
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    manager.set_disabled_clients_file(str(disabled_file))
    manager.disable_client("site-a")

    assert manager.enable_client("site-a") is True

    assert not manager.is_client_disabled("site-a")
    assert json.loads(disabled_file.read_text()) == {"disabled_clients": []}


def test_disabled_clients_file_can_be_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    manager.set_disabled_clients_file("disabled_clients.json")

    manager.disable_client("site-a")

    assert json.loads((tmp_path / "disabled_clients.json").read_text()) == {"disabled_clients": ["site-a"]}


def test_disabled_client_save_runs_under_manager_lock():
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

    assert calls == [False, False]


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


def _make_authenticate_request(client_name: str, client_type: str, origin: str) -> MagicMock:
    request = _make_request(client_name)
    headers = {
        CellMessageHeaderKeys.CLIENT_NAME: client_name,
        CellMessageHeaderKeys.CLIENT_TYPE: client_type,
        CellMessageHeaderKeys.PROJECT_NAME: "project",
        CellMessageHeaderKeys.CLIENT_IP: "1.2.3.4",
        MessageHeaderKey.ORIGIN: origin,
    }
    request.get_header.side_effect = lambda key: headers.get(key)
    return request


def test_authenticate_records_admin_origin_binding():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    request = _make_authenticate_request("admin@nvidia.com", ClientType.ADMIN, "admin_abc123")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="admin@nvidia.com")

    client = manager.authenticate(request, fl_ctx)

    assert client is not None
    # admin clients are still kept out of the regular clients map
    assert client.token not in manager.clients
    record = manager.admin_clients[client.token]
    assert record.name == "admin@nvidia.com"
    assert record.fqcn == "admin_abc123"
    assert manager.resolve_admin_client_fqcn("admin@nvidia.com", client.token) == "admin_abc123"


def test_authenticate_regular_client_does_not_touch_admin_map():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    request = _make_authenticate_request("site-a", ClientType.REGULAR, "site-a")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="site-a")

    client = manager.authenticate(request, fl_ctx)

    assert client is not None
    assert manager.clients[client.token] is client
    assert manager.admin_clients == {}
    assert manager.resolve_admin_client_fqcn("site-a", client.token) is None


def test_resolve_admin_client_fqcn_requires_matching_name_and_token():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    request = _make_authenticate_request("admin@nvidia.com", ClientType.ADMIN, "admin_abc123")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="admin@nvidia.com")
    client = manager.authenticate(request, fl_ctx)

    # a known token with a mismatched claimed name fails CLOSED (empty binding never matches)
    assert manager.resolve_admin_client_fqcn("someone-else", client.token) == ""
    # an unknown token is simply not an admin registration
    assert manager.resolve_admin_client_fqcn("admin@nvidia.com", "unknown-token") is None


def test_resolve_admin_client_fqcn_refreshes_last_used():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    request = _make_authenticate_request("admin@nvidia.com", ClientType.ADMIN, "admin_abc123")
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="admin@nvidia.com")
    client = manager.authenticate(request, fl_ctx)

    record = manager.admin_clients[client.token]
    record.last_used -= ADMIN_CLIENT_FQCN_TTL - 10  # old, but not yet stale

    assert manager.resolve_admin_client_fqcn("admin@nvidia.com", client.token) == "admin_abc123"
    assert record.last_used == pytest.approx(time.time(), abs=5)


def test_stale_admin_origin_binding_is_rejected_on_resolve():
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="admin@nvidia.com")
    request = _make_authenticate_request("admin@nvidia.com", ClientType.ADMIN, "admin_old")
    old_client = manager.authenticate(request, fl_ctx)
    record = manager.admin_clients[old_client.token]
    record.last_used -= ADMIN_CLIENT_FQCN_TTL + 1
    stale_time = record.last_used

    # an expired binding fails CLOSED: the empty binding can never match a message
    # origin, so the origin check rejects instead of being skipped (None would skip it)
    assert manager.resolve_admin_client_fqcn("admin@nvidia.com", old_client.token) == ""
    assert old_client.token in manager.admin_clients
    # a rejected resolve must not refresh last_used (that would resurrect the binding)
    assert record.last_used == stale_time


def test_stale_admin_origin_bindings_are_kept_fail_closed():
    # Stale bindings are deliberately NOT pruned: dropping the record would make the
    # token "unknown" and skip origin validation entirely (fail open).
    manager = ClientManager(project_name="project", min_num_clients=1, max_num_clients=10)
    fl_ctx = _make_fl_ctx(secure_mode=False, client_name="admin@nvidia.com")

    request = _make_authenticate_request("admin@nvidia.com", ClientType.ADMIN, "admin_stale")
    stale_client = manager.authenticate(request, fl_ctx)
    manager.admin_clients[stale_client.token].last_used -= ADMIN_CLIENT_FQCN_TTL + 1
    request = _make_authenticate_request("admin@nvidia.com", ClientType.ADMIN, "admin_new")
    new_client = manager.authenticate(request, fl_ctx)

    assert set(manager.admin_clients) == {stale_client.token, new_client.token}
    assert manager.resolve_admin_client_fqcn("admin@nvidia.com", stale_client.token) == ""
    assert manager.resolve_admin_client_fqcn("admin@nvidia.com", new_client.token) == "admin_new"


def test_set_client_props_sets_site_config():
    site_config = {"format_version": 1, "labels": {"region": "us-east"}}
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.CLIENT_SITE_CONFIG, site_config, private=True, sticky=False)

    client = Client(name="site-1", token="token")
    ClientManager._set_client_props(client, "server.site-1", fl_ctx)

    assert client.get_site_config() == site_config
