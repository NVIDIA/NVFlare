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

import pytest

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.api import AdminAPI
from nvflare.fuel.hci.client.api_spec import AdminConfigKey
from nvflare.fuel.hci.proto import InternalCommands
from nvflare.fuel.sec.ephemeral_admin_cert import EphemeralAdminCertFiles


def test_admin_api_hydrates_missing_cert_pair_from_step_ca(monkeypatch, tmp_path):
    key_file = tmp_path / "client.key"
    cert_file = tmp_path / "client.crt"
    key_file.write_text("key", encoding="utf-8")
    cert_file.write_text("cert", encoding="utf-8")
    resolved_files = EphemeralAdminCertFiles(client_key=str(key_file), client_cert=str(cert_file))
    captured = {}

    def _fake_obtain(config, root_ca_file):
        captured["config"] = config
        captured["root_ca_file"] = root_ca_file
        return resolved_files

    monkeypatch.setattr("nvflare.fuel.hci.client.api.obtain_ephemeral_admin_cert_files", _fake_obtain)

    api = AdminAPI(
        user_name="alice@nvidia.com",
        admin_config={
            AdminConfigKey.PROJECT_NAME: "project",
            AdminConfigKey.CA_CERT: "rootCA.pem",
            AdminConfigKey.EPHEMERAL_ADMIN_CERT: {
                "provider": "step_ca",
                "provider_config": {
                    "ca_url": "https://step-ca.example.com",
                    "provisioner": "nvflare-admin-oidc",
                },
            },
        },
        cmd_modules=[],
    )

    assert api.client_key == str(key_file)
    assert api.client_cert == str(cert_file)
    assert api.ephemeral_admin_cert_files is resolved_files
    assert "subject" not in captured["config"]
    assert captured["root_ca_file"] == "rootCA.pem"


def test_admin_api_reports_invalid_ephemeral_renewal_window_as_config_error():
    with pytest.raises(ConfigError, match="renewal_window must be a number"):
        AdminAPI(
            user_name="alice@nvidia.com",
            admin_config={
                AdminConfigKey.PROJECT_NAME: "project",
                AdminConfigKey.CA_CERT: "rootCA.pem",
                AdminConfigKey.EPHEMERAL_ADMIN_CERT: {
                    "provider": "step_ca",
                    "renewal_window": "one minute",
                    "provider_config": {},
                },
            },
            cmd_modules=[],
        )


def test_admin_api_renews_expiring_ephemeral_cert_and_resets_connection(monkeypatch, tmp_path):
    old_key_file = tmp_path / "old.key"
    old_cert_file = tmp_path / "old.crt"
    new_key_file = tmp_path / "new.key"
    new_cert_file = tmp_path / "new.crt"
    for path in (old_key_file, old_cert_file, new_key_file, new_cert_file):
        path.write_text(path.name, encoding="utf-8")

    class _FakeCell:
        stopped = False

        def stop(self):
            self.stopped = True

    old_files = EphemeralAdminCertFiles(
        client_key=str(old_key_file),
        client_cert=str(old_cert_file),
        expires_at=1.0,
    )
    new_files = EphemeralAdminCertFiles(
        client_key=str(new_key_file),
        client_cert=str(new_cert_file),
        expires_at=9999999999.0,
    )
    issued_files = [old_files, new_files]

    def _fake_obtain(config, root_ca_file):
        return issued_files.pop(0)

    monkeypatch.setattr("nvflare.fuel.hci.client.api.obtain_ephemeral_admin_cert_files", _fake_obtain)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.load_cert_file", lambda path: path)
    monkeypatch.setattr(
        "nvflare.fuel.hci.client.api.get_cn_from_cert",
        lambda path: "alice@nvidia.com" if path == str(old_cert_file) else "bob@nvidia.com",
    )

    api = AdminAPI(
        user_name="alice@nvidia.com",
        admin_config={
            AdminConfigKey.PROJECT_NAME: "project",
            AdminConfigKey.CA_CERT: "rootCA.pem",
            AdminConfigKey.UID_SOURCE: "cert",
            AdminConfigKey.EPHEMERAL_ADMIN_CERT: {
                "provider": "step_ca",
                "renewal_window": 60.0,
                "provider_config": {
                    "ca_url": "https://step-ca.example.com",
                    "provisioner": "nvflare-admin-oidc",
                },
            },
        },
        cmd_modules=[],
    )

    assert api.client_key == str(old_key_file)
    assert api.user_name == "alice@nvidia.com"
    assert api.fl_ctx_mgr.identity_name == "alice@nvidia.com"
    old_cell = _FakeCell()
    api.cell = old_cell
    api.aux_runner = object()
    api.server_sess_active = True
    api.token = "old-token"
    api.login_result = "old-result"

    assert api.ensure_client_cert_valid()

    assert api.client_key == str(new_key_file)
    assert api.client_cert == str(new_cert_file)
    assert api.user_name == "bob@nvidia.com"
    assert api.fl_ctx_mgr.identity_name == "bob@nvidia.com"
    assert old_cell.stopped
    assert api.cell is None
    assert api.aux_runner is None
    assert not api.server_sess_active
    assert api.token is None
    assert api.login_result is None


def test_user_login_builds_command_after_identity_renewal(monkeypatch):
    class _FakeIdentityAsserter:
        cert_data = b"cert"

        def __init__(self, private_key_file, cert_file):
            pass

        def sign_common_name(self, nonce):
            return b"signature"

    api = object.__new__(AdminAPI)
    api.user_name = "alice@nvidia.com"
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.study = "default"
    api.cell = object()
    api.login_result = None
    captured = {}

    def _renew():
        api.user_name = "bob@nvidia.com"
        api.cell = None
        return True

    def _connect():
        api.cell = object()

    def _server_execute(command, reply_processor, headers):
        captured["command"] = command
        captured["headers"] = headers
        api.login_result = "REJECT"

    api.ensure_client_cert_valid = _renew
    api.connect = _connect
    api.server_execute = _server_execute
    monkeypatch.setattr("nvflare.fuel.hci.client.api.IdentityAsserter", _FakeIdentityAsserter)

    api._user_login()

    assert captured["command"] == f"{InternalCommands.CERT_LOGIN} bob@nvidia.com"
    assert captured["headers"]["user_name"] == "bob@nvidia.com"


def test_user_login_reconnects_when_cell_is_missing_without_another_renewal(monkeypatch):
    class _FakeIdentityAsserter:
        cert_data = b"cert"

        def __init__(self, private_key_file, cert_file):
            pass

        def sign_common_name(self, nonce):
            return b"signature"

    api = object.__new__(AdminAPI)
    api.user_name = "alice@nvidia.com"
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.study = "default"
    api.cell = None
    api.login_result = None
    connected = []

    api.ensure_client_cert_valid = lambda: False

    def _connect():
        connected.append(True)
        api.cell = object()

    def _server_execute(command, reply_processor, headers):
        api.login_result = "REJECT"

    api.connect = _connect
    api.server_execute = _server_execute
    monkeypatch.setattr("nvflare.fuel.hci.client.api.IdentityAsserter", _FakeIdentityAsserter)

    api._user_login()

    assert connected == [True]


def test_connect_cleans_up_partial_cell_after_authentication_failure(monkeypatch):
    class _FakeCell:
        stopped = False

        def __init__(self, **kwargs):
            pass

        def register_request_cb(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    class _FailingAuthenticator:
        def __init__(self, **kwargs):
            pass

        def authenticate(self, **kwargs):
            raise RuntimeError("authentication failed")

    monkeypatch.setattr("nvflare.fuel.hci.client.api.Cell", _FakeCell)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.NetAgent", lambda cell: None)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.Authenticator", _FailingAuthenticator)

    api = AdminAPI(
        user_name="alice@nvidia.com",
        admin_config={
            AdminConfigKey.PROJECT_NAME: "project",
            AdminConfigKey.CA_CERT: "rootCA.pem",
            AdminConfigKey.CLIENT_CERT: "client.crt",
            AdminConfigKey.CLIENT_KEY: "client.key",
        },
        cmd_modules=[],
    )

    with pytest.raises(RuntimeError, match="authentication failed"):
        api.connect()

    assert api.cell is None
    assert api.aux_runner is None
    assert api.object_streamer is None
