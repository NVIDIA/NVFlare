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

from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.hci.client.api import AdminAPI, ResultKey
from nvflare.fuel.hci.client.api_spec import AdminConfigKey, CommandContext, get_admin_oidc_config
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.credentials import (
    OIDC_ADMIN_PLACEHOLDER_NAME,
    CertAdminCredentials,
    OidcAdminCredentials,
    make_admin_credentials,
)
from nvflare.fuel.hci.proto import redact_headers


def test_do_client_command_preserves_command_props():
    api = AdminAPI.__new__(AdminAPI)
    captured = {}

    def _new_command_context(command, args, ent):
        ctx = CommandContext()
        ctx.set_command(command)
        ctx.set_command_args(args)
        ctx.set_command_entry(ent)
        return ctx

    def _handler(args, ctx):
        captured["props"] = ctx.get_command_props()
        ctx.set_command_result({"status": "ok"})

    api._new_command_context = _new_command_context
    ent = SimpleNamespace(handler=_handler)

    result = api._do_client_command("submit_job hello", ["submit_job", "hello"], ent, props={"study": "study-a"})

    assert result == {"status": "ok"}
    assert captured["props"] == {"study": "study-a"}


def test_user_login_sends_study_header(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api.credentials = CertAdminCredentials()
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.user_name = "admin@nvidia.com"
    api.study = "cancer-research"
    api.login_result = None
    captured = {}

    class _FakeIdentityAsserter:
        cert_data = "cert-data"

        def __init__(self, private_key_file, cert_file):
            assert private_key_file == "client.key"
            assert cert_file == "client.crt"

        @staticmethod
        def sign_common_name(nonce=""):
            return "signature"

    monkeypatch.setattr("nvflare.fuel.hci.client.credentials.IdentityAsserter", _FakeIdentityAsserter)

    def _fake_server_execute(command, reply_processor, headers=None):
        captured["command"] = command
        captured["headers"] = headers
        api.login_result = "OK"

    api.server_execute = _fake_server_execute
    api._after_login = lambda: {"status": "ok"}

    result = api._user_login()

    assert result == {"status": "ok"}
    assert captured["command"] == "_cert_login admin@nvidia.com"
    assert captured["headers"]["study"] == "cancer-research"


def test_user_login_defaults_study_header(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api.credentials = CertAdminCredentials()
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.user_name = "admin@nvidia.com"
    api.study = DEFAULT_STUDY
    api.login_result = None
    captured = {}

    class _FakeIdentityAsserter:
        cert_data = "cert-data"

        def __init__(self, private_key_file, cert_file):
            pass

        @staticmethod
        def sign_common_name(nonce=""):
            return "signature"

    monkeypatch.setattr("nvflare.fuel.hci.client.credentials.IdentityAsserter", _FakeIdentityAsserter)

    def _fake_server_execute(command, reply_processor, headers=None):
        captured["headers"] = headers
        api.login_result = "OK"

    api.server_execute = _fake_server_execute
    api._after_login = lambda: {"status": "ok"}

    api._user_login()

    assert captured["headers"]["study"] == DEFAULT_STUDY


def test_user_login_parses_structured_reject_code(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api.credentials = CertAdminCredentials()
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.user_name = "admin@nvidia.com"
    api.study = "cancer-research"
    api.login_result = None

    class _FakeIdentityAsserter:
        cert_data = "cert-data"

        def __init__(self, private_key_file, cert_file):
            pass

        @staticmethod
        def sign_common_name(nonce=""):
            return "signature"

    monkeypatch.setattr("nvflare.fuel.hci.client.credentials.IdentityAsserter", _FakeIdentityAsserter)

    def _fake_server_execute(command, reply_processor, headers=None):
        api.login_result = (
            "REJECT: AUTH_STUDY_USER_NOT_MAPPED: user 'admin@nvidia.com' is not mapped to study 'cancer-research'"
        )

    api.server_execute = _fake_server_execute

    result = api._user_login()

    assert result["status"] == APIStatus.ERROR_AUTHENTICATION
    assert result["details"] == "user 'admin@nvidia.com' is not mapped to study 'cancer-research'"
    assert result[ResultKey.AUTH_CODE] == "AUTH_STUDY_USER_NOT_MAPPED"


def test_oidc_user_login_sends_id_token_and_study_header():
    api = AdminAPI.__new__(AdminAPI)
    api.credentials = OidcAdminCredentials()
    api.auth_type = "oidc"
    api.oidc_config = {"id_token": "id-token"}
    api.study = "cancer-research"
    api.login_result = None
    captured = {}

    def _fake_server_execute(command, reply_processor, headers=None):
        captured["command"] = command
        captured["headers"] = headers
        api.login_result = "OK"

    api.server_execute = _fake_server_execute
    api._after_login = lambda: {"status": "ok"}

    result = api._user_login()

    assert result == {"status": "ok"}
    assert captured["command"] == "_oidc_login"
    assert captured["headers"] == {"id_token": "id-token", "study": "cancer-research"}


def test_oidc_user_login_reads_id_token_from_env(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api.credentials = OidcAdminCredentials()
    api.auth_type = "oidc"
    api.oidc_config = {"id_token_env": "NVFLARE_TEST_OIDC_TOKEN"}
    api.study = DEFAULT_STUDY
    api.login_result = None
    captured = {}
    monkeypatch.setenv("NVFLARE_TEST_OIDC_TOKEN", "env-id-token")

    def _fake_server_execute(command, reply_processor, headers=None):
        captured["headers"] = headers
        api.login_result = "OK"

    api.server_execute = _fake_server_execute
    api._after_login = lambda: {"status": "ok"}

    api._user_login()

    assert captured["headers"]["id_token"] == "env-id-token"


def test_debug_header_redaction_hides_oidc_tokens():
    redacted = redact_headers(
        {
            "id_token": "id-token",
            "access_token": "access-token",
            "study": "cancer-research",
        }
    )

    assert redacted == {
        "id_token": "********",
        "access_token": "********",
        "study": "cancer-research",
    }


def test_oidc_admin_config_does_not_require_client_private_key():
    api = AdminAPI(
        user_name="admin@nvidia.com",
        admin_config={
            AdminConfigKey.CA_CERT: "rootCA.pem",
            AdminConfigKey.CONNECTION_SECURITY: ConnectionSecurity.TLS,
            AdminConfigKey.AUTH_TYPE: "oidc",
            AdminConfigKey.OIDC: {"id_token_env": "NVFLARE_TEST_OIDC_TOKEN"},
        },
        cmd_modules=[],
        auto_login_max_tries=0,
    )

    assert api.auth_type == "oidc"
    assert api.client_cert is None
    assert api.client_key is None
    assert api.oidc_config == {"id_token_env": "NVFLARE_TEST_OIDC_TOKEN"}


def test_oidc_admin_config_without_username_uses_placeholder():
    api = AdminAPI(
        user_name="",
        admin_config={
            AdminConfigKey.CA_CERT: "rootCA.pem",
            AdminConfigKey.CONNECTION_SECURITY: ConnectionSecurity.TLS,
            AdminConfigKey.AUTH_TYPE: "oidc",
            AdminConfigKey.OIDC: {"issuer": "https://keycloak.example.com/realms/nvflare", "client_id": "nvflare"},
        },
        cmd_modules=[],
        auto_login_max_tries=0,
    )

    assert api.user_name == OIDC_ADMIN_PLACEHOLDER_NAME


def test_oidc_config_ignored_when_admin_auth_type_is_cert():
    oidc_config = get_admin_oidc_config(
        {
            AdminConfigKey.AUTH_TYPE: "cert",
            AdminConfigKey.OIDC: {"issuer": "https://keycloak.example.com/realms/nvflare"},
        }
    )

    assert oidc_config == {}


def test_oidc_admin_config_rejects_clear_transport_without_client_private_key():
    with pytest.raises(ConfigError, match="requires TLS or mTLS"):
        AdminAPI(
            user_name="admin@nvidia.com",
            admin_config={
                AdminConfigKey.CA_CERT: "rootCA.pem",
                AdminConfigKey.CONNECTION_SECURITY: ConnectionSecurity.CLEAR,
                AdminConfigKey.AUTH_TYPE: "oidc",
                AdminConfigKey.OIDC: {"id_token_env": "NVFLARE_TEST_OIDC_TOKEN"},
            },
            cmd_modules=[],
            auto_login_max_tries=0,
        )


def test_oidc_connect_verifies_server_without_asserting_client_identity(monkeypatch):
    captured = {}

    class _FakeTokenVerifier:
        pass

    class _FakeCoreCell:
        @staticmethod
        def add_incoming_filter(**kwargs):
            captured["incoming_filter"] = kwargs

    class _FakeCell:
        def __init__(self, **kwargs):
            captured["cell"] = kwargs
            self.core_cell = _FakeCoreCell()

        @staticmethod
        def register_request_cb(**kwargs):
            captured["request_cb"] = kwargs

        @staticmethod
        def start():
            captured["cell_started"] = True

    class _FakeNetAgent:
        def __init__(self, cell):
            captured["net_agent_cell"] = cell

    class _FakeAuthenticator:
        def __init__(self, **kwargs):
            captured["authenticator"] = kwargs

        @staticmethod
        def authenticate(shared_fl_ctx, abort_signal):
            return "token", "token-signature", "ssid", _FakeTokenVerifier()

    monkeypatch.setattr("nvflare.fuel.hci.client.api.Cell", _FakeCell)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.NetAgent", _FakeNetAgent)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.Authenticator", _FakeAuthenticator)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.TokenVerifier", _FakeTokenVerifier)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.flare_decomposers.register", lambda: None)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.set_add_auth_headers_filters", lambda *args, **kwargs: None)

    api = AdminAPI(
        user_name="admin@nvidia.com",
        admin_config={
            AdminConfigKey.CA_CERT: "rootCA.pem",
            AdminConfigKey.CONNECTION_SECURITY: ConnectionSecurity.TLS,
            AdminConfigKey.AUTH_TYPE: "oidc",
            AdminConfigKey.OIDC: {"id_token_env": "NVFLARE_TEST_OIDC_TOKEN"},
        },
        cmd_modules=[],
        auto_login_max_tries=0,
    )

    api.connect()

    assert captured["cell"]["credentials"][DriverParams.CONNECTION_SECURITY.value] == ConnectionSecurity.TLS
    assert captured["authenticator"]["secure_mode"] is True
    assert captured["authenticator"]["assert_client_identity"] is False
    assert captured["authenticator"]["private_key_file"] is None
    assert captured["authenticator"]["cert_file"] is None


def test_oidc_connect_with_client_certs_does_not_assert_client_identity(monkeypatch):
    # OIDC over mTLS: client certs are used for transport only; asserting app-level identity
    # would register under the placeholder name, which can never match the cert CN.
    captured = {}

    class _FakeTokenVerifier:
        pass

    class _FakeCoreCell:
        @staticmethod
        def add_incoming_filter(**kwargs):
            captured["incoming_filter"] = kwargs

    class _FakeCell:
        def __init__(self, **kwargs):
            captured["cell"] = kwargs
            self.core_cell = _FakeCoreCell()

        @staticmethod
        def register_request_cb(**kwargs):
            captured["request_cb"] = kwargs

        @staticmethod
        def start():
            captured["cell_started"] = True

    class _FakeNetAgent:
        def __init__(self, cell):
            captured["net_agent_cell"] = cell

    class _FakeAuthenticator:
        def __init__(self, **kwargs):
            captured["authenticator"] = kwargs

        @staticmethod
        def authenticate(shared_fl_ctx, abort_signal):
            return "token", "token-signature", "ssid", _FakeTokenVerifier()

    monkeypatch.setattr("nvflare.fuel.hci.client.api.Cell", _FakeCell)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.NetAgent", _FakeNetAgent)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.Authenticator", _FakeAuthenticator)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.TokenVerifier", _FakeTokenVerifier)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.flare_decomposers.register", lambda: None)
    monkeypatch.setattr("nvflare.fuel.hci.client.api.set_add_auth_headers_filters", lambda *args, **kwargs: None)

    api = AdminAPI(
        user_name="",
        admin_config={
            AdminConfigKey.CA_CERT: "rootCA.pem",
            AdminConfigKey.CLIENT_CERT: "client.crt",
            AdminConfigKey.CLIENT_KEY: "client.key",
            AdminConfigKey.CONNECTION_SECURITY: ConnectionSecurity.MTLS,
            AdminConfigKey.AUTH_TYPE: "oidc",
            AdminConfigKey.OIDC: {"id_token_env": "NVFLARE_TEST_OIDC_TOKEN"},
        },
        cmd_modules=[],
        auto_login_max_tries=0,
    )

    api.connect()

    # certs are still supplied for the mTLS transport
    credentials = captured["cell"]["credentials"]
    assert credentials[DriverParams.CLIENT_CERT.value] == "client.crt"
    assert credentials[DriverParams.CLIENT_KEY.value] == "client.key"
    # but app-level client identity is not asserted, while server identity is still verified
    assert captured["authenticator"]["assert_client_identity"] is False
    assert captured["authenticator"]["secure_mode"] is True


def _make_oidc_api(auto_login_max_tries=1):
    api = AdminAPI.__new__(AdminAPI)
    api.credentials = OidcAdminCredentials()
    api.auth_type = "oidc"
    api.oidc_config = {"issuer": "https://keycloak.example.com/realms/nvflare", "client_id": "nvflare"}
    api.study = DEFAULT_STUDY
    api.user_name = "admin@nvidia.com"
    api.event_handlers = []
    api._debug = False
    api.login_result = None
    api._oidc_login_token = None
    api.auto_login_max_tries = auto_login_max_tries
    return api


def test_oidc_login_fetches_id_token_once_across_retries(monkeypatch):
    # the (possibly interactive) token source must be invoked once per login(), not per retry
    api = _make_oidc_api(auto_login_max_tries=3)
    fetch_count = {"n": 0}
    sent_tokens = []

    def _fake_get_id_token(api_):
        fetch_count["n"] += 1
        return f"id-token-{fetch_count['n']}"

    def _fake_server_execute(command, reply_processor, headers=None):
        sent_tokens.append(headers["id_token"])
        api.login_result = None  # communication error -> retry

    api.credentials.get_id_token = _fake_get_id_token
    api.server_execute = _fake_server_execute
    monkeypatch.setattr("nvflare.fuel.hci.client.api.AUTO_LOGIN_INTERVAL", 0)

    result = api._try_login()

    assert result["status"] == APIStatus.ERROR_RUNTIME
    assert fetch_count["n"] == 1
    assert sent_tokens == ["id-token-1", "id-token-1", "id-token-1"]
    assert api._oidc_login_token is None  # not kept beyond the login attempt


def test_oidc_login_auth_reject_stops_retries_and_clears_cached_token(monkeypatch):
    api = _make_oidc_api(auto_login_max_tries=5)
    cleared = []
    attempts = {"n": 0}

    def _fake_server_execute(command, reply_processor, headers=None):
        attempts["n"] += 1
        api.login_result = "REJECT: AUTH_OIDC_INVALID_TOKEN: token rejected"

    api.credentials.get_id_token = lambda api_: "stale-id-token"
    api.server_execute = _fake_server_execute
    monkeypatch.setattr("nvflare.fuel.sec.oidc.clear_cached_id_token", lambda config: cleared.append(config))
    monkeypatch.setattr("nvflare.fuel.hci.client.api.AUTO_LOGIN_INTERVAL", 0)

    result = api._try_login()

    assert result["status"] == APIStatus.ERROR_AUTHENTICATION
    assert attempts["n"] == 1  # authentication REJECT stops the retry loop
    assert cleared == [api.oidc_config]  # the rejected token is purged from the cache


def test_logout_preserves_cached_oidc_token(monkeypatch):
    # Every one-shot CLI command closes its session (Session.close -> api.logout), so a
    # logout that cleared the cache would force a fresh browser SSO on each command.
    api = _make_oidc_api()
    api.in_logout = False
    cleared = []

    api.server_execute = lambda command: {"status": "ok"}
    api.close = lambda: None
    monkeypatch.setattr("nvflare.fuel.sec.oidc.clear_cached_id_token", lambda config: cleared.append(config))

    api.logout()

    assert cleared == []


def test_logout_does_not_clear_oidc_cache_for_cert_auth(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api.credentials = CertAdminCredentials()
    api.auth_type = "cert"
    api.in_logout = False
    cleared = []

    api.server_execute = lambda command: {"status": "ok"}
    api.close = lambda: None
    monkeypatch.setattr("nvflare.fuel.sec.oidc.clear_cached_id_token", lambda config: cleared.append(config))

    api.logout()

    assert cleared == []


def test_server_invalid_command_log_redacts_sensitive_headers():
    redacted = redact_headers(
        {
            "id_token": "id-token",
            "access_token": "access-token",
            "__token__": "cell-token",
            "__token_signature__": "cell-token-signature",
            "Authorization": "Bearer abc",
            "study": "cancer-research",
            "cellnet.origin": "admin_abc",
        }
    )

    assert redacted == {
        "id_token": "********",
        "access_token": "********",
        "__token__": "********",
        "__token_signature__": "********",
        "Authorization": "********",
        "study": "cancer-research",
        "cellnet.origin": "admin_abc",
    }
    assert redact_headers(None) == {}


def test_make_admin_credentials_rejects_unknown_auth_type():
    with pytest.raises(ConfigError, match="unsupported admin authentication type 'jwt'"):
        make_admin_credentials("jwt")


def test_make_admin_credentials_defaults_to_cert_when_unset():
    assert isinstance(make_admin_credentials(""), CertAdminCredentials)
    assert isinstance(make_admin_credentials(None), CertAdminCredentials)
    assert isinstance(make_admin_credentials(" Cert "), CertAdminCredentials)
    assert isinstance(make_admin_credentials("oidc"), OidcAdminCredentials)
