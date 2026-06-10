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

from nvflare.apis.fl_constant import SystemConfigs
from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.hci.proto import InternalCommands
from nvflare.fuel.hci.security import IdentityKey
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule, SessionManager
from nvflare.fuel.sec.authn import AUTH_ERROR_CODE_UNAVAILABLE, AuthError
from nvflare.fuel.sec.principal import AUTH_METHOD_OIDC, Principal
from nvflare.fuel.utils.config_service import ConfigService


class _FakeConnection:
    def __init__(self, props=None):
        self._props = dict(props or {})
        self.strings = []
        self.tokens = []

    def get_prop(self, key, default=None):
        return self._props.get(key, default)

    def set_prop(self, key, value):
        self._props[key] = value

    def append_string(self, msg, meta=None):
        self.strings.append((msg, meta))

    def append_token(self, token):
        self.tokens.append(token)


class _FakeVerifier:
    @staticmethod
    def verify_common_name(asserter_cert, asserted_cn, signature, nonce):
        return True


class _FakeIdAsserter:
    cert = "asserter-cert"

    @staticmethod
    def sign(data, return_str=True):
        assert return_str
        return "signature"


class _FakeHciServer:
    @staticmethod
    def get_id_verifier():
        return _FakeVerifier()

    @staticmethod
    def get_id_asserter():
        return _FakeIdAsserter()


class _FakeCell:
    @staticmethod
    def fire_and_forget(*args, **kwargs):
        return None


class _FakeStudyRegistry:
    def __init__(self, users=None, studies=None):
        self.users = users or {}
        self.studies = studies or {}

    def has_study(self, study):
        return study in self.studies

    def has_user(self, user_name, study):
        return (user_name, study) in self.users

    def get_sites(self, study):
        return self.studies.get(study)


class _FakeStudyRegistryService:
    registry = None

    @staticmethod
    def get_registry():
        return _FakeStudyRegistryService.registry


def _make_conn(study=None):
    headers = {"cert": "cert-bytes", "signature": "signature"}
    if study is not None:
        headers["study"] = study
    return _FakeConnection(
        props={
            ConnProps.CMD_HEADERS: headers,
            ConnProps.HCI_SERVER: _FakeHciServer(),
            ConnProps.REQUEST: Message(headers={MessageHeaderKey.ORIGIN: "admin-client"}),
        }
    )


def _make_oidc_conn(study=None, id_token="id-token"):
    headers = {}
    if id_token:
        headers["id_token"] = id_token
    if study is not None:
        headers["study"] = study
    return _FakeConnection(
        props={
            ConnProps.CMD_HEADERS: headers,
            ConnProps.HCI_SERVER: _FakeHciServer(),
            ConnProps.REQUEST: Message(headers={MessageHeaderKey.ORIGIN: "admin-client"}),
        }
    )


class _FakeOidcProvider:
    def __init__(self):
        self.credentials = None

    def authenticate(self, credentials):
        self.credentials = credentials
        return Principal(
            subject="keycloak-subject",
            username="admin@nvidia.com",
            email="admin@nvidia.com",
            org="nvidia",
            raw_roles=["flare_project_admin"],
            groups=["/flare/project-admins"],
            issuer="https://keycloak.example.com/realms/nvflare",
            token_id="token-id",
            effective_role="project_admin",
            auth_method=AUTH_METHOD_OIDC,
        )


class _RejectingOidcProvider:
    def __init__(self, error=None):
        self.error = error or AuthError("bad token")

    def authenticate(self, credentials):
        raise self.error


def test_handle_cert_login_rejects_unknown_study_when_registry_exists(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.load_crt_bytes", lambda _data: object())
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.cert_to_dict",
        lambda _cert: {"subject": {"commonName": "admin@nvidia.com"}},
    )
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.get_identity_info",
        lambda _cert_dict: {IdentityKey.ORG: "nvidia", IdentityKey.ROLE: "project_admin"},
    )
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = _FakeStudyRegistry(studies={"cancer-research": {"site-a"}})

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)
    conn = _make_conn(study="trial-study")

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [("REJECT: AUTH_UNKNOWN_STUDY: unknown study 'trial-study'", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_rejects_when_server_auth_type_is_oidc(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, admin_auth_config={"type": "oidc"})
    conn = _make_conn(study=None)

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [("REJECT: AUTH_CERT_DISABLED: certificate admin login is disabled", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_strips_oidc_auth_type(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, admin_auth_config={"type": " oidc "})
    conn = _make_conn(study=None)

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [("REJECT: AUTH_CERT_DISABLED: certificate admin login is disabled", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_rejects_invalid_auth_type(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, admin_auth_config={"type": "oidc-disabled"})
    conn = _make_conn(study=None)

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [
            ("REJECT: AUTH_CONFIG_INVALID: unsupported admin authentication type 'oidc-disabled'", None)
        ]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_oidc_admin_config_can_be_read_from_startup_config_service(tmp_path):
    ConfigService.reset()
    ConfigService.initialize(section_files={}, config_path=[str(tmp_path)], parsed_args=None, var_dict={})
    ConfigService.add_section(
        SystemConfigs.STARTUP_CONF,
        {
            "auth": {
                "admin": {
                    "type": "oidc",
                    "oidc": {
                        "issuer": "https://keycloak.example.com/realms/nvflare",
                        "client_id": "nvflare-admin",
                        "jwks_url": "https://keycloak.example.com/realms/nvflare/protocol/openid-connect/certs",
                    },
                }
            }
        },
    )

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)

    try:
        oidc_config = login._get_oidc_admin_config()

        assert oidc_config["issuer"] == "https://keycloak.example.com/realms/nvflare"
        assert oidc_config["client_id"] == "nvflare-admin"
    finally:
        session_mgr.shutdown()
        ConfigService.reset()


def test_handle_cert_login_rejects_unmapped_user_when_registry_exists(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.load_crt_bytes", lambda _data: object())
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.cert_to_dict",
        lambda _cert: {"subject": {"commonName": "admin@nvidia.com"}},
    )
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.get_identity_info",
        lambda _cert_dict: {IdentityKey.ORG: "nvidia", IdentityKey.ROLE: "project_admin"},
    )
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = _FakeStudyRegistry(
        studies={"cancer-research": {"site-a"}},
        users={("other-admin@nvidia.com", "cancer-research"): True},
    )

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)
    conn = _make_conn(study="cancer-research")

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [
            (
                "REJECT: AUTH_STUDY_USER_NOT_MAPPED: user 'admin@nvidia.com' is not mapped to study 'cancer-research'",
                None,
            )
        ]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_accepts_mapped_user_for_valid_study(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.load_crt_bytes", lambda _data: object())
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.cert_to_dict",
        lambda _cert: {"subject": {"commonName": "admin@nvidia.com"}},
    )
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.get_identity_info",
        lambda _cert_dict: {IdentityKey.ORG: "nvidia", IdentityKey.ROLE: "project_admin"},
    )
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = _FakeStudyRegistry(
        studies={"cancer-research": {"site-a"}},
        users={("admin@nvidia.com", "cancer-research"): True},
    )

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)
    conn = _make_conn(study="cancer-research")

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [("OK", None)]
        assert len(conn.tokens) == 1
        session = list(session_mgr.sessions.values())[0]
        assert session.active_study == "cancer-research"
        assert session.user_name == "admin@nvidia.com"
        assert session.user_role == "project_admin"
        assert session.principal.policy_name() == "admin@nvidia.com"
        assert session.principal.policy_org() == "nvidia"
        assert session.principal.policy_role() == "project_admin"
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_defaults_to_default_study_without_registry(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.load_crt_bytes", lambda _data: object())
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.cert_to_dict",
        lambda _cert: {"subject": {"commonName": "admin@nvidia.com"}},
    )
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.get_identity_info",
        lambda _cert_dict: {IdentityKey.ORG: "nvidia", IdentityKey.ROLE: "project_admin"},
    )
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)
    conn = _make_conn(study=None)

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [("OK", None)]
        assert len(conn.tokens) == 1
        assert list(session_mgr.sessions.values())[0].active_study == DEFAULT_STUDY
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_rejects_non_default_study_without_registry(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.load_crt_bytes", lambda _data: object())
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.cert_to_dict",
        lambda _cert: {"subject": {"commonName": "admin@nvidia.com"}},
    )
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.get_identity_info",
        lambda _cert_dict: {IdentityKey.ORG: "nvidia", IdentityKey.ROLE: "project_admin"},
    )
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)
    conn = _make_conn(study="study-a")

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [
            ("REJECT: AUTH_STUDY_NOT_CONFIGURED: study 'study-a' is not configured on the server", None)
        ]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_oidc_login_rejects_unenrolled_user_for_study(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = _FakeStudyRegistry(studies={"cancer-research": {"site-a"}})

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, oidc_auth_provider=_FakeOidcProvider())
    conn = _make_oidc_conn(study="cancer-research", id_token="id-token")

    try:
        login.handle_oidc_login(conn, [InternalCommands.OIDC_LOGIN])

        assert conn.strings == [
            (
                "REJECT: AUTH_STUDY_USER_NOT_MAPPED: user 'admin@nvidia.com' is not mapped to study 'cancer-research'",
                None,
            )
        ]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_oidc_login_accepts_enrolled_user_for_study(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = _FakeStudyRegistry(
        studies={"cancer-research": {"site-a"}},
        users={("admin@nvidia.com", "cancer-research"): True},
    )

    oidc_provider = _FakeOidcProvider()
    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, oidc_auth_provider=oidc_provider)
    conn = _make_oidc_conn(study="cancer-research", id_token="id-token")

    try:
        login.handle_oidc_login(conn, [InternalCommands.OIDC_LOGIN])

        assert conn.strings == [("OK", None)]
        assert len(conn.tokens) == 1
        assert oidc_provider.credentials == "id-token"
        session = list(session_mgr.sessions.values())[0]
        assert session.active_study == "cancer-research"
        assert session.user_name == "admin@nvidia.com"
        assert session.user_role == "project_admin"
        assert session.principal.auth_method == AUTH_METHOD_OIDC
        assert session.principal.subject == "keycloak-subject"
    finally:
        session_mgr.shutdown()


def test_handle_oidc_login_rejects_unknown_study(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = _FakeStudyRegistry(studies={"cancer-research": {"site-a"}})

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, oidc_auth_provider=_FakeOidcProvider())
    conn = _make_oidc_conn(study="trial-study", id_token="id-token")

    try:
        login.handle_oidc_login(conn, [InternalCommands.OIDC_LOGIN])

        assert conn.strings == [("REJECT: AUTH_UNKNOWN_STUDY: unknown study 'trial-study'", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_oidc_login_rejects_missing_id_token(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, oidc_auth_provider=_FakeOidcProvider())
    conn = _make_oidc_conn(id_token="")

    try:
        login.handle_oidc_login(conn, [InternalCommands.OIDC_LOGIN])

        assert conn.strings == [("REJECT: AUTH_OIDC_MISSING_TOKEN: missing OIDC id_token", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_oidc_login_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, oidc_auth_provider=_RejectingOidcProvider())
    conn = _make_oidc_conn(id_token="bad-token")

    try:
        login.handle_oidc_login(conn, [InternalCommands.OIDC_LOGIN])

        assert conn.strings == [("REJECT: AUTH_OIDC_INVALID_TOKEN: OIDC token rejected", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_oidc_login_rejects_when_oidc_not_configured(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, admin_auth_config={})
    conn = _make_oidc_conn(id_token="id-token")

    try:
        login.handle_oidc_login(conn, [InternalCommands.OIDC_LOGIN])

        assert conn.strings == [("REJECT: AUTH_OIDC_NOT_CONFIGURED: OIDC admin authentication is not configured", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_fails_closed_for_non_mapping_admin_auth_config(tmp_path, monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    ConfigService.reset()
    ConfigService.initialize(section_files={}, config_path=[str(tmp_path)], parsed_args=None, var_dict={})
    ConfigService.add_section(SystemConfigs.STARTUP_CONF, {"auth": {"admin": "oidc"}})

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)
    conn = _make_conn(study=None)

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [
            ("REJECT: AUTH_CONFIG_INVALID: invalid server config: 'auth.admin' must be a mapping", None)
        ]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()
        ConfigService.reset()


def test_handle_oidc_login_uses_auth_error_code_over_message_text(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    _FakeStudyRegistryService.registry = None

    # message text says 'not configured' but the machine-readable code must win
    provider = _RejectingOidcProvider(AuthError("provider not configured properly", code=AUTH_ERROR_CODE_UNAVAILABLE))
    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr, oidc_auth_provider=provider)
    conn = _make_oidc_conn(id_token="id-token")

    try:
        login.handle_oidc_login(conn, [InternalCommands.OIDC_LOGIN])

        assert conn.strings == [("REJECT: AUTH_OIDC_UNAVAILABLE: provider not configured properly", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()
