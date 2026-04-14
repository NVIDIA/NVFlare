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

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.hci.security import IdentityKey
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule, SessionManager


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


class _FakeSession:
    def __init__(self):
        self.created_args = None

    def make_token(self, id_asserter):
        return "session-token"


class _FakeSessionManager:
    def __init__(self):
        self.created = []

    def create_session(self, **kwargs):
        self.created.append(kwargs)
        return _FakeSession()


class _FakeCell:
    @staticmethod
    def fire_and_forget(*args, **kwargs):
        return None


class _FakeStudyRegistry:
    def __init__(self, roles=None, studies=None):
        self.roles = roles or {}
        self.studies = studies or {}

    def has_study(self, study):
        return study in self.studies

    def get_role(self, user_name, study):
        return self.roles.get((user_name, study))

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

        assert conn.strings == [("REJECT", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


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
        roles={("other-admin@nvidia.com", "cancer-research"): "lead"},
    )

    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)
    conn = _make_conn(study="cancer-research")

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [("REJECT", None)]
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
        roles={("admin@nvidia.com", "cancer-research"): "lead"},
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

        assert conn.strings == [("REJECT", None)]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()
