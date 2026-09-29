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

import datetime

import pytest

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.identity import CellIdentityResolver
from nvflare.fuel.f3.message import Message
from nvflare.fuel.hci.security import IdentityKey
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule, SessionManager
from nvflare.fuel.utils.admin_name_utils import new_admin_client_name
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, sign_content
from nvflare.private.fed.utils.identity_utils import IdentityVerifier


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
    def verify_common_name(asserter_cert, asserted_cn, signature, nonce, intermediate_certs=None):
        return True


class _FakeCert:
    not_valid_after = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)


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


class _CertlessHciServer:
    @staticmethod
    def get_id_verifier():
        return None

    @staticmethod
    def get_id_asserter():
        return None


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


def _patch_cert_login(monkeypatch, entitlements=()):
    monkeypatch.setattr("nvflare.fuel.hci.server.login.load_crt_chain_bytes", lambda _data: [_FakeCert()])
    monkeypatch.setattr("nvflare.fuel.hci.server.login.validate_admin_leaf_cert", lambda _cert: None)
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.get_admin_study_entitlements",
        lambda _cert: entitlements,
    )


def test_handle_cert_login_rejects_missing_server_identity():
    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)
    conn = _make_conn()
    conn.set_prop(ConnProps.HCI_SERVER, _CertlessHciServer())

    try:
        login.handle_cert_login(conn, ["CERT_LOGIN", "admin@nvidia.com"])

        assert conn.strings == [
            ("REJECT: AUTH_SERVER_IDENTITY_UNAVAILABLE: server signing identity is not configured", None)
        ]
        assert conn.tokens == []
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_handle_logout_ends_preverified_session_without_decoding_token():
    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    login = LoginModule(session_mgr)
    session = session_mgr.create_session("admin@nvidia.com", "nvidia", "project_admin", "admin-client")
    conn = _FakeConnection(props={ConnProps.SESSION: session})

    try:
        login.handle_logout(conn, ["logout"])

        assert session_mgr.get_sessions() == []
        assert conn.strings == [("OK", None)]
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_rejects_unknown_study_when_registry_exists(monkeypatch):
    _patch_cert_login(monkeypatch, ("trial-study",))
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


def test_handle_cert_login_rejects_unmapped_user_when_registry_exists(monkeypatch):
    _patch_cert_login(monkeypatch)
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
    _patch_cert_login(monkeypatch)
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
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_defaults_to_default_study_without_registry(monkeypatch):
    _patch_cert_login(monkeypatch, ("study-a",))
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
        session = list(session_mgr.sessions.values())[0]
        assert session.active_study == DEFAULT_STUDY
        assert session.cert_studies == ("study-a",)
    finally:
        session_mgr.shutdown()


def test_handle_cert_login_rejects_non_default_study_without_registry(monkeypatch):
    _patch_cert_login(monkeypatch, ("study-a",))
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


@pytest.mark.parametrize(
    "study,registry_allowed,cert_study,expected_reply",
    [
        ("study-a", False, "study-a", "OK"),
        ("study-a", True, "study-b", "OK"),
        ("study-a", True, "study-a", "OK"),
        (
            "study-a",
            False,
            "study-b",
            "REJECT: AUTH_STUDY_USER_NOT_MAPPED: user 'admin@nvidia.com' is not mapped to study 'study-a'",
        ),
        (DEFAULT_STUDY, False, "study-a", "OK"),
        (DEFAULT_STUDY, False, "study-a\n", "REJECT"),
        (DEFAULT_STUDY, False, "default\n", "REJECT"),
        ("study-a", False, "study-a\n", "REJECT"),
        ("study-a", True, "study-a\n", "REJECT"),
        ("study-a", False, "default\n", "REJECT"),
        ("study-a", True, "default\n", "REJECT"),
    ],
)
def test_handle_cert_login_with_real_study_certificate(
    monkeypatch, tmp_path, study, registry_allowed, cert_study, expected_reply
):
    root_key, root_pub = generate_keys()
    root = Identity("root", "nvidia")
    root_cert = generate_cert(root, root, root_key, root_pub, ca=True)
    root_path = tmp_path / "root.crt"
    root_path.write_bytes(serialize_cert(root_cert))
    admin_key, admin_pub = generate_keys()
    user = "admin@nvidia.com"
    cert = generate_cert(
        Identity(user, "nvidia", "lead"),
        root,
        root_key,
        admin_pub,
        uri_names=[f"https://nvidia.com/nvflare/v1/project/demo-project/study/{cert_study}"],
    )
    origin = new_admin_client_name()
    if expected_reply == "REJECT":
        with pytest.raises(ValueError, match="malformed study URI"):
            CellIdentityResolver().require_match(origin, user, "admin", peer_cert=cert)
    else:
        CellIdentityResolver().require_match(origin, user, "admin", peer_cert=cert)
    conn = _make_conn(study=study)
    conn.get_prop(ConnProps.CMD_HEADERS).update(
        cert=serialize_cert(cert), signature=sign_content(user, admin_key, return_str=False)
    )
    verifier = IdentityVerifier(str(root_path))
    monkeypatch.setattr(conn.get_prop(ConnProps.HCI_SERVER), "get_id_verifier", lambda: verifier)
    monkeypatch.setattr("nvflare.fuel.hci.server.login.StudyRegistryService", _FakeStudyRegistryService, raising=False)
    users = {(user, "study-a"): True} if registry_allowed else {}
    _FakeStudyRegistryService.registry = (
        None if study == DEFAULT_STUDY else _FakeStudyRegistry(studies={"study-a": {"site-a"}}, users=users)
    )
    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)

    try:
        LoginModule(session_mgr).handle_cert_login(conn, ["CERT_LOGIN", user])
        assert conn.strings == [(expected_reply, None)]
        if expected_reply == "OK":
            assert len(conn.tokens) == 1
            session = next(iter(session_mgr.sessions.values()))
            assert session.active_study == study
            assert session.cert_studies == (cert_study,)
            assert session.user_role == "lead"
        else:
            assert conn.tokens == []
            assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()
