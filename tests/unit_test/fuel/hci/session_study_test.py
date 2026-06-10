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
from unittest.mock import MagicMock

import pytest

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.hci.base64_utils import b64str_to_str, str_to_b64str
from nvflare.fuel.hci.server.sess import ENDED_SESSION_BLOCKLIST_MARGIN, Session, SessionManager
from nvflare.fuel.sec.principal import AUTH_METHOD_CERT, AUTH_METHOD_OIDC, Principal


def _cert_principal(user_name="admin@nvidia.com", org="nvidia", role="lead"):
    return Principal.from_legacy_admin(username=user_name, org=org, role=role, auth_method=AUTH_METHOD_CERT)


class _FakeIdAsserter:
    cert = "server-cert"

    @staticmethod
    def sign(data, return_str=True):
        assert return_str
        return "signature"


def test_session_token_round_trip_preserves_study():
    session = Session(
        sess_id="session-id",
        user_name="admin@nvidia.com",
        org="nvidia",
        role="lead",
        origin_fqcn="origin",
        active_study="cancer-research",
    )

    token = session.make_token(_FakeIdAsserter())
    restored = Session.decode_token(token)

    assert restored.active_study == "cancer-research"
    assert restored.user_name == "admin@nvidia.com"
    assert restored.user_org == "nvidia"
    assert restored.user_role == "lead"


def test_session_token_uses_study_field_name():
    session = Session(
        sess_id="session-id",
        user_name="admin@nvidia.com",
        org="nvidia",
        role="lead",
        origin_fqcn="origin",
        active_study="cancer-research",
    )

    token = session.make_token(_FakeIdAsserter())
    payload = json.loads(b64str_to_str(token.split(":")[0]))

    assert payload["study"] == "cancer-research"
    assert "t" not in payload


def test_session_token_includes_absolute_expiration():
    session_mgr = SessionManager(MagicMock(), idle_timeout=3600, monitor_interval=3600, max_session_lifetime=60)
    try:
        session = session_mgr.create_session_from_principal(_cert_principal("admin@nvidia.com"), origin_fqcn="origin")
        token = session.make_token(_FakeIdAsserter())
        payload = json.loads(b64str_to_str(token.split(":")[0]))

        assert payload["iat"] <= payload["exp"]
        assert payload["epoch"] == session_mgr.session_epoch
        assert payload["exp"] <= int(time.time()) + 60
    finally:
        session_mgr.shutdown()


def test_oidc_session_expiration_is_bounded_by_identity_provider_token():
    oidc_exp = time.time() + 30
    principal = Principal(
        subject="keycloak-subject",
        username="admin@nvidia.com",
        org="nvidia",
        issuer="https://keycloak.example.com/realms/nvflare",
        token_exp=oidc_exp,
        effective_role="project_admin",
        auth_method=AUTH_METHOD_OIDC,
    )
    session_mgr = SessionManager(MagicMock(), idle_timeout=3600, monitor_interval=3600, max_session_lifetime=3600)
    try:
        session = session_mgr.create_session_from_principal(principal, origin_fqcn="origin")
        token = session.make_token(_FakeIdAsserter())
        payload = json.loads(b64str_to_str(token.split(":")[0]))

        assert payload["exp"] == int(oidc_exp)
    finally:
        session_mgr.shutdown()


def test_decode_token_rejects_expired_session_token():
    session = Session(
        sess_id="session-id",
        user_name="admin@nvidia.com",
        org="nvidia",
        role="lead",
        origin_fqcn="origin",
        active_study="cancer-research",
        expires_at=time.time() - 1,
    )
    token = session.make_token(_FakeIdAsserter())

    with pytest.raises(ValueError, match="expired"):
        Session.decode_token(token)


def test_session_epoch_rejects_token_from_previous_server_process():
    first_mgr = SessionManager(MagicMock(), idle_timeout=3600, monitor_interval=3600)
    second_mgr = SessionManager(MagicMock(), idle_timeout=3600, monitor_interval=3600)
    try:
        session = first_mgr.create_session_from_principal(_cert_principal("admin@nvidia.com"), origin_fqcn="origin")
        token = session.make_token(_FakeIdAsserter())

        assert second_mgr.get_session(token) is None
        with pytest.raises(ValueError, match="previous server session"):
            second_mgr.recreate_session(token, origin_fqcn="origin", id_asserter=None)
    finally:
        first_mgr.shutdown()
        second_mgr.shutdown()


def test_ended_session_token_cannot_recreate_session():
    session_mgr = SessionManager(MagicMock(), idle_timeout=3600, monitor_interval=3600)
    try:
        session = session_mgr.create_session_from_principal(_cert_principal("admin@nvidia.com"), origin_fqcn="origin")
        token = session.make_token(_FakeIdAsserter())

        session_mgr.end_session_by_token(token)

        assert session_mgr.get_session(token) is None
        with pytest.raises(ValueError, match="session has ended"):
            session_mgr.recreate_session(token, origin_fqcn="origin", id_asserter=None)
    finally:
        session_mgr.shutdown()


def test_decode_token_defaults_legacy_session_study():
    legacy_payload = json.dumps({"n": "admin@nvidia.com", "r": "lead", "o": "nvidia", "s": "session-id"})
    token = f"{str_to_b64str(legacy_payload)}:signature"

    restored = Session.decode_token(token)

    assert restored.active_study == DEFAULT_STUDY


def test_decode_token_accepts_legacy_t_study_field():
    legacy_payload = json.dumps(
        {"n": "admin@nvidia.com", "r": "lead", "o": "nvidia", "s": "session-id", "t": "legacy-study"}
    )
    token = f"{str_to_b64str(legacy_payload)}:signature"

    restored = Session.decode_token(token)

    assert restored.active_study == "legacy-study"


def test_session_token_round_trip_preserves_oidc_principal_metadata():
    principal = Principal(
        subject="keycloak-subject",
        username="admin@nvidia.com",
        org="nvidia",
        raw_roles=["flare_project_admin"],
        groups=["/flare/project-admins"],
        issuer="https://keycloak.example.com/realms/nvflare",
        token_id="token-id",
        token_exp=time.time() + 300,
        effective_role="project_admin",
        auth_method=AUTH_METHOD_OIDC,
    )
    session = Session(
        sess_id="session-id",
        user_name=principal.policy_name(),
        org=principal.policy_org(),
        role=principal.policy_role(),
        origin_fqcn="origin",
        active_study="cancer-research",
        principal=principal,
    )

    token = session.make_token(_FakeIdAsserter())
    restored = Session.decode_token(token)

    assert restored.user_name == "admin@nvidia.com"
    assert restored.user_org == "nvidia"
    assert restored.user_role == "project_admin"
    assert restored.principal == principal


def test_cert_session_token_keeps_legacy_fields_without_principal_metadata():
    session_mgr = SessionManager(MagicMock(), idle_timeout=3600, monitor_interval=3600)
    try:
        session = session_mgr.create_session_from_principal(
            _cert_principal("admin@nvidia.com"),
            origin_fqcn="origin",
            active_study="cancer-research",
        )
        assert session.principal.auth_method == AUTH_METHOD_CERT

        token = session.make_token(_FakeIdAsserter())
        payload = json.loads(b64str_to_str(token.split(":")[0]))

        assert payload["n"] == "admin@nvidia.com"
        assert payload["o"] == "nvidia"
        assert payload["r"] == "lead"
        assert "p" not in payload

        restored = Session.decode_token(token)
        assert restored.user_name == "admin@nvidia.com"
        assert restored.user_org == "nvidia"
        assert restored.user_role == "lead"
        assert restored.active_study == "cancer-research"
    finally:
        session_mgr.shutdown()


def test_session_identity_is_derived_from_principal():
    principal = Principal(
        subject="keycloak-subject",
        username="admin@nvidia.com",
        org="nvidia",
        effective_role="project_admin",
        auth_method=AUTH_METHOD_OIDC,
        token_exp=time.time() + 300,
    )
    session = Session(sess_id="session-id", origin_fqcn="origin", principal=principal)

    assert session.user_name == principal.policy_name()
    assert session.user_org == principal.policy_org()
    assert session.user_role == principal.policy_role()


def test_ended_session_blocklist_entry_expires_with_token():
    session_mgr = SessionManager(MagicMock(), idle_timeout=3600, monitor_interval=3600, max_session_lifetime=3600)
    try:
        session = session_mgr.create_session_from_principal(_cert_principal("admin@nvidia.com"), origin_fqcn="origin")
        session_mgr.end_session_by_id(session.sess_id)

        assert session_mgr.ended_sessions[session.sess_id] == session.expires_at + ENDED_SESSION_BLOCKLIST_MARGIN
    finally:
        session_mgr.shutdown()


def test_monitor_prunes_expired_blocklist_entries():
    session_mgr = SessionManager(MagicMock(), idle_timeout=3600, monitor_interval=0.05, max_session_lifetime=3600)
    try:
        session = session_mgr.create_session_from_principal(_cert_principal("admin@nvidia.com"), origin_fqcn="origin")
        session_mgr.end_session_by_id(session.sess_id)
        with session_mgr.sess_update_lock:
            session_mgr.ended_sessions[session.sess_id] = time.time() - 1

        deadline = time.time() + 5
        while time.time() < deadline:
            with session_mgr.sess_update_lock:
                if session.sess_id not in session_mgr.ended_sessions:
                    break
            time.sleep(0.05)
        assert session.sess_id not in session_mgr.ended_sessions
    finally:
        session_mgr.shutdown()


def test_monitor_ends_all_dead_sessions_with_distinct_reasons():
    cell = MagicMock()
    session_mgr = SessionManager(cell, idle_timeout=600, monitor_interval=0.05, max_session_lifetime=3600)
    try:
        expired = session_mgr.create_session_from_principal(
            _cert_principal("expired@nvidia.com"), origin_fqcn="origin-expired"
        )
        idle = session_mgr.create_session_from_principal(_cert_principal("idle@nvidia.com"), origin_fqcn="origin-idle")
        expired.expires_at = time.time() - 1
        idle.last_active_time = time.time() - 601

        deadline = time.time() + 5
        while time.time() < deadline:
            with session_mgr.sess_update_lock:
                if not session_mgr.sessions:
                    break
            time.sleep(0.05)

        assert session_mgr.sessions == {}
        assert expired.sess_id in session_mgr.ended_sessions
        assert idle.sess_id in session_mgr.ended_sessions

        reasons_by_target = {
            call.kwargs["targets"]: call.kwargs["message"].payload for call in cell.fire_and_forget.call_args_list
        }
        assert "maximum lifetime" in reasons_by_target["origin-expired"]
        assert "inactivity" in reasons_by_target["origin-idle"]
    finally:
        session_mgr.shutdown()


def test_decode_token_rejects_malformed_principal_metadata():
    payload = json.dumps(
        {
            "n": "admin@nvidia.com",
            "r": "project_admin",
            "o": "nvidia",
            "s": "session-id",
            "p": "not-a-principal-dict",
        }
    )
    token = f"{str_to_b64str(payload)}:signature"

    with pytest.raises(ValueError, match="invalid principal data"):
        Session.decode_token(token)
