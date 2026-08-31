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

import pytest

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.hci.base64_utils import b64str_to_str, str_to_b64str
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ProtoKey
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.sess import Session, SessionManager
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key
from nvflare.private.fed.utils.identity_utils import IdentityAsserter, TokenVerifier


class _FakeIdAsserter:
    cert = "server-cert"

    @staticmethod
    def sign(data, return_str=True):
        assert return_str
        return "signature"


class _FakeTokenVerifier:
    def __init__(self, _cert):
        pass

    @staticmethod
    def verify(_nonce, _data, signature):
        return signature == "signature"


class _FakeCell:
    pass


class _FakeHciServer:
    @staticmethod
    def get_id_asserter():
        return _FakeIdAsserter()


@pytest.fixture(autouse=True)
def _patch_token_verifier(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.sess.TokenVerifier", _FakeTokenVerifier)


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
    restored = Session.decode_token(token, _FakeIdAsserter())

    assert restored.active_study == "cancer-research"
    assert restored.user_name == "admin@nvidia.com"
    assert restored.user_org == "nvidia"
    assert restored.user_role == "lead"


def test_session_token_round_trip_with_real_signature(monkeypatch, tmp_path):
    private_key, public_key = generate_keys()
    identity = Identity("server")
    cert = generate_cert(identity, identity, private_key, public_key, ca=True)
    key_path = tmp_path / "server.key"
    cert_path = tmp_path / "server.crt"
    key_path.write_bytes(serialize_pri_key(private_key))
    cert_path.write_bytes(serialize_cert(cert))
    id_asserter = IdentityAsserter(str(key_path), str(cert_path))
    monkeypatch.setattr("nvflare.fuel.hci.server.sess.TokenVerifier", TokenVerifier)
    session = Session("session-id", "admin@nvidia.com", "nvidia", "lead", "origin")

    restored = Session.decode_token(session.make_token(id_asserter), id_asserter)

    assert restored.sess_id == session.sess_id
    assert restored.user_name == session.user_name


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


def test_session_token_round_trip_preserves_cert_expiry():
    cert_exp = time.time() + 60
    session = Session(
        sess_id="session-id",
        user_name="admin@nvidia.com",
        org="nvidia",
        role="lead",
        origin_fqcn="origin",
        active_study="cancer-research",
        cert_exp=cert_exp,
    )

    restored = Session.decode_token(session.make_token(_FakeIdAsserter()), _FakeIdAsserter())

    assert restored.cert_exp == cert_exp
    assert not restored.is_cert_expired(now=cert_exp - 1)
    assert restored.is_cert_expired(now=cert_exp)


def test_decode_token_defaults_legacy_session_study():
    legacy_payload = json.dumps({"n": "admin@nvidia.com", "r": "lead", "o": "nvidia", "s": "session-id"})
    token = f"{str_to_b64str(legacy_payload)}:signature"

    restored = Session.decode_token(token, _FakeIdAsserter())

    assert restored.active_study == DEFAULT_STUDY


def test_decode_token_accepts_legacy_t_study_field():
    legacy_payload = json.dumps(
        {"n": "admin@nvidia.com", "r": "lead", "o": "nvidia", "s": "session-id", "t": "legacy-study"}
    )
    token = f"{str_to_b64str(legacy_payload)}:signature"

    restored = Session.decode_token(token, _FakeIdAsserter())

    assert restored.active_study == "legacy-study"


def test_decode_token_rejects_missing_identity_asserter():
    payload = json.dumps({"n": "attacker", "r": "project_admin", "o": "attacker-org", "s": "forged-id"})
    token = f"{str_to_b64str(payload)}:attacker-signature"

    with pytest.raises(ValueError, match="identity asserter"):
        Session.decode_token(token, None)


def test_decode_token_rejects_invalid_signature():
    payload = json.dumps({"n": "attacker", "r": "project_admin", "o": "attacker-org", "s": "forged-id"})
    token = f"{str_to_b64str(payload)}:attacker-signature"

    assert Session.decode_token(token, _FakeIdAsserter()) is None


def test_recreate_session_rejects_token_without_identity_asserter():
    payload = json.dumps({"n": "attacker", "r": "project_admin", "o": "attacker-org", "s": "forged-id"})
    token = f"{str_to_b64str(payload)}:attacker-signature"
    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)

    try:
        with pytest.raises(ValueError, match="identity asserter"):
            session_mgr.recreate_session(token, "attacker", None)
        assert session_mgr.sessions == {}
    finally:
        session_mgr.shutdown()


def test_registered_check_session_command_accepts_signed_live_token():
    session_mgr = SessionManager(_FakeCell(), idle_timeout=3600, monitor_interval=3600)
    session = session_mgr.create_session("admin@nvidia.com", "nvidia", "lead", "origin")
    token = session.make_token(_FakeIdAsserter())
    conn = Connection(props={ConnProps.HCI_SERVER: _FakeHciServer()})
    conn.request = {ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: token}]}

    try:
        check_session = next(
            command.handler_func
            for command in session_mgr.get_spec().cmd_specs
            if command.name == InternalCommands.CHECK_SESSION
        )
        check_session(conn, [InternalCommands.CHECK_SESSION])

        assert conn.buffer.data == [{ProtoKey.TYPE: ProtoKey.STRING, ProtoKey.DATA: "OK"}]
    finally:
        session_mgr.shutdown()
