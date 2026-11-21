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

from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ProtoKey
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule
from nvflare.fuel.hci.server.sess import SessionManager
from nvflare.fuel.sec.admin_audit import audit_context
from nvflare.security.study_registry import StudyRegistry, StudyRegistryService
from tests.admin_session_test_utils import register_admin_transport
from tests.timing_utils import ManualClock


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.sess.threading.Thread.start", lambda _: None)
    manager = SessionManager(Mock(), idle_timeout=10)
    try:
        yield manager
    finally:
        manager.shutdown()


def _command_conn(token, grant):
    conn = Connection(props={ConnProps.ADMIN_TRANSPORT_GRANT: grant})
    conn.request = {ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: token}]}
    return conn


@pytest.mark.parametrize("command", [InternalCommands.CHECK_SESSION, "list_jobs"])
def test_registered_session_requires_its_issued_token_and_transport(manager, command):
    _, grant = register_admin_transport(manager)
    _, other_grant = register_admin_transport(manager)
    session, token = manager.create_session(grant, "org", "lead")
    assert session

    def accepted(presented_token, presented_grant):
        conn = _command_conn(presented_token, presented_grant)
        if command != InternalCommands.CHECK_SESSION:
            with audit_context():
                return LoginModule(manager).pre_command(conn, [command])
        manager.handle_check_session(conn, [command])
        return conn.buffer.data[0][ProtoKey.DATA] == "OK"

    assert accepted(token, grant)
    assert not accepted(token, other_grant)
    assert not accepted(token, None)
    assert not accepted("unissued-token", grant)
    assert manager.get_sessions() == [session]


@pytest.mark.parametrize("termination", ["logout", "idle", "certificate_expired", "transport_replaced"])
def test_session_termination_revokes_both_credentials_without_resurrection(monkeypatch, manager, termination):
    clock = ManualClock(now=100.0)
    monkeypatch.setattr("nvflare.fuel.hci.server.sess.time", clock)
    request, grant = register_admin_transport(manager, cert_exp=105 if termination == "certificate_expired" else 200)
    session, token = manager.create_session(grant, "org", "lead")
    assert session
    cancel = Mock()
    assert manager.bind_download(session.sess_id, "download", cancel)
    assert manager.get_transport(request) is grant
    assert manager.get_session(token, grant) is session

    if termination == "logout":
        conn = _command_conn(token, grant)
        with audit_context():
            assert LoginModule(manager).pre_command(conn, [InternalCommands.LOGOUT])
            LoginModule(manager).handle_logout(conn, [InternalCommands.LOGOUT])
    elif termination == "transport_replaced":
        register_admin_transport(manager, origin=grant.origin, fingerprint=b"replacement-cert")
    else:
        clock.advance(5 if termination == "certificate_expired" else 10)

    assert manager.get_transport(request) is None
    assert manager.get_session(token, grant) is None
    assert manager.create_session(grant, "org", "lead") == (None, None)
    assert manager.get_sessions() == []
    cancel.assert_called_once_with()


@pytest.mark.parametrize("binding", ["certificate", "connection"])
def test_transport_credential_requires_its_original_peer_and_origin(manager, binding):
    request, grant = register_admin_transport(manager, connection=object() if binding == "connection" else None)
    assert manager.get_transport(request) is grant
    actual_endpoint = request.get_prop(MessagePropKey.ENDPOINT)
    request.set_header(MessageHeaderKey.ORIGIN, "other-origin")
    assert manager.get_transport(request) is None
    request.set_header(MessageHeaderKey.ORIGIN, actual_endpoint.name)
    key = MessagePropKey.CONNECTION if binding == "connection" else MessagePropKey.PEER_CERT_FINGERPRINT
    request.set_prop(key, object())
    assert manager.get_transport(request) is None


@pytest.mark.parametrize("cert_studies,initially_mapped", [((), True), (("trial",), True), (("trial",), False)])
def test_study_removal_revokes_session_even_if_membership_is_added_back(manager, cert_studies, initially_mapped):
    _, grant = register_admin_transport(manager)

    def registry(members):
        return StudyRegistry({"format_version": "1.0", "studies": {"trial": {"admins": members}}})

    original = StudyRegistryService.get_registry()
    try:
        StudyRegistryService.initialize(registry([grant.user_name] if initially_mapped else []))
        session, token = manager.create_session(grant, "org", "lead", "trial", cert_studies)
        assert session and session.cert_studies == cert_studies
        cancel = Mock()
        assert manager.bind_download(session.sess_id, "download", cancel)
        StudyRegistryService.initialize(registry([grant.user_name]))
        StudyRegistryService.initialize(registry([]))
        assert manager.get_session(token, grant) is None
        cancel.assert_not_called()
        manager.check_sessions()
        cancel.assert_called_once_with()
        StudyRegistryService.initialize(registry([grant.user_name]))
        assert manager.get_session(token, grant) is None
        assert manager.create_session(grant, "org", "lead", "trial", cert_studies) == (None, None)
    finally:
        StudyRegistryService.initialize(original)


def test_direct_goodbye_revokes_session_but_another_connection_cannot(manager):
    connection = object()
    request, grant = register_admin_transport(manager, connection=connection)
    session, token = manager.create_session(grant, "org", "lead")
    request.set_prop(MessagePropKey.CONNECTION, object())
    manager.close_transport(request)
    assert manager.get_session(token, grant) is session
    request.set_prop(MessagePropKey.CONNECTION, connection)
    manager.close_transport(request)
    assert manager.get_session(token, grant) is None
    assert manager.get_transport(request) is None


def test_retrying_verified_login_rotates_only_the_hci_token(manager):
    request, grant = register_admin_transport(manager)
    old_session, old_token = manager.create_session(grant, "org", "lead")
    cancel = Mock()
    assert manager.bind_download(old_session.sess_id, "download", cancel)

    new_session, new_token = manager.create_session(grant, "org", "lead")

    assert new_session is not old_session
    assert manager.get_session(old_token, grant) is None
    assert manager.get_session(new_token, grant) is new_session
    assert manager.get_transport(request, require_session=True) is grant
    assert manager.get_sessions() == [new_session]
    cancel.assert_called_once_with()
