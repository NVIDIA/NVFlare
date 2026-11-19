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
from types import SimpleNamespace

import pytest

from nvflare.fuel.f3.message import Message
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands
from nvflare.fuel.hci.reg import CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.audit import CommandAudit
from nvflare.fuel.hci.server.authz import AuthzFilter, PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule
from nvflare.fuel.hci.server.reg import CommandFilter, ServerCommandRegister
from nvflare.fuel.hci.server.sess import Session, SessionManager
from nvflare.fuel.sec.admin_audit import AdminContext, audit_context, snapshot, update_context
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.fuel.sec.authz import AuthorizationService


@pytest.fixture
def audit_log(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    auditor = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", auditor)
    yield auditor, lambda: [json.loads(line) for line in path.read_text().splitlines()]
    auditor.close()


class _VerifiedIdentity(CommandFilter):
    def pre_command(self, conn, args):
        update_context(user="admin@example.com", identity_source="locally_verified")
        conn.set_prop(ConnProps.USER_NAME, "admin@example.com")
        return True


def _register(auditor, name="delete_job", handler=None, authz=None, filters=()):
    reg = ServerCommandRegister(None)
    reg.register_module_spec(
        CommandModuleSpec("test", [CommandSpec(name, "test", "test", handler or (lambda *_: None), authz)])
    )
    reg.finalize()
    reg.add_filter(_VerifiedIdentity())
    reg.add_filter(AuthzFilter())
    for f in filters:
        reg.add_filter(f)
    reg.add_filter(CommandAudit(auditor))
    return reg


def test_sensitive_handling_references_request_without_arguments(audit_log):
    auditor, records = audit_log
    conn = Connection()
    _register(auditor).process_command(conn, "test.delete_job never-record-this-argument")
    entries = records()
    assert [r["action"] for r in entries] == [
        "delete_job",
        "admin.command.dispatched",
        "admin.command.handler_returned",
    ]
    request = entries[0]["event_id"]
    assert conn.get_prop(ConnProps.EVENT_ID) == request
    assert all(r["reference"] == request for r in entries[1:])
    assert all(r["user"] == "admin@example.com" for r in entries)
    assert "never-record-this-argument" not in json.dumps(entries)
    assert snapshot() == AdminContext()


@pytest.mark.parametrize("name,expected_count", [("check_status", 1), (InternalCommands.CHECK_SESSION, 0)])
def test_routine_and_excluded_polling_have_no_additional_records(audit_log, name, expected_count):
    auditor, records = audit_log
    _register(auditor, name=name).process_command(Connection(), name)
    assert len(records()) == expected_count
    assert snapshot() == AdminContext()


@pytest.mark.parametrize(
    "decision,action,reason",
    [
        ((False, ""), "admin.permission.denied", "policy_denied"),
        ((False, "private policy failure"), "admin.permission.error", "policy_evaluation_error"),
    ],
)
def test_policy_failure_is_recorded_before_command_audit(audit_log, monkeypatch, decision, action, reason):
    auditor, records = audit_log
    monkeypatch.setattr(AuthorizationService, "authorize", lambda _: decision)
    called = []
    reg = _register(
        auditor,
        handler=lambda *_: called.append(True),
        authz=lambda *_: PreAuthzReturnCode.REQUIRE_AUTHZ,
    )
    reg.process_command(Connection(), "delete_job private-job-argument")
    entries = records()
    assert called == []
    assert len(entries) == 1
    assert entries[0]["action"] == action
    assert f"reason={reason}" in entries[0]["message"]
    assert entries[0]["user"] == "admin@example.com"
    assert "reference" not in entries[0]
    assert "private" not in json.dumps(entries)


def test_explicit_role_denial_does_not_gain_generic_preprocessing_rejection(audit_log):
    from nvflare.private.fed.server.cmd_utils import CommandUtil

    auditor, records = audit_log
    conn = Connection(props={ConnProps.USER_ROLE: "org_admin"})
    reg = _register(auditor, authz=CommandUtil().must_be_project_admin)
    reg.process_command(conn, "delete_job")
    entries = records()
    assert len(entries) == 1
    assert entries[0]["action"] == "admin.permission.denied"
    assert "reason=role_denied" in entries[0]["message"]


def test_preprocessing_rejection_has_one_decision_record(audit_log):
    auditor, records = audit_log
    _register(auditor, authz=lambda *_: PreAuthzReturnCode.ERROR).process_command(Connection(), "delete_job")
    assert len(records()) == 1
    assert records()[0]["action"] == "admin.permission.rejected"


@pytest.mark.parametrize("stage", ["before_handler", "handler", "after_handler"])
def test_processing_errors_describe_the_observed_stage(audit_log, stage):
    auditor, records = audit_log

    def fail(*_):
        raise RuntimeError("secret exception text")

    class BrokenFilter(CommandFilter):
        def pre_command(self, conn, args):
            if stage == "before_handler":
                fail()
            return True

        def post_command(self, conn, args):
            if stage == "after_handler":
                fail()

    reg = _register(auditor, handler=fail if stage == "handler" else None, filters=[BrokenFilter()])
    reg.process_command(Connection(), "delete_job private-argument")
    errors = [r for r in records() if r["action"] == "admin.command.error"]
    assert len(errors) == 1
    assert f"stage={stage}" in errors[0]["message"]
    assert "secret" not in json.dumps(records())
    assert snapshot() == AdminContext()


@pytest.mark.parametrize("kind", ["unknown", "ambiguous", "missing"])
def test_unresolved_commands_never_record_the_untrusted_name(audit_log, kind):
    auditor, records = audit_log
    reg = ServerCommandRegister(None)
    if kind == "ambiguous":
        for scope in ("first", "second"):
            reg.register_module_spec(
                CommandModuleSpec(scope, [CommandSpec("untrusted-secret", "", "", lambda *_: None)])
            )
        reg.finalize()
    reg.process_command(Connection(), "" if kind == "missing" else "untrusted-secret")
    entries = records()
    assert len(entries) == 1
    assert entries[0]["action"] == "admin.command.rejected"
    assert entries[0]["user"] == "unknown"
    assert "untrusted-secret" not in json.dumps(entries)


@pytest.fixture
def session_manager(monkeypatch):
    monkeypatch.setattr("nvflare.fuel.hci.server.sess.threading.Thread.start", lambda _: None)
    manager = SessionManager(SimpleNamespace(fire_and_forget=lambda **_: None), idle_timeout=10)
    yield manager
    manager.shutdown()


def _login_conn(monkeypatch, *, study="default", verifier=True, asserter=True):
    import nvflare.fuel.hci.server.login as login_module

    def verify(**_):
        if isinstance(verifier, Exception):
            raise verifier
        return verifier

    identity = SimpleNamespace(verify_common_name=verify)
    signer = SimpleNamespace(sign=lambda *_args, **_kwargs: "session-signature") if asserter else None
    hci = SimpleNamespace(get_id_verifier=lambda: identity, get_id_asserter=lambda: signer)
    cert = SimpleNamespace(not_valid_after_utc=SimpleNamespace(timestamp=lambda: 9999999999))
    monkeypatch.setattr(login_module, "load_crt_chain_bytes", lambda _: [cert])
    monkeypatch.setattr(login_module, "validate_admin_leaf_cert", lambda _: None)
    monkeypatch.setattr(login_module, "get_admin_study_entitlements", lambda _: ())
    monkeypatch.setattr(login_module, "cert_to_dict", lambda _: {})
    monkeypatch.setattr(login_module, "get_identity_info", lambda _: {})
    monkeypatch.setattr(login_module.StudyRegistryService, "get_registry", lambda: None)
    conn = Connection(
        props={
            ConnProps.CMD_HEADERS: {"cert": "private-certificate", "signature": "private-signature", "study": study},
            ConnProps.HCI_SERVER: hci,
            ConnProps.REQUEST: Message(headers={"origin": "admin/example"}),
        }
    )
    return conn


@pytest.mark.parametrize(
    "case,reason,actor,action",
    [
        ("arguments", "invalid_arguments", "unknown", "rejected"),
        ("server_identity", "server_identity_unavailable", "unknown", "rejected"),
        ("verification", "verification_failed", "unknown", "rejected"),
        ("verification_error", "verification_error", "unknown", "error"),
        ("certificate", "invalid_admin_certificate", "unknown", "rejected"),
        ("study_type", "invalid_study", "admin@example.com", "rejected"),
        ("study_name", "invalid_study", "admin@example.com", "rejected"),
        ("study_missing", "study_not_configured", "admin@example.com", "rejected"),
        ("study_unknown", "unknown_study", "admin@example.com", "rejected"),
        ("membership", "study_membership_denied", "admin@example.com", "rejected"),
    ],
)
def test_login_rejection_and_error_attribution(audit_log, session_manager, monkeypatch, case, reason, actor, action):
    import nvflare.fuel.hci.server.login as login_module

    _, records = audit_log
    study = 123 if case == "study_type" else "invalid/study" if case == "study_name" else "trial"
    conn = _login_conn(
        monkeypatch,
        study=study,
        verifier=RuntimeError("private failure") if case == "verification_error" else case != "verification",
        asserter=case != "server_identity",
    )
    if case == "certificate":

        def bad_cert(_):
            raise ValueError("private certificate")

        monkeypatch.setattr(login_module, "validate_admin_leaf_cert", bad_cert)
    if case in ("study_unknown", "membership"):
        registry = SimpleNamespace(has_study=lambda _: case == "membership", has_user=lambda *_: False)
        monkeypatch.setattr(login_module.StudyRegistryService, "get_registry", lambda: registry)
    login = LoginModule(session_manager)
    with audit_context(AdminContext(request_id="request-login")):
        login.handle_cert_login(
            conn,
            (
                [InternalCommands.CERT_LOGIN]
                if case == "arguments"
                else [InternalCommands.CERT_LOGIN, "admin@example.com"]
            ),
        )
    entries = records()
    assert len(entries) == 1
    assert entries[0]["action"] == f"admin.login.{action}"
    assert entries[0]["user"] == actor
    assert entries[0]["reference"] == "request-login"
    assert f"reason={reason}" in entries[0]["message"]
    assert "private" not in json.dumps(entries)
    assert session_manager.sessions == {}


def test_login_success_links_created_session_and_never_writes_its_secrets(audit_log, session_manager, monkeypatch):
    from nvflare.apis.job_def import DEFAULT_STUDY

    _, records = audit_log
    conn = _login_conn(monkeypatch, study=DEFAULT_STUDY)
    with audit_context(AdminContext(request_id="request-login")):
        LoginModule(session_manager).handle_cert_login(conn, [InternalCommands.CERT_LOGIN, "admin@example.com"])
    entries = records()
    assert [r["action"] for r in entries] == ["admin.session.created", "admin.login.succeeded"]
    assert all(r["user"] == "admin@example.com" and r["reference"] == "request-login" for r in entries)
    session = next(iter(session_manager.sessions.values()))
    assert session.audit_reference == entries[0]["event_id"]
    text = json.dumps(entries)
    assert session.sess_id not in text
    assert "private" not in text
    assert "session-signature" not in text


@pytest.mark.parametrize("reason", ["logout", "inactivity", "certificate_expired"])
def test_actual_session_termination_once_with_fixed_reason(audit_log, session_manager, reason):
    _, records = audit_log
    session = session_manager.create_session("admin@example.com", "org", "role", "admin/example")
    creation = records()[0]
    session_manager.end_session_by_id(session.sess_id, audit_reason=reason)
    session_manager.end_session_by_id(session.sess_id, audit_reason=reason)
    entries = records()
    assert len(entries) == 2
    assert entries[1]["action"] == "admin.session.ended"
    assert entries[1]["reference"] == creation["event_id"]
    assert f"reason={reason}" in entries[1]["message"]
    assert session.sess_id not in json.dumps(entries)


def test_restored_session_has_verified_actor_and_termination_reference(audit_log, session_manager, monkeypatch):
    _, records = audit_log
    session = Session("private-session", "restored@example.com", "org", "role", "")
    monkeypatch.setattr(Session, "decode_token", lambda *_: session)
    session_manager.recreate_session("private-token", "admin/example", None)
    session_manager.end_session_by_id(session.sess_id)
    entries = records()
    assert [r["action"] for r in entries] == ["admin.session.restored", "admin.session.ended"]
    assert all(r["user"] == "restored@example.com" for r in entries)
    assert entries[1]["reference"] == entries[0]["event_id"]
    assert "private" not in json.dumps(entries)


def test_session_emits_outside_shared_lock_and_before_cleanup(audit_log, session_manager, monkeypatch):
    auditor, records = audit_log
    original = auditor.add_event
    checked = []

    def checked_add(*args, **kwargs):
        acquired = session_manager.sess_update_lock.acquire(blocking=False)
        assert acquired
        session_manager.sess_update_lock.release()
        checked.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(auditor, "add_event", checked_add)
    session = session_manager.create_session("admin@example.com", "org", "role", "admin/example")

    def bad_cancel():
        assert records()[-1]["action"] == "admin.session.ended"
        raise RuntimeError("private cleanup failure")

    session_manager.bind_download(session.sess_id, "transaction", bad_cancel)
    session_manager.end_session_by_id(session.sess_id)
    assert len(checked) == 2
    assert len(records()) == 2
    assert session_manager.sessions == {}


def test_invalid_session_record_has_unknown_actor_and_no_token(audit_log, session_manager):
    _, records = audit_log
    conn = Connection()
    with audit_context(AdminContext()):
        assert not LoginModule(session_manager).pre_command(conn, ["delete_job"])
    assert len(records()) == 1
    assert records()[0]["action"] == "admin.session.rejected"
    assert records()[0]["user"] == "unknown"


def test_all_new_records_use_only_existing_string_fields(audit_log, session_manager):
    _, records = audit_log
    session = session_manager.create_session("admin@example.com", "org", "role", "admin/example")
    session_manager.end_session_by_id(session.sess_id)
    allowed = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
    for entry in records():
        assert set(entry) <= allowed
        assert all(isinstance(v, str) for v in entry.values())
        assert entry["timestamp"].endswith("+00:00")


@pytest.mark.parametrize("boundary", ["preprocessing", "policy"])
def test_permission_processing_exception_is_a_single_decision(audit_log, monkeypatch, boundary):
    auditor, records = audit_log

    def fail(*_):
        raise RuntimeError("private policy details")

    monkeypatch.setattr(AuthorizationService, "authorize", fail)
    reg = _register(
        auditor,
        authz=fail if boundary == "preprocessing" else lambda *_: PreAuthzReturnCode.REQUIRE_AUTHZ,
    )
    reg.process_command(Connection(), "delete_job private-argument")
    entries = records()
    decisions = [r for r in entries if r["action"].startswith("admin.permission.")]
    assert len(decisions) == 1
    assert decisions[0]["action"] == "admin.permission.error"
    expected_reason = "preprocessing_error" if boundary == "preprocessing" else "policy_evaluation_error"
    assert f"reason={expected_reason}" in decisions[0]["message"]
    assert len([r for r in entries if r["action"] == "admin.command.error"]) == 1
    assert "private" not in json.dumps(entries)


def test_malformed_certificate_records_processing_error_without_changing_exception(
    audit_log, session_manager, monkeypatch
):
    import nvflare.fuel.hci.server.login as login_module

    _, records = audit_log
    conn = _login_conn(monkeypatch)

    def parse_error(_):
        raise ValueError("private malformed certificate")

    monkeypatch.setattr(login_module, "load_crt_chain_bytes", parse_error)
    with pytest.raises(ValueError, match="private malformed certificate"):
        LoginModule(session_manager).handle_cert_login(conn, [InternalCommands.CERT_LOGIN, "attacker@example.com"])
    assert len(records()) == 1
    assert records()[0]["action"] == "admin.login.error"
    assert records()[0]["user"] == "unknown"
    assert "private" not in json.dumps(records())
    assert "attacker@example.com" not in json.dumps(records())


@pytest.mark.parametrize("certificate_expired", [False, True])
def test_session_monitor_records_the_actual_expiry_reason(audit_log, session_manager, monkeypatch, certificate_expired):
    from tests.timing_utils import isolate_time

    _, records = audit_log
    clock = isolate_time(monkeypatch, "nvflare.fuel.hci.server.sess")
    monkeypatch.setattr(clock, "time", lambda: 0)
    session_manager.create_session(
        "admin@example.com", "org", "role", "admin/example", cert_exp=5 if certificate_expired else None
    )
    monkeypatch.setattr(clock, "time", lambda: 5 if certificate_expired else 11)
    session_manager.check_sessions()
    entries = records()
    assert len(entries) == 2
    expected = "certificate_expired" if certificate_expired else "inactivity"
    assert f"reason={expected}" in entries[-1]["message"]
    assert session_manager.sessions == {}


def test_failed_session_restoration_never_accepts_supplied_identity(audit_log, session_manager, monkeypatch):
    _, records = audit_log
    conn = _login_conn(monkeypatch)
    conn.set_prop(ConnProps.USER_NAME, "unverified@example.com")
    monkeypatch.setattr(conn, "get_token", lambda: "private-session-token")
    monkeypatch.setattr(session_manager, "get_session", lambda *_: None)

    def failed_restore(*_):
        raise ValueError("private-session-token")

    monkeypatch.setattr(session_manager, "recreate_session", failed_restore)
    with audit_context(AdminContext()):
        assert not LoginModule(session_manager).pre_command(conn, ["delete_job"])
    assert len(records()) == 1
    assert records()[0]["user"] == "unknown"
    assert records()[0]["action"] == "admin.session.rejected"
    assert "private" not in json.dumps(records())
    assert "unverified" not in json.dumps(records())


def test_registry_starts_fresh_and_restores_outer_attribution(audit_log):
    auditor, records = audit_log
    with audit_context(AdminContext(user="outer@example.com", request_id="outer-request")):
        _register(auditor).process_command(Connection(), "not-registered-secret")
        assert snapshot().user == "outer@example.com"
        assert snapshot().request_id == "outer-request"
    assert records()[0]["user"] == "unknown"
    assert "reference" not in records()[0]


def test_request_audit_preparation_failure_does_not_reject_operation(audit_log):
    auditor, records = audit_log

    class InvalidEntry:
        @property
        def name(self):
            raise ValueError("private audit construction failure")

    conn = Connection(props={ConnProps.CMD_ENTRY: InvalidEntry()})
    assert CommandAudit(auditor).pre_command(conn, ["delete_job"])
    assert records() == []


def test_closed_writer_does_not_replay_or_block_sensitive_operation(audit_log):
    auditor, records = audit_log
    calls = []
    auditor.close()
    conn = Connection()
    _register(auditor, handler=lambda *_: calls.append("applied")).process_command(conn, "delete_job")
    assert calls == ["applied"]
    assert records() == []
    assert conn.get_prop(ConnProps.EVENT_ID) == ""
