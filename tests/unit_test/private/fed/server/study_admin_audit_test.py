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
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.client import ClientPropKey
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.sec.admin_audit import AdminContext, audit_context, reject
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.fed.server import study_cmds

_ALLOWED_FIELDS = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
_INITIAL = {
    "format_version": "1.0",
    "studies": {"study1": {"site_orgs": {"org_a": ["site-a"]}, "admins": ["verified@example.com"]}},
}


class _Connection:
    def __init__(self, engine, role="org_admin"):
        self.app_ctx = engine
        self.props = {
            ConnProps.USER_NAME: "verified@example.com",
            ConnProps.USER_ROLE: role,
            ConnProps.USER_ORG: "org_a",
        }
        self.replies = []

    def get_prop(self, key, default=None):
        return self.props.get(key, default)

    def append_dict(self, data, meta=None):
        self.replies.append(deepcopy(data))

    def append_error(self, data, meta=None):
        self.replies.append(data)


@pytest.fixture
def study_audit(tmp_path, monkeypatch):
    registry_path = tmp_path / "study_registry.json"
    registry_path.write_text(json.dumps(_INITIAL))
    audit_path = tmp_path / "audit.jsonl"
    writer = Auditor(str(audit_path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    held = [False]

    def acquire(_timeout):
        assert not held[0]
        held[0] = True
        return True

    def release():
        assert held[0]
        held[0] = False

    original_add = writer.add_event

    def write_outside_lock(*args, **kwargs):
        assert not held[0], "administrative recording must not hold the study lock"
        return original_add(*args, **kwargs)

    monkeypatch.setattr(writer, "add_event", write_outside_lock)
    monkeypatch.setattr(study_cmds.StudyRegistryService, "acquire_lock", acquire)
    monkeypatch.setattr(study_cmds.StudyRegistryService, "release_lock", release)
    reloads = []
    monkeypatch.setattr(study_cmds.StudyRegistryService, "initialize", reloads.append)
    engine = MagicMock()
    engine.get_workspace.return_value.get_study_registry_file_path.return_value = str(registry_path)
    engine.get_workspace.return_value.get_study_registry_write_path.return_value = str(registry_path)
    client = MagicMock()
    client.get_prop.side_effect = lambda key, default="": "org_a" if key == ClientPropKey.ORG else default
    engine.client_manager.get_client_from_name.return_value = client
    engine.job_def_manager = None
    monkeypatch.setattr(study_cmds, "ServerEngine", MagicMock)

    def records():
        events = [json.loads(line) for line in audit_path.read_text().splitlines()]
        for event in events:
            assert event.keys() <= _ALLOWED_FIELDS
            assert all(isinstance(value, str) for value in event.values())
            assert "scope" not in event
            assert "job" not in event
            assert event["user"] == "verified@example.com"
            assert event["reference"] == "request-1"
        return events

    fixture = SimpleNamespace(
        module=study_cmds.StudyCommandModule(),
        connection=_Connection(engine),
        engine=engine,
        registry_path=registry_path,
        audit_path=audit_path,
        records=records,
        held=held,
        reloads=reloads,
    )
    with audit_context(AdminContext(user="verified@example.com", request_id="request-1", command="register_study")):
        try:
            yield fixture
        finally:
            writer.close()
    assert not held[0]


def _actions(events):
    return [event["action"] for event in events]


def test_register_records_creation_and_membership_without_raw_arguments(study_audit):
    case = study_audit
    case.module.cmd_register_study(case.connection, ["register_study", "study2", "--sites", "site-a,site-b"])
    stored = json.loads(case.registry_path.read_text())["studies"]["study2"]
    assert stored["site_orgs"]["org_a"] == ["site-a", "site-b"]
    events = case.records()
    assert _actions(events) == [
        "study.register.applied",
        "study.site_add.applied",
        "study.site_add.applied",
        "study.user_add.applied",
    ]
    assert all("study=study2" in event["message"] for event in events)
    assert "--sites" not in case.audit_path.read_text()


def test_register_existing_null_membership_records_both_changes(study_audit):
    case = study_audit
    original = {"format_version": "1.0", "studies": {"study1": {"site_orgs": None, "admins": None}}}
    case.registry_path.write_text(json.dumps(original))
    case.connection.props[ConnProps.USER_ROLE] = "project_admin"

    case.module.cmd_register_study(case.connection, ["register_study", "study1", "--site-org", "org_a:site-a"])

    stored = json.loads(case.registry_path.read_text())["studies"]["study1"]
    assert stored["site_orgs"] == {"org_a": ["site-a"]}
    assert stored["admins"] == ["verified@example.com"]
    assert case.connection.replies[-1]["sites"] == ["site-a"]
    events = case.records()
    assert _actions(events) == ["study.register.applied", "study.site_add.applied", "study.user_add.applied"]
    assert "site=site-a" in events[1]["message"]
    assert "target=verified@example.com" in events[2]["message"]


def test_register_existing_identical_configuration_records_unchanged(study_audit):
    case = study_audit
    case.module.cmd_register_study(case.connection, ["register_study", "study1", "--sites", "site-a"])
    assert _actions(case.records()) == [
        "study.register.unchanged",
        "study.site_add.unchanged",
        "study.user_add.unchanged",
    ]


@pytest.mark.parametrize(
    "command,expected,remaining",
    [
        ("add_study_site", {"site-a": "unchanged", "site-b": "applied"}, ["site-a", "site-b"]),
        ("remove_study_site", {"site-a": "applied", "site-b": "unchanged"}, []),
    ],
)
def test_site_membership_records_each_target_and_unchanged_state(study_audit, command, expected, remaining):
    case = study_audit
    if command == "remove_study_site":
        original = deepcopy(_INITIAL)
        original["studies"]["study1"]["site_orgs"]["org_b"] = ["site-b"]
        case.registry_path.write_text(json.dumps(original))
    getattr(case.module, "cmd_" + command)(case.connection, [command, "study1", "--sites", "site-a,site-b"])
    stored = json.loads(case.registry_path.read_text())["studies"]["study1"]
    assert stored["site_orgs"]["org_a"] == remaining
    events = case.records()
    assert len(events) == 2
    operation = "site_add" if command == "add_study_site" else "site_remove"
    for site, result in expected.items():
        matching = [event for event in events if "site=" + site in event["message"]]
        assert len(matching) == 1
        assert matching[0]["action"] == f"study.{operation}.{result}"


def test_remove_absent_site_does_not_audit_unverified_name(study_audit):
    case = study_audit
    target = "private-token-in-site-position"

    case.module.cmd_remove_study_site(case.connection, ["remove_study_site", "study1", "--sites", "site-a," + target])

    assert case.connection.replies[-1]["removed"] == ["site-a"]
    assert case.connection.replies[-1]["not_enrolled"] == [target]
    case.engine.client_manager.get_client_from_name.assert_not_called()
    removed, absent = case.records()
    assert removed["action"] == "study.site_remove.applied"
    assert "site=site-a" in removed["message"]
    assert absent["action"] == "study.site_remove.unchanged"
    assert "reason=no_change" in absent["message"]
    assert "site=" not in absent["message"]
    assert target not in case.audit_path.read_text()


@pytest.mark.parametrize("command", ["add_study_user", "remove_study_user"])
def test_user_membership_applied_then_unchanged_preserves_reply(study_audit, command):
    case = study_audit
    target = "new@example.com" if command == "add_study_user" else "verified@example.com"
    handler = getattr(case.module, "cmd_" + command)
    handler(case.connection, [command, "study1", target])
    handler(case.connection, [command, "study1", target])
    operation = "user_add" if command == "add_study_user" else "user_remove"
    assert _actions(case.records()) == [f"study.{operation}.applied", f"study.{operation}.unchanged"]
    assert case.connection.replies[-1]["error_code"] in {"USER_ALREADY_IN_STUDY", "USER_NOT_IN_STUDY"}
    applied, unchanged = case.records()
    assert "target=" + target in applied["message"]
    if command == "add_study_user":
        assert "target=" + target in unchanged["message"]
    else:
        assert "reason=not_member" in unchanged["message"]
        assert "target=" not in unchanged["message"]


@pytest.mark.parametrize("command", ["add_study_user", "remove_study_user"])
@pytest.mark.parametrize("same_study", [False, True])
def test_study_lookup_rejection_omits_unverified_user(study_audit, command, same_study):
    case = study_audit
    target = "private-token-in-user-position"
    # A study that exists but is outside the caller's organization has the
    # same established response as one that does not exist.
    case.connection.props[ConnProps.USER_ORG] = "org_b"
    study = "study1" if same_study else "missing"
    initial = case.registry_path.read_text()

    getattr(case.module, "cmd_" + command)(case.connection, [command, study, target])

    assert case.connection.replies[-1]["error_code"] == "STUDY_NOT_FOUND"
    assert case.registry_path.read_text() == initial
    assert not case.reloads
    (event,) = case.records()
    operation = "user_add" if command == "add_study_user" else "user_remove"
    assert event["action"] == f"study.{operation}.rejected"
    assert "reason=study_not_found" in event["message"]
    assert "target=" not in event["message"]
    assert target not in case.audit_path.read_text()


def test_persisted_policy_survives_live_reload_failure(study_audit, monkeypatch):
    case = study_audit

    def fail_reload(_registry):
        raise RuntimeError("session-secret /private/registry/path")

    monkeypatch.setattr(study_cmds.StudyRegistryService, "initialize", fail_reload)
    case.module.cmd_add_study_user(case.connection, ["add_study_user", "study1", "new@example.com"])
    assert "new@example.com" in json.loads(case.registry_path.read_text())["studies"]["study1"]["admins"]
    assert _actions(case.records()) == ["study.user_add.persisted", "study.user_add.reload_error"]
    assert case.connection.replies[-1]["error_code"] == "INTERNAL_ERROR"
    assert "session-secret" not in case.audit_path.read_text()
    assert "/private/registry/path" not in case.audit_path.read_text()


def test_write_failure_does_not_claim_persistence_or_live_reload(study_audit, monkeypatch):
    case = study_audit
    initial = case.registry_path.read_text()

    def fail_write(*_args):
        raise OSError("session-secret /private/registry/path")

    monkeypatch.setattr(case.module, "_write_registry_config", fail_write)
    case.module.cmd_add_study_user(case.connection, ["add_study_user", "study1", "new@example.com"])
    assert case.registry_path.read_text() == initial
    assert not case.reloads
    [event] = case.records()
    assert event["action"] == "study.user_add.error"
    assert "target=" not in event["message"]
    assert "session-secret" not in case.audit_path.read_text()


@pytest.mark.parametrize("command", ["remove_study_site", "remove_study_user"])
def test_failed_removal_keeps_only_established_membership_targets(study_audit, monkeypatch, command):
    case = study_audit
    marker = "unverified-site-marker"
    initial = case.registry_path.read_text()

    def fail_write(*_args):
        raise OSError("private-write-error")

    monkeypatch.setattr(case.module, "_write_registry_config", fail_write)
    if command == "remove_study_site":
        case.module.cmd_remove_study_site(case.connection, [command, "study1", "--sites", "site-a," + marker])
    else:
        case.module.cmd_remove_study_user(case.connection, [command, "study1", "verified@example.com"])

    assert case.connection.replies[-1]["error_code"] == "INTERNAL_ERROR"
    assert case.registry_path.read_text() == initial
    assert not case.reloads
    case.engine.client_manager.get_client_from_name.assert_not_called()
    events = case.records()
    assert all("reason=processing_error" in event["message"] for event in events)
    if command == "remove_study_site":
        assert _actions(events) == ["study.site_remove.error", "study.site_remove.error"]
        assert "site=site-a" in events[0]["message"]
        assert "site=" not in events[1]["message"]
    else:
        assert _actions(events) == ["study.user_remove.error"]
        assert "target=verified@example.com" in events[0]["message"]
    assert marker not in case.audit_path.read_text()
    assert "private-write-error" not in case.audit_path.read_text()


def test_invalid_registry_membership_does_not_establish_site_target(study_audit):
    case = study_audit
    marker = "unverified-site-marker"
    registry = json.loads(case.registry_path.read_text())
    registry["studies"]["study1"]["site_orgs"]["org_b"] = "prefix-" + marker + "-suffix"
    initial = json.dumps(registry)
    case.registry_path.write_text(initial)

    case.module.cmd_remove_study_site(case.connection, ["remove_study_site", "study1", "--sites", "site-a," + marker])

    assert case.connection.replies[-1]["error_code"] == "INTERNAL_ERROR"
    assert case.registry_path.read_text() == initial
    assert not case.reloads
    events = case.records()
    assert _actions(events) == ["study.site_remove.error", "study.site_remove.error"]
    assert "site=site-a" in events[0]["message"]
    assert "site=" not in events[1]["message"]
    assert marker not in case.audit_path.read_text()


def test_applied_change_survives_reply_failure(study_audit, monkeypatch):
    case = study_audit
    calls = []

    def reply_once_failed(data, meta=None):
        calls.append(data)
        if len(calls) == 1:
            raise RuntimeError("reply failed")

    monkeypatch.setattr(case.connection, "append_dict", reply_once_failed)
    case.module.cmd_add_study_user(case.connection, ["add_study_user", "study1", "new@example.com"])
    assert "new@example.com" in json.loads(case.registry_path.read_text())["studies"]["study1"]["admins"]
    assert len(case.reloads) == 1
    assert _actions(case.records()) == ["study.user_add.applied"]


def test_remove_study_rejected_while_associated_jobs_exist(study_audit):
    from nvflare.apis.job_def import JobMetaKey
    from nvflare.apis.job_def_manager_spec import JobDefManagerSpec

    case = study_audit
    manager = MagicMock(spec=JobDefManagerSpec)
    manager.get_all_jobs.return_value = [SimpleNamespace(meta={JobMetaKey.STUDY.value: "study1"})]
    case.engine.job_def_manager = manager
    case.module.cmd_remove_study(case.connection, ["remove_study", "study1"])
    assert "study1" in json.loads(case.registry_path.read_text())["studies"]
    assert _actions(case.records()) == ["study.remove.rejected"]
    assert "reason=study_has_jobs" in case.records()[0]["message"]


def test_remove_study_applied_after_registry_write_and_reload(study_audit):
    case = study_audit
    case.module.cmd_remove_study(case.connection, ["remove_study", "study1"])
    assert "study1" not in json.loads(case.registry_path.read_text())["studies"]
    assert _actions(case.records()) == ["study.remove.applied"]


def test_argument_rejection_omits_unvalidated_study_and_raw_input(study_audit):
    case = study_audit
    case.module.cmd_add_study_user(case.connection, ["add_study_user", "../session-secret", "new@example.com"])
    assert _actions(case.records()) == ["study.user_add.rejected"]
    assert "session-secret" not in case.audit_path.read_text()
    assert "new@example.com" not in case.audit_path.read_text()


def test_role_rejection_is_one_decision_even_with_outer_preprocessor_rejection(study_audit):
    case = study_audit
    case.connection.props[ConnProps.USER_ROLE] = "member"
    assert case.module.authorize_study_admin(case.connection, []) == PreAuthzReturnCode.ERROR
    reject("authorization.preprocess.rejected", reason="preprocessing_rejected")
    assert _actions(case.records()) == ["authorization.role.denied"]


def test_lock_rejection_is_recorded_without_mutating_configuration(study_audit, monkeypatch):
    case = study_audit
    monkeypatch.setattr(study_cmds.StudyRegistryService, "acquire_lock", lambda _timeout: False)
    initial = case.registry_path.read_text()
    case.module.cmd_add_study_user(case.connection, ["add_study_user", "study1", "new@example.com"])
    assert case.registry_path.read_text() == initial
    assert _actions(case.records()) == ["study.user_add.rejected"]
    assert "reason=lock_timeout" in case.records()[0]["message"]


@pytest.mark.parametrize("command", ["register_study", "add_study_site"])
@pytest.mark.parametrize("has_known_site", [False, True])
def test_membership_batch_rejection_omits_unverified_sites(study_audit, command, has_known_site):
    case = study_audit
    target = "private-token-in-site-position"
    manager = case.engine.client_manager
    known_site = manager.get_client_from_name.return_value
    manager.get_client_from_name.side_effect = lambda name: known_site if has_known_site and name == "site-a" else None
    before = case.registry_path.read_text()

    getattr(case.module, "cmd_" + command)(case.connection, [command, "study1", "--sites", "site-a," + target])

    assert case.connection.replies[-1]["error_code"] == "INVALID_SITE"
    assert case.registry_path.read_text() == before
    events = case.records()
    if command == "register_study":
        assert events[0]["action"] == "study.register.rejected"
        events = events[1:]
    assert _actions(events) == ["study.site_add.rejected", "study.site_add.rejected"]
    assert all("reason=invalid_site" in event["message"] for event in events)
    if has_known_site:
        assert "site=site-a" in events[0]["message"]
    else:
        assert "site=" not in events[0]["message"]
    assert "site=" not in events[1]["message"]
    assert target not in case.audit_path.read_text()


def test_failed_audit_writer_does_not_prevent_study_update(study_audit, monkeypatch):
    case = study_audit

    def fail_write(*_args, **_kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(AuditService.the_auditor, "add_event", fail_write)
    case.module.cmd_add_study_user(case.connection, ["add_study_user", "study1", "new@example.com"])
    assert "new@example.com" in json.loads(case.registry_path.read_text())["studies"]["study1"]["admins"]
    assert "error_code" not in case.connection.replies[-1]
    assert not case.records()


def test_percent_email_membership_add_and_remove_have_applied_records(study_audit):
    case = study_audit
    target = "valid%tag@example.com"
    case.module.cmd_add_study_user(case.connection, ["add_study_user", "study1", target])
    assert target in json.loads(case.registry_path.read_text())["studies"]["study1"]["admins"]
    case.module.cmd_remove_study_user(case.connection, ["remove_study_user", "study1", target])
    assert target not in json.loads(case.registry_path.read_text())["studies"]["study1"]["admins"]
    events = case.records()
    assert _actions(events) == ["study.user_add.applied", "study.user_remove.applied"]
    assert all("target=" + target in event["message"] for event in events)


def test_register_percent_email_caller_has_applied_membership_record(study_audit):
    case = study_audit
    caller = "valid%tag@example.com"
    case.connection.props[ConnProps.USER_NAME] = caller
    with audit_context(AdminContext(user=caller, request_id="request-1", command="register_study")):
        case.module.cmd_register_study(case.connection, ["register_study", "study2", "--sites", "site-a"])
    assert caller in json.loads(case.registry_path.read_text())["studies"]["study2"]["admins"]
    events = [json.loads(line) for line in case.audit_path.read_text().splitlines()]
    assert _actions(events) == ["study.register.applied", "study.site_add.applied", "study.user_add.applied"]
    assert "target=" + caller in events[-1]["message"]
    for event in events:
        assert event.keys() <= _ALLOWED_FIELDS
        assert all(isinstance(value, str) for value in event.values())
        assert event["user"] == caller
        assert event["reference"] == "request-1"
        assert "scope" not in event
        assert "job" not in event
