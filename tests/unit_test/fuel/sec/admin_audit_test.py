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
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.sec import admin_audit as audit
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.fed.simulator.simulator_audit import SimulatorAuditor

FIELDS = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}


@pytest.fixture
def writer(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    monkeypatch.setattr(audit, "_failed_attempts", 0)
    monkeypatch.setattr(audit, "_last_warning", float("-inf"))
    yield writer, path
    writer.close()


def records(path):
    result = [json.loads(line) for line in path.read_text().splitlines()]
    for item in result:
        assert set(item) <= FIELDS
        assert all(isinstance(value, str) for value in item.values())
        assert item["timestamp"].endswith("+00:00")
    return result


def test_actor_job_scope_and_reference_without_new_fields(writer):
    _, path = writer
    with audit.audit_context(audit.AdminContext(user="verified-admin", command="delete_job")):
        request = audit.record("delete_job", ref="")
        audit.update_context(request_id=request, job_id="job-1", scope_name="research")
        event = audit.record("job.delete.applied", reason="manager_completed")
    first, second = records(path)
    assert first["event_id"] == request
    assert first["user"] == second["user"] == "verified-admin"
    assert "reference" not in first and "job" not in first and "scope" not in first
    assert set(second) == FIELDS
    assert second["reference"] == request
    assert second["event_id"] == event
    assert second["job"] == "job-1" and second["scope"] == "research"
    assert audit.snapshot() == audit.AdminContext()


def test_nested_context_reset_and_immutable_callback_snapshot(writer):
    _, path = writer
    with audit.audit_context(audit.AdminContext(user="alice", request_id="original")):
        saved = audit.snapshot()
        with pytest.raises(RuntimeError), audit.audit_context(user="bob", request_id="other"):
            audit.record("admin.command.dispatched")
            raise RuntimeError("operation error")
        assert audit.snapshot() == saved
    with pytest.raises(FrozenInstanceError):
        saved.user = "mallory"
    audit.record("download.transfer.completed", context=saved, reason="all_receivers_succeeded")
    assert [r["user"] for r in records(path)] == ["bob", "alice"]
    assert records(path)[-1]["reference"] == "original"


def test_request_context_does_not_cross_threads(writer):
    from tests.timing_utils import ThreadGroup

    _, path = writer
    with audit.audit_context(audit.AdminContext(user="alice", request_id="alice-request")):
        with ThreadGroup() as workers:
            workers.start(lambda: audit.record("admin.command.dispatched"))
    assert records(path)[0]["user"] == "unknown"
    assert "reference" not in records(path)[0]


def test_rejection_once_per_request_and_next_request_independent(writer):
    _, path = writer
    for _ in range(2):
        with audit.audit_context(audit.AdminContext()):
            assert audit.reject("admin.permission.denied", reason="role_denied")
            assert audit.reject("admin.permission.rejected", reason="preprocessing_rejected") == ""
    assert len(records(path)) == 2


@pytest.mark.parametrize("invalid", [None, 3, "private/path", "bad\nidentifier", "x" * 257])
def test_bad_context_preparation_is_contained(writer, invalid):
    _, path = writer
    with audit.audit_context(job_id=invalid):
        assert audit.snapshot() == audit.AdminContext()
        assert audit.record("admin.command.dispatched")
    assert records(path)[0]["user"] == "unknown"
    assert audit._failed_attempts == 1


def test_bad_record_data_does_not_leak_secrets(writer):
    _, path = writer
    assert audit.record("admin.permission.denied", credential="session-secret") == ""
    assert audit.record("admin.permission.denied", reason="private/path/secret") == ""
    assert audit.record("admin.permission.denied", context=object()) == ""
    assert audit.prepare(lambda: (_ for _ in ()).throw(ValueError("credential-secret"))) is None
    assert audit._failed_attempts == 4
    assert records(path) == []


@pytest.mark.parametrize("result", ["", None, 4])
def test_unexpected_empty_or_invalid_writer_return_is_failure(writer, result):
    _, path = writer
    fake = SimpleNamespace(add_event=lambda **kwargs: result)
    assert audit.record("admin.command.dispatched", writer=fake) == ""
    assert audit._failed_attempts == 1
    assert records(path) == []


def test_missing_and_closed_writers_are_failures(writer, monkeypatch):
    instance, _ = writer
    monkeypatch.setattr(AuditService, "the_auditor", None)
    assert audit.record("admin.command.dispatched") == ""
    instance.close()
    assert audit.record("admin.command.dispatched", writer=instance) == ""
    assert audit._failed_attempts == 2


@pytest.mark.parametrize("boundary", ["write", "flush"])
def test_write_and_flush_failure_return_no_fabricated_identifier(writer, monkeypatch, boundary):
    instance, path = writer
    underlying = instance.audit_file

    class FailingFile:
        def write(self, text):
            if boundary == "write":
                raise OSError("private filename and credential")
            return underlying.write(text)

        def flush(self):
            raise OSError("session-secret")

    monkeypatch.setattr(instance, "audit_file", FailingFile())
    assert audit.record("admin.command.dispatched") == ""
    assert audit._failed_attempts == 1
    monkeypatch.setattr(instance, "audit_file", underlying)
    # A flush error may occur after writing: failed attempts are not exact lost counts.
    underlying.flush()
    assert len(records(path)) == (boundary == "flush")


def test_simulator_and_excluded_polling_are_intentionally_silent(writer, monkeypatch):
    _, path = writer
    monkeypatch.setattr(AuditService, "the_auditor", SimulatorAuditor())
    assert audit.record("admin.command.dispatched") == ""
    monkeypatch.setattr(AuditService, "the_auditor", None)
    assert audit.record("_check_session") == ""
    assert audit._failed_attempts == 0
    assert records(path) == []


def test_diagnostics_throttled_with_protected_fixed_fallback(writer, monkeypatch):
    calls = []
    monkeypatch.setattr(AuditService, "the_auditor", None)
    clock = Mock(return_value=10.0)
    monkeypatch.setattr(audit, "time", SimpleNamespace(monotonic=clock))
    warning = Mock(side_effect=RuntimeError("diagnostic-secret"))
    monkeypatch.setattr(audit, "_logger", SimpleNamespace(warning=warning))
    monkeypatch.setattr(audit.os, "write", lambda fd, data: calls.append((fd, data)))
    for _ in range(4):
        assert audit.record("admin.command.dispatched") == ""
    assert len(calls) == warning.call_count == 1
    assert calls[0] == (2, b"Administrative audit recording failed; operation continued.\n")
    clock.return_value = 70.0
    assert audit.record("admin.command.dispatched") == ""
    assert len(calls) == warning.call_count == 2
    assert warning.call_args.args[-1] == 5
    monkeypatch.setattr(audit.os, "write", Mock(side_effect=OSError("fallback-secret")))
    clock.return_value = 130.0
    assert audit.record("admin.command.dispatched") == ""
    assert audit._failed_attempts == 6


def test_ordinary_audit_callers_keep_failure_semantics(writer, monkeypatch):
    instance, _ = writer
    monkeypatch.setattr(instance, "add_event", Mock(side_effect=OSError("writer failure")))
    with pytest.raises(OSError):
        AuditService.add_event("old-user", "old-action")
    assert audit.record("admin.command.dispatched") == ""


def test_job_context_arguments_are_keyword_only_and_service_compatible(writer):
    _, path = writer
    event = AuditService.add_event("admin", "operation", job_id="job", scope_name="scope")
    assert records(path)[0]["event_id"] == event
    assert records(path)[0]["job"] == "job"
    with pytest.raises(TypeError):
        AuditService.add_event("admin", "operation", "", "", "job", "scope")


def test_sensitive_operations_are_explicit():
    assert audit.is_sensitive("study_mgmt.remove_study")
    assert audit.is_sensitive("sys.shell")
    assert not audit.is_sensitive("check_status")
    assert not audit.is_sensitive("_check_session")


def test_existing_writer_subclass_signature_keeps_ordinary_admin_events(tmp_path):
    class ExistingAuditor(Auditor):
        def add_event(self, user, action, ref="", msg=""):
            return super().add_event(user, action, ref, msg)

    path = tmp_path / "custom.jsonl"
    writer = ExistingAuditor(str(path))
    try:
        request = audit.record("delete_job", writer=writer, user="admin")
        assert request
        assert records(path)[0]["event_id"] == request
        assert records(path)[0]["user"] == "admin"
    finally:
        writer.close()


def test_supported_privacy_scope_names_preserve_job_and_actor(writer):
    _, path = writer
    with audit.audit_context(user="admin@test.org", request_id="request"):
        audit.update_context(job_id="job-1", scope_name="research cohort")
        assert audit.record("job.delete.applied")
    event = records(path)[0]
    assert event["job"] == "job-1"
    assert event["scope"] == "research cohort"
    assert event["user"] == "admin@test.org"
    assert event["reference"] == "request"
    assert audit._failed_attempts == 0


def test_invalid_optional_scope_does_not_discard_job_or_keep_old_scope(writer):
    _, path = writer
    with audit.audit_context(user="admin@test.org", request_id="request", job_id="old-job", scope_name="old-scope"):
        audit.update_context(job_id="new-job", scope_name=object())
        assert audit.record("job.delete.applied")
    event = records(path)[0]
    assert event["job"] == "new-job"
    assert "scope" not in event
    assert event["user"] == "admin@test.org"
    assert event["reference"] == "request"
    assert audit._failed_attempts == 1
