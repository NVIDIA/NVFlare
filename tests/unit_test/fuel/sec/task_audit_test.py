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

import gc
import json
import threading
import weakref
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.fuel.sec import admin_audit
from nvflare.fuel.sec import task_audit as audit
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.fed.simulator.simulator_audit import SimulatorAuditor
from tests.timing_utils import ThreadGroup

FIELDS = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}


@pytest.fixture
def writer(tmp_path, monkeypatch):
    path = tmp_path / "task-audit.jsonl"
    instance = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", instance)
    monkeypatch.setattr(admin_audit, "_failed_attempts", 0)
    monkeypatch.setattr(admin_audit, "_last_warning", float("-inf"))
    yield instance, path
    instance.close()


def records(path):
    result = [json.loads(line) for line in path.read_text().splitlines()]
    for item in result:
        assert set(item) <= FIELDS
        assert all(isinstance(value, str) for value in item.values())
        assert datetime.fromisoformat(item["timestamp"]).utcoffset() == timedelta(0)
    return result


def test_real_writer_keeps_existing_fields_and_quotes_approved_resource_names(writer):
    _, path = writer
    context = audit.build_context(
        job_id="job-1",
        scope_name="研究 cohort 50%\nnext",
        task_name="train 50% 研究\nnext",
        task_id="task 1",
        workflow_id="flow 研究",
        attempt_id="attempt%1",
        site="site A",
        reference="established-reference",
    )
    event = audit.record(
        "task.execution.completed", context=context, reason="executor_returned", status="OK", stage="execute"
    )
    assert event
    [item] = records(path)
    assert set(item) == FIELDS
    assert item["event_id"] == event
    assert item["user"] == "unknown"
    assert item["job"] == "job-1"
    assert item["scope"] == context.scope_name
    assert item["reference"] == "established-reference"
    assert item["action"] == "task.execution.completed"
    assert item["message"] == (
        'reason=executor_returned status=OK stage=execute task="train 50% \\u7814\\u7a76\\nnext" '
        'task_id="task 1" workflow="flow \\u7814\\u7a76" attempt_id="attempt%1" site="site A"'
    )
    assert len(path.read_text().splitlines()) == 1


def test_empty_optional_values_are_omitted_and_references_are_not_invented(writer):
    _, path = writer
    first = audit.record("workflow.started")
    second = audit.record("workflow.completed")
    items = records(path)
    assert [item["event_id"] for item in items] == [first, second]
    assert first and second and first != second
    assert all(set(item) == {"event_id", "timestamp", "user", "action"} for item in items)


def test_context_reads_only_local_job_and_scope_and_copies_values_without_retaining_source(writer):
    _, path = writer

    class LocalContext:
        def __init__(self):
            self.job = "local-job"
            self.scope = "local scope"

        def get_job_id(self):
            return self.job

        def get_prop(self, name, default):
            assert name == FLContextKey.EFFECTIVE_JOB_SCOPE_NAME
            assert default == ""
            return self.scope

    source = LocalContext()
    source_ref = weakref.ref(source)
    context = audit.build_context(source, task_name="approved-task")
    source.job = "other-job"
    source.scope = "other-scope"
    del source
    gc.collect()
    assert source_ref() is None
    assert context == audit.TaskAuditContext(job_id="local-job", scope_name="local scope", task_name="approved-task")
    assert all(isinstance(getattr(context, field.name), str) for field in fields(context))
    with pytest.raises(FrozenInstanceError):
        context.job_id = "changed"
    assert audit.record("task.execution.started", context=context)
    assert records(path)[0]["job"] == "local-job"


def test_explicit_context_values_do_not_read_unneeded_source_properties(writer):
    class UnusableSource:
        def get_job_id(self):
            raise AssertionError("must use explicit job")

        def get_prop(self, *args):
            raise AssertionError("must use explicit scope")

    context = audit.build_context(UnusableSource(), job_id="approved-job", scope_name=None)
    assert context == audit.TaskAuditContext(job_id="approved-job")
    assert admin_audit._failed_attempts == 0


def test_nested_context_refinement_resets_even_when_operation_raises(writer):
    _, path = writer
    original = audit.current_context()
    outer = audit.TaskAuditContext(job_id="outer")
    inner = audit.TaskAuditContext(job_id="inner")
    refined = audit.TaskAuditContext(job_id="refined")
    with audit.use_context(outer):
        with pytest.raises(RuntimeError, match="operation failed"), audit.use_context(inner):
            assert audit.current_context() == inner
            audit.set_context(refined)
            audit.record("task.execution.started")
            raise RuntimeError("operation failed")
        assert audit.current_context() == outer
        audit.record("workflow.completed")
    assert audit.current_context() == original
    assert [item["job"] for item in records(path)] == ["refined", "outer"]


def test_context_isolated_during_overlapping_threads(writer):
    _, path = writer
    ready = [threading.Event(), threading.Event()]
    release = threading.Event()
    initial = []

    def worker(index):
        initial.append(audit.current_context())
        with audit.use_context(audit.TaskAuditContext(job_id=f"thread-{index}")):
            ready[index].set()
            assert release.wait(10)
            assert audit.record("task.execution.started")

    with audit.use_context(audit.TaskAuditContext(job_id="parent")):
        with ThreadGroup(release) as workers:
            workers.start(worker, 0)
            workers.start(worker, 1)
            assert all(event.wait(10) for event in ready)
            assert audit.current_context().job_id == "parent"
            release.set()
    assert initial == [audit.TaskAuditContext(), audit.TaskAuditContext()]
    assert {item["job"] for item in records(path)} == {"thread-0", "thread-1"}


def test_no_argument_inherits_context_but_explicit_none_never_inherits(writer):
    _, path = writer
    with audit.use_context(audit.TaskAuditContext(job_id="current-job", reference="current-reference")):
        assert audit.record("workflow.started")
        assert audit.record("workflow.started", context=None) == ""
    [item] = records(path)
    assert item["job"] == "current-job"
    assert item["reference"] == "current-reference"
    assert admin_audit._failed_attempts == 1


@pytest.mark.parametrize("field", [field.name for field in fields(audit.TaskAuditContext)])
def test_every_context_field_requires_a_string(writer, field):
    _, path = writer
    assert audit.build_context(**{field: object()}) is None
    assert audit.record("task.execution.started", context=audit.TaskAuditContext(**{field: object()})) == ""
    assert records(path) == []
    assert admin_audit._failed_attempts == 2


def test_invalid_context_replaces_stale_attribution_until_scope_reset(writer):
    _, path = writer
    with audit.use_context(audit.TaskAuditContext(job_id="old-job")):
        with audit.use_context(None):
            assert audit.current_context() is None
            assert audit.record("workflow.started") == ""
        assert audit.current_context().job_id == "old-job"
        assert audit.set_context(object()) is None
        assert audit.record("workflow.started") == ""
    assert records(path) == []
    assert admin_audit._failed_attempts == 4


@pytest.mark.parametrize("field", ["action", "reason", "status", "stage"])
@pytest.mark.parametrize("invalid", ["private/path/secret", "line\nbreak", "x" * 129, 7, None])
def test_fixed_tokens_reject_private_or_invalid_values_before_writing(writer, field, invalid):
    _, path = writer
    kwargs = {"action": "task.execution.completed", field: invalid}
    assert audit.record(**kwargs) == ""
    assert records(path) == []
    assert admin_audit._failed_attempts == 1


def test_return_codes_only_expose_built_in_values(writer):
    supplied_codes = [value for name, value in vars(ReturnCode).items() if name.isupper() and isinstance(value, str)]
    assert supplied_codes
    for code in supplied_codes:
        assert audit.code_status(code) == code
    assert audit.code_status(None) == "unspecified"
    for code in ("secret/path/private-code", "almost_OK", object(), [ReturnCode.OK]):
        assert audit.code_status(code) == "custom_return_code"
    _, path = writer
    assert audit.record("task.execution.completed", status=audit.code_status("private-code"))
    assert records(path)[0]["message"] == "status=custom_return_code"


@pytest.mark.parametrize("state", ["missing", "closed", "empty", "none", "invalid"])
def test_unavailable_or_nonacknowledging_writer_is_diagnosed(writer, monkeypatch, state):
    instance, path = writer
    if state == "missing":
        monkeypatch.setattr(AuditService, "the_auditor", None)
    elif state == "closed":
        instance.close()
    else:
        result = {"empty": "", "none": None, "invalid": 42}[state]
        monkeypatch.setattr(AuditService, "the_auditor", SimpleNamespace(add_event=lambda **kwargs: result))
    assert audit.record("workflow.completed") == ""
    assert admin_audit._failed_attempts == 1
    assert records(path) == []


@pytest.mark.parametrize("boundary", ["write", "flush"])
def test_real_writer_failure_does_not_fabricate_acknowledgement_or_leak_exception(writer, monkeypatch, boundary, capfd):
    instance, path = writer
    underlying = instance.audit_file

    class FailingFile:
        def write(self, text):
            if boundary == "write":
                raise OSError("secret-private-path")
            return underlying.write(text)

        def flush(self):
            raise OSError("secret-private-path")

    with monkeypatch.context() as patch:
        patch.setattr(instance, "audit_file", FailingFile())
        assert audit.record("workflow.completed") == ""
    underlying.flush()
    # A failed flush can follow a write; the diagnostic counts attempts, not exact lost records.
    assert len(records(path)) == (1 if boundary == "flush" else 0)
    assert admin_audit._failed_attempts == 1
    assert capfd.readouterr().err == "Audit recording failed; failed attempts since startup: 1\n"
    assert "secret-private-path" not in path.read_text()


def test_simulator_marker_is_intentionally_silent(writer, monkeypatch):
    _, path = writer
    monkeypatch.setattr(AuditService, "the_auditor", SimulatorAuditor())
    with audit.collect():
        assert audit.record("task.execution.started") == ""
    assert admin_audit._failed_attempts == 0
    assert records(path) == []


def test_preparation_and_diagnostic_failures_cannot_change_operation_result(writer, monkeypatch):
    _, path = writer

    def broken_preparation():
        raise ValueError("secret-preparation-value")

    def broken_diagnostic():
        raise OSError("secret-diagnostic-value")

    monkeypatch.setattr(audit, "recording_failed", broken_diagnostic)

    @audit.collect_events
    def operation():
        assert audit.prepare(broken_preparation, default="safe-default") == "safe-default"
        assert audit.build_context(SimpleNamespace()) is None
        assert audit.record("task.execution.failed", context=None) == ""
        return "actual-operation-result"

    assert operation() == "actual-operation-result"
    assert records(path) == []


def test_normal_writer_exceptions_are_contained_but_legacy_calls_still_raise(writer, monkeypatch):
    instance, _ = writer

    def broken_writer(*args, **kwargs):
        raise OSError("writer failed")

    monkeypatch.setattr(instance, "add_event", broken_writer)
    assert audit.record("workflow.completed") == ""
    with pytest.raises(OSError, match="writer failed"):
        AuditService.add_event("legacy-user", "legacy.action")
    assert admin_audit._failed_attempts == 1


@pytest.mark.parametrize("boundary", ["prepare", "writer"])
def test_process_control_exceptions_are_not_suppressed(writer, monkeypatch, boundary):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        if boundary == "prepare":
            audit.prepare(interrupt)
        else:
            monkeypatch.setattr(writer[0], "add_event", interrupt)
            audit.record("workflow.completed")
    assert admin_audit._failed_attempts == 0


def test_nested_collectors_defer_records_and_diagnostics_until_after_lock_release(writer, monkeypatch):
    instance, path = writer
    operation_lock = threading.Lock()
    observations = []
    real_add_event = instance.add_event
    real_diagnostic = audit.recording_failed

    def assert_unlocked(kind):
        acquired = operation_lock.acquire(blocking=False)
        observations.append((kind, acquired))
        if acquired:
            operation_lock.release()

    def write(**kwargs):
        assert_unlocked("write")
        return real_add_event(**kwargs)

    def diagnose():
        assert_unlocked("diagnostic")
        real_diagnostic()

    monkeypatch.setattr(instance, "add_event", write)
    monkeypatch.setattr(audit, "recording_failed", diagnose)

    @audit.collect_events
    def inner():
        with operation_lock:
            assert audit.record("task.execution.started") == ""
            assert audit.record("invalid/action") == ""
            assert observations == []
            assert records(path) == []

    with pytest.raises(RuntimeError, match="operation failed"), audit.collect():
        inner()
        assert observations == []
        assert audit.record("workflow.failed") == ""
        raise RuntimeError("operation failed")
    assert observations == [("write", True), ("write", True), ("diagnostic", True)]
    assert [item["action"] for item in records(path)] == ["task.execution.started", "workflow.failed"]
    assert admin_audit._failed_attempts == 1


def test_collector_captures_each_context_at_observation_time(writer):
    _, path = writer
    with audit.collect():
        with audit.use_context(audit.TaskAuditContext(job_id="first")):
            audit.record("task.execution.started")
        with audit.use_context(audit.TaskAuditContext(job_id="second")):
            audit.record("task.execution.completed")
        assert records(path) == []
    assert [item["job"] for item in records(path)] == ["first", "second"]


def test_collectors_do_not_cross_threads(writer):
    _, path = writer
    release = threading.Event()
    observed = threading.Event()
    finished = threading.Event()

    def worker():
        with audit.collect():
            audit.record("worker.started")
            observed.set()
            assert release.wait(10)
        finished.set()

    with ThreadGroup(release) as workers:
        workers.start(worker)
        assert observed.wait(10)
        assert records(path) == []
        assert audit.record("parent.started")
        assert [item["action"] for item in records(path)] == ["parent.started"]
        release.set()
        assert finished.wait(10)
    assert [item["action"] for item in records(path)] == ["parent.started", "worker.started"]


def test_collector_has_finite_capacity_and_reports_overflow_after_flush(writer):
    _, path = writer
    with audit.collect():
        for _ in range(1027):
            assert audit.record("task.result.received") == ""
        assert records(path) == []
        assert admin_audit._failed_attempts == 0
    assert len(records(path)) == 1024
    assert admin_audit._failed_attempts == 3
    assert audit.record("workflow.completed")
    assert len(records(path)) == 1025


def test_collector_detaches_before_diagnostic_reentry(writer, monkeypatch):
    _, path = writer
    diagnostics = []

    def diagnose():
        diagnostics.append("called")
        assert audit.record("diagnostic.observed")

    monkeypatch.setattr(audit, "recording_failed", diagnose)
    with audit.collect():
        audit.record("invalid/action")
    assert diagnostics == ["called"]
    assert [item["action"] for item in records(path)] == ["diagnostic.observed"]


def test_reentrant_writer_does_not_recurse_and_later_records_still_work(writer, monkeypatch):
    instance, path = writer
    original = instance.add_event
    calls = []

    def reenter(**kwargs):
        calls.append(kwargs["action"])
        assert audit.record("recursive.observation") == ""
        return original(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(instance, "add_event", reenter)
        with audit.collect():
            audit.record("workflow.completed")
    assert calls == ["workflow.completed"]
    assert admin_audit._failed_attempts == 1
    assert audit.record("workflow.next")
    assert [item["action"] for item in records(path)] == ["workflow.completed", "workflow.next"]


def test_administrative_context_cannot_supply_task_actor_or_reference(writer):
    _, path = writer
    with admin_audit.audit_context(user="verified-admin", request_id="administrative-request", job_id="admin-job"):
        assert audit.record("workflow.started")
    [item] = records(path)
    assert item["user"] == "unknown"
    assert "reference" not in item and "job" not in item


def test_existing_four_argument_writer_remains_usable_without_job_context(writer, monkeypatch):
    instance, path = writer
    original = instance.add_event

    def existing_signature(user, action, ref="", msg=""):
        return original(user, action, ref, msg)

    monkeypatch.setattr(instance, "add_event", existing_signature)
    event = audit.record("workflow.started")
    assert event
    assert records(path)[0]["event_id"] == event
    assert admin_audit._failed_attempts == 0


def test_writer_lookup_failure_is_contained_and_diagnosed(writer, monkeypatch):
    _, path = writer

    def unavailable():
        raise OSError("private service detail")

    monkeypatch.setattr(AuditService, "get_auditor", unavailable)
    assert audit.record("workflow.started") == ""
    assert admin_audit._failed_attempts == 1
    assert records(path) == []
