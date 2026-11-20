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
import threading
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FilterKey, FLContextKey, ReservedKey, ReturnCode, TaskResultReceipt
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import TaskExecutionInterrupted
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.fuel.sec.task_audit import TaskAuditContext, current_context
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.client import client_runner as subject
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from tests.timing_utils import ThreadGroup

FIELDS = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}


class _Executor(Executor):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.callback = lambda _name, _data, _ctx, _abort: make_reply(ReturnCode.OK)

    def execute(self, name, data, ctx, abort):
        self.calls += 1
        return self.callback(name, data, ctx, abort)


@pytest.fixture
def system(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    runner = subject.ClientRunner.__new__(subject.ClientRunner)
    runner.task_router = subject.TaskRouter()
    runner.task_data_filters = {}
    runner.task_result_filters = {}
    runner.default_task_fetch_interval = 0.5
    runner.parent_target = "server"
    runner.job_id = "job-1"
    runner.engine = Mock()
    runner.run_abort_signal = Signal()
    runner._run_abort_requested = False
    runner.task_lock = threading.Lock()
    runner.running_tasks = {}
    runner.task_check_timeout = 5.0
    runner.task_check_interval = 0.0
    runner.get_task_timeout = 5.0
    runner.submit_task_result_timeout = 5.0
    for method in ("log_debug", "log_info", "log_error", "log_exception", "fire_event", "fire_event_with_data"):
        setattr(runner, method, Mock())
    executor = _Executor()
    runner.task_router.add_executor(["train"], executor)
    ctx = FLContext()
    ctx.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=False)
    ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "research cohort", private=True, sticky=False)
    peer = FLContext()
    peer.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    peer.set_prop(ReservedKey.IDENTITY_NAME, "server", private=False, sticky=False)
    ctx.set_peer_context(peer)
    data = Shareable({"payload": "private-task-payload"})
    data.set_header(ReservedKey.AUDIT_EVENT_ID, "server-audit")
    data.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt-1")
    data.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow-1")
    task = TaskAssignment("train", "task-1", data)
    yield runner, executor, ctx, task, writer, path
    writer.close()


def _all_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _records(path):
    all_records = _all_records(path)
    records = [r for r in all_records if r.get("action", "").startswith("task.")]
    for event in records:
        assert set(event) <= FIELDS
        assert all(isinstance(value, str) for value in event.values())
        assert event["timestamp"].endswith("+00:00")
        assert event["user"] == "unknown"
        assert event["job"] == "job-1"
        assert event["scope"] == "research cohort"
        assert "private-" not in json.dumps(event)
    return records


def _actions(path):
    return [r["action"] for r in _records(path)]


def _publication(path):
    events = [r for r in _records(path) if r["action"].startswith("task.publication.")]
    assert len(events) == 1
    return events[0]


@pytest.mark.parametrize("reference", ["private-opaque-reference", {"private-not-a-string": True}])
def test_success_uses_locally_written_receipt_without_copying_untrusted_reference(system, reference):
    runner, executor, ctx, task, _, path = system
    task.data.set_header(ReservedKey.AUDIT_EVENT_ID, reference)
    result = runner._process_task(task, ctx)
    assert result.get_return_code() == ReturnCode.OK
    assert executor.calls == 1
    assert runner.running_tasks == {}
    records = _records(path)
    assert _actions(path) == [
        "task.assignment.received",
        "task.execution.started",
        "task.execution.returned",
        "task.result.prepared",
    ]
    receipt = next(r for r in _all_records(path) if r.get("message") == "received task from server")
    assert task.data.get_header(ReservedKey.AUDIT_EVENT_ID) == reference
    assert receipt["reference"] == str(reference)
    assert [r["reference"] for r in records[:3]] == [receipt["event_id"]] * 3
    assert records[-1]["reference"] == result.get_header(ReservedKey.AUDIT_EVENT_ID)
    for event in records:
        assert 'task="train"' in event["message"]
        assert 'task_id="task-1"' in event["message"]
        assert 'attempt_id="attempt-1"' in event["message"]
        assert 'workflow="workflow-1"' in event["message"]
        assert 'site="site-1"' in event["message"]
    assert current_context() == TaskAuditContext()


def test_direct_assignment_does_not_link_the_supplied_reference(system):
    runner, executor, ctx, task, _, path = system
    request = task.data
    request.set_header(ReservedHeaderKey.TASK_NAME, task.name)
    request.set_header(ReservedHeaderKey.TASK_ID, task.task_id)
    request.set_header(ReservedKey.AUDIT_EVENT_ID, "private-direct-reference")

    result = runner._handle_do_task("do_task", request, ctx)

    assert result.get_return_code() == ReturnCode.OK
    assert executor.calls == 1
    received, started, returned, prepared = _records(path)
    assert [event["action"] for event in (received, started, returned)] == [
        "task.assignment.received",
        "task.execution.started",
        "task.execution.returned",
    ]
    assert all("reference" not in event for event in (received, started, returned))
    assert prepared["action"] == "task.result.prepared"
    assert prepared["reference"] == result.get_header(ReservedKey.AUDIT_EVENT_ID)
    receipt = next(r for r in _all_records(path) if r.get("message") == "received task from server")
    assert receipt["reference"] == "private-direct-reference"


@pytest.mark.parametrize("peer", [None, "bad-type", "other-job"])
def test_rejected_assignment_does_not_use_unverified_identity(system, peer):
    runner, executor, ctx, task, _, path = system
    if peer == "other-job":
        foreign = FLContext()
        foreign.set_prop(ReservedKey.RUN_NUM, "other-job", private=False, sticky=False)
        ctx.set_peer_context(foreign)
    else:
        ctx.set_peer_context(peer)
    task.data.set_header(ReservedKey.AUDIT_EVENT_ID, "forged-reference")
    result = runner._process_task(task, ctx)
    assert result.get_return_code() != ReturnCode.OK
    assert executor.calls == 0
    rejected = [r for r in _records(path) if r["action"] == "task.assignment.rejected"]
    assert len(rejected) == 1
    assert "reference" not in rejected[0]
    for key in ("task=", "task_id=", "attempt_id=", "workflow="):
        assert key not in rejected[0]["message"]
    assert "forged-reference" not in json.dumps(_records(path))


@pytest.mark.parametrize(
    "failure, expected_action, expected_code",
    [
        ("runtime", "task.execution.error", ReturnCode.EXECUTION_RESULT_ERROR),
        ("exception", "task.execution.error", ReturnCode.EXECUTION_EXCEPTION),
        ("unsafe", "task.execution.error", ReturnCode.UNSAFE_JOB),
        ("invalid_result", "task.execution.error", ReturnCode.EXECUTION_RESULT_ERROR),
        ("abort_return", "task.execution.aborted", ReturnCode.TASK_ABORTED),
        ("abort_error", "task.execution.aborted", ReturnCode.TASK_ABORTED),
    ],
)
def test_executor_outcomes_preserve_return_and_run_once(system, failure, expected_action, expected_code):
    runner, executor, ctx, task, _, path = system

    def execute(_name, _data, _ctx, abort):
        if failure.startswith("abort"):
            abort.trigger(True)
            if failure == "abort_error":
                raise RuntimeError("private-executor-error")
            return make_reply(ReturnCode.OK)
        if failure == "runtime":
            raise RuntimeError("private-executor-error")
        if failure == "exception":
            raise ValueError("private-executor-error")
        if failure == "unsafe":
            raise UnsafeJobError("private-executor-error")
        return object()

    executor.callback = execute
    result = runner._process_task(task, ctx)
    assert result.get_return_code() == expected_code
    assert executor.calls == 1
    assert _actions(path).count(expected_action) == 1
    assert "task.execution.returned" not in _actions(path)
    assert _actions(path)[-1] == "task.result.prepared"
    assert current_context() == TaskAuditContext()


def test_interruption_is_rethrown_without_error_result(system):
    runner, executor, ctx, task, _, path = system

    def interrupt(*_args):
        raise TaskExecutionInterrupted("private-worker-error")

    executor.callback = interrupt
    with pytest.raises(TaskExecutionInterrupted, match="private-worker-error"):
        runner._process_task(task, ctx)
    assert executor.calls == 1
    assert _actions(path) == ["task.assignment.received", "task.execution.started", "task.execution.interrupted"]
    assert current_context() == TaskAuditContext()


@pytest.mark.parametrize("stage", [FilterKey.IN, FilterKey.OUT])
@pytest.mark.parametrize("failure", ["exception", "abort", "invalid", "unsafe"])
def test_filter_result_boundaries_and_immutable_task_identity(system, monkeypatch, stage, failure):
    runner, executor, ctx, task, _, path = system

    def filters(_filter_name, data, _ctx, _filters, _task_name, direction, **kwargs):
        if direction != stage:
            return data
        if failure == "invalid":
            return object()
        if failure == "abort":
            kwargs["abort_signal"].trigger(True)
        if failure == "unsafe":
            raise UnsafeJobError("private-filter-error")
        raise ValueError("private-filter-error")

    monkeypatch.setattr(subject, "apply_filters", filters)
    result = runner._process_task(task, ctx)
    assert result.get_return_code() != ReturnCode.OK
    assert executor.calls == (stage == FilterKey.OUT)
    event = next(r for r in _records(path) if r["action"].startswith("task.filter."))
    assert f"stage={'data_filter' if stage == FilterKey.IN else 'result_filter'}" in event["message"]
    receipt = next(r for r in _all_records(path) if r.get("message") == "received task from server")
    assert event["reference"] == receipt["event_id"]
    assert 'task_id="task-1"' in event["message"]
    assert _actions(path)[-1] == "task.result.prepared"


def test_custom_application_code_is_not_exposed_as_audit_detail(system):
    runner, executor, ctx, task, _, path = system
    executor.callback = lambda *_args: make_reply("private-custom-code")
    result = runner._process_task(task, ctx)
    assert result.get_return_code() == "private-custom-code"
    records = _records(path)
    assert "status=custom_return_code" in records[-1]["message"]
    assert "status=custom_return_code" in records[-2]["message"]


def test_missing_application_code_is_not_recorded_as_success(system):
    runner, executor, ctx, task, _, path = system
    executor.callback = lambda *_args: Shareable()

    result = runner._process_task(task, ctx)

    assert result.get_return_code() == ReturnCode.OK  # Preserve the application default.
    assert result.get_return_code(default=None) is None
    returned, prepared = _records(path)[-2:]
    assert returned["action"] == "task.execution.returned"
    assert prepared["action"] == "task.result.prepared"
    assert all("status=unspecified" in event["message"] for event in (returned, prepared))


def test_filter_mutation_cannot_change_approved_snapshot(system, monkeypatch):
    runner, _, ctx, task, _, path = system

    def filters(_filter_name, data, fl_ctx, _filters, _name, _direction, **_kwargs):
        fl_ctx.set_prop(FLContextKey.TASK_ID, "private-forged-task", private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "changed-scope", private=True, sticky=False)
        return data

    monkeypatch.setattr(subject, "apply_filters", filters)
    result = runner._process_task(task, ctx)
    assert result.get_return_code() == ReturnCode.OK
    for event in _records(path):
        assert 'task_id="task-1"' in event["message"]
    assert "private-forged-task" not in json.dumps(_records(path))


def test_new_audit_write_failure_does_not_replay_or_replace_result(system, monkeypatch):
    runner, executor, ctx, task, writer, path = system
    original = writer.add_event

    def fail_new(user, action, *args, **kwargs):
        if action.startswith("task."):
            raise OSError("private-audit-write-error")
        return original(user, action, *args, **kwargs)

    monkeypatch.setattr(writer, "add_event", fail_new)
    result = runner._process_task(task, ctx)
    assert result.get_return_code() == ReturnCode.OK
    assert executor.calls == 1
    assert result.get_header(ReservedKey.AUDIT_EVENT_ID)
    assert _records(path) == []


def test_legacy_audit_failure_behavior_is_unchanged(system, monkeypatch):
    runner, executor, ctx, task, writer, path = system
    monkeypatch.setattr(writer, "add_job_event", Mock(side_effect=OSError("private-legacy-write-error")))
    result = runner._process_task(task, ctx)
    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert executor.calls == 0
    assert not result.get_header(ReservedKey.AUDIT_EVENT_ID)
    assert "task.execution.started" not in _actions(path)


def test_audit_snapshot_property_failure_does_not_change_execution(system, monkeypatch):
    runner, executor, ctx, task, _, path = system
    original = ctx.set_prop

    def set_prop(key, *args, **kwargs):
        if key == "_task_audit_context":
            raise RuntimeError("private-context-error")
        return original(key, *args, **kwargs)

    monkeypatch.setattr(ctx, "set_prop", set_prop)
    result = runner._process_task(task, ctx)
    assert result.get_return_code() == ReturnCode.OK
    assert executor.calls == 1
    assert _actions(path).count("task.execution.returned") == 1


def test_new_writes_happen_outside_task_lock(system, monkeypatch):
    runner, _, ctx, task, writer, path = system
    original = writer.add_event
    locked_writes = []

    def write(user, action, *args, **kwargs):
        if action.startswith("task."):
            if runner.task_lock.acquire(blocking=False):
                runner.task_lock.release()
            else:
                locked_writes.append(action)
        return original(user, action, *args, **kwargs)

    monkeypatch.setattr(writer, "add_event", write)
    runner._process_task(task, ctx)
    assert not locked_writes
    assert _actions(path)[-1] == "task.result.prepared"


@pytest.mark.parametrize(
    "mode", ["receipt", "legacy", "closed", "abort_before", "abort_after", "lost_ack", "exception"]
)
def test_publication_emits_one_terminal_fact_across_retries(system, monkeypatch, mode):
    runner, executor, ctx, task, _, path = system
    if mode == "legacy":
        task.attempt_id = None
        task.data.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, None)
    result = runner._process_task(task, ctx)
    reference = result.get_header(ReservedKey.AUDIT_EVENT_ID)
    check_count = 0
    send_count = 0

    def check(*_args):
        nonlocal check_count
        check_count += 1
        assert check_count < 5
        if mode == "closed":
            ctx.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.TASK_CLOSED, private=True, sticky=False)
            return subject._TASK_CHECK_RESULT_TASK_GONE
        return subject._TASK_CHECK_RESULT_OK

    def send(_result, fl_ctx, **_kwargs):
        nonlocal send_count
        send_count += 1
        if mode == "exception":
            raise OSError("private-transport-error")
        if mode == "abort_after":
            runner.run_abort_signal.trigger(True)
            return False
        if mode == "lost_ack" and send_count == 1:
            return False
        receipt = None if mode == "legacy" else TaskResultReceipt.RECEIVED
        fl_ctx.set_prop(FLContextKey.TASK_RESULT_RECEIPT, receipt, private=True, sticky=False)
        return True

    runner._check_task_once = check
    runner.engine.send_task_result = send
    monkeypatch.setattr(subject, "delete_msg_root", lambda _msg: None)
    if mode == "abort_before":
        runner.run_abort_signal.trigger(True)
    if mode == "exception":
        with pytest.raises(OSError, match="private-transport-error"):
            runner._send_task_result(result, task.task_id, ctx)
    else:
        sent = runner._send_task_result(result, task.task_id, ctx)
        assert sent is (mode in {"receipt", "legacy", "lost_ack"})
    event = _publication(path)
    expected = {
        "receipt": "received",
        "legacy": "acknowledged",
        "closed": "closed",
        "abort_before": "aborted",
        "abort_after": "aborted",
        "lost_ack": "received",
        "exception": "error",
    }[mode]
    assert event["action"] == f"task.publication.{expected}"
    assert event["reference"] == reference
    assert executor.calls == 1
    assert send_count == (2 if mode == "lost_ack" else 0 if mode in {"closed", "abort_before"} else 1)
    if mode.startswith("abort"):
        assert f"reason={'before_submission' if mode == 'abort_before' else 'after_submission'}" in event["message"]
    if mode == "exception":
        assert "stage=after_submission" in event["message"]


def test_publication_uses_snapshot_taken_before_send_hook(system, monkeypatch):
    runner, executor, ctx, task, _, path = system
    runner.engine.get_task_assignment.return_value = task
    runner._check_task_once = Mock(return_value=subject._TASK_CHECK_RESULT_OK)
    monkeypatch.setattr(subject, "delete_msg_root", lambda _msg: None)
    sent_ids = []

    def mutate_before_send(event, fl_ctx):
        if event == EventType.BEFORE_SEND_TASK_RESULT:
            reply = fl_ctx.get_prop(FLContextKey.TASK_RESULT)
            reply.set_header(ReservedKey.AUDIT_EVENT_ID, "private-wire-replacement")
            fl_ctx.set_prop(
                subject._TASK_AUDIT_RESULT_REFERENCE, "private-context-replacement", private=True, sticky=False
            )

    def send(reply, fl_ctx, **_kwargs):
        sent_ids.append(reply.get_header(ReservedKey.AUDIT_EVENT_ID))
        fl_ctx.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED, private=True, sticky=False)
        return True

    runner.fire_event = mutate_before_send
    runner.engine.send_task_result = send

    _, processed = runner.fetch_and_run_one_task(ctx)

    assert processed
    assert executor.calls == 1
    assert sent_ids == ["private-wire-replacement"]  # The legacy transport stays unchanged.
    prepared = next(r for r in _records(path) if r["action"] == "task.result.prepared")
    event = _publication(path)
    assert event["action"] == "task.publication.received"
    assert event["reference"] == prepared["reference"]
    written = next(r for r in _all_records(path) if r["event_id"] == event["reference"])
    assert written["message"] == "submit result OK"


def test_parent_closure_wins_when_stop_arrives_during_check(system):
    runner, executor, ctx, task, _, path = system
    result = runner._process_task(task, ctx)
    reply = make_reply(ReturnCode.TASK_UNKNOWN)
    reply.set_header(ReservedHeaderKey.TASK_ID, task.task_id)
    reply.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, task.attempt_id)
    reply.set_header(ReservedHeaderKey.WORKFLOW, task.workflow_id)
    reply.set_header(ReservedHeaderKey.TASK_RESULT_RECEIPT, TaskResultReceipt.TASK_CLOSED)

    def receive_closure(**_kwargs):
        runner.run_abort_signal.trigger(True)
        return {runner.parent_target: reply}

    runner.engine.send_aux_request.side_effect = receive_closure

    assert runner._send_task_result(result, task.task_id, ctx) is False

    runner.engine.send_aux_request.assert_called_once()
    runner.engine.send_task_result.assert_not_called()
    assert executor.calls == 1
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
    event = _publication(path)
    assert event["action"] == "task.publication.closed"
    assert "reason=task_closed" in event["message"]


def test_lost_acknowledgement_can_be_confirmed_without_new_upload(system):
    runner, executor, ctx, task, _, path = system
    result = runner._process_task(task, ctx)

    def check(*_args):
        ctx.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED, private=True, sticky=False)
        return subject._TASK_CHECK_RESULT_OK

    runner._check_task_once = check
    assert runner._send_task_result(result, task.task_id, ctx)
    runner.engine.send_task_result.assert_not_called()
    assert executor.calls == 1
    assert _publication(path)["action"] == "task.publication.received"


@pytest.mark.parametrize("control", [None, SpecialTaskName.TRY_AGAIN, SpecialTaskName.END_RUN])
def test_no_successful_control_polling_records(system, control):
    runner, executor, ctx, _, _, path = system
    runner.engine.get_task_assignment.return_value = (
        None if control is None else TaskAssignment(control, "", Shareable())
    )
    runner.fetch_and_run_one_task(ctx)
    assert executor.calls == 0
    assert _records(path) == []


def test_execution_start_is_visible_while_executor_is_still_running(system):
    runner, executor, ctx, task, _, path = system
    entered = threading.Event()
    release = threading.Event()
    results = []

    def execute(*_args):
        entered.set()
        release.wait()
        return make_reply(ReturnCode.OK)

    executor.callback = execute
    with ThreadGroup(release) as workers:
        workers.start(lambda: results.append(runner._process_task(task, ctx)))
        assert entered.wait(5.0), "executor did not enter"
        actions = _actions(path)
        assert actions == ["task.assignment.received", "task.execution.started"]
        assert "task.execution.returned" not in actions
        release.set()
    assert len(results) == 1
    assert results[0].get_return_code() == ReturnCode.OK
    assert executor.calls == 1
    assert _actions(path)[-2:] == ["task.execution.returned", "task.result.prepared"]


@pytest.mark.parametrize("stage", ["before_submission", "submission_unknown"])
def test_publication_exception_preserves_handoff_evidence_and_original_error(system, monkeypatch, stage):
    runner, executor, ctx, task, _, path = system
    result = runner._process_task(task, ctx)
    failure = OSError("private-publication-error")

    def fail(*_args, **_kwargs):
        raise failure

    if stage == "before_submission":
        runner._check_task_once = fail
    else:
        runner._check_task_once = lambda *_args: subject._TASK_CHECK_RESULT_OK
        runner.engine.send_task_result = Mock(side_effect=fail)
        original = ctx.get_prop

        def get_prop(key, *args, **kwargs):
            if key == FLContextKey.TASK_RESULT_SUBMISSION_ATTEMPTED:
                raise RuntimeError("private-audit-read-error")
            return original(key, *args, **kwargs)

        monkeypatch.setattr(ctx, "get_prop", get_prop)
    with pytest.raises(OSError) as caught:
        runner._send_task_result(result, task.task_id, ctx)
    assert caught.value is failure
    event = _publication(path)
    assert event["action"] == "task.publication.error"
    assert f"stage={stage}" in event["message"]
    assert executor.calls == 1
    assert runner.engine.send_task_result.call_count == (stage != "before_submission")
