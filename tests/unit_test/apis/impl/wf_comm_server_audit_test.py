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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, TaskResultReceipt
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.impl import wf_comm_server as subject
from nvflare.apis.impl.task_manager import TaskManager
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.fuel.sec.task_audit import build_context, collect, current_context, use_context

FIELDS = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}


@pytest.fixture
def audit_file(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    locks = []

    class CheckingAuditor(Auditor):
        def add_event(self, *args, **kwargs):
            assert all(not lock.locked() for lock in locks), "audit I/O held an operation lock"
            return super().add_event(*args, **kwargs)

    writer = CheckingAuditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    yield path, writer, locks
    writer.close()


def _records(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for event in records:
        assert set(event) <= FIELDS
        assert all(isinstance(value, str) for value in event.values())
        assert event["timestamp"].endswith("+00:00")
        assert event["user"] == "unknown"
    assert "private-" not in path.read_text()
    return records


def _assigned(audit_file):
    _, _, locks = audit_file
    engine = Mock()
    ctx = FLContextManager(engine=engine, identity_name="server", job_id="job-1").new_context()
    ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "research cohort", private=True, sticky=False)
    ctx.set_peer_context(FLContextManager(identity_name="site-1", job_id="job-1").new_context())
    engine.new_context.return_value = ctx
    wf = WFCommServer()
    wf._engine = engine
    wf.controller = Mock()
    wf._task_audit_base = build_context(ctx, workflow_id="wf-1")
    task = Task("train", Shareable({"payload": "private-model-content"}), result_received_cb=Mock())
    task.props["___mgr"] = Mock(spec=TaskManager)
    task.props["___mgr"].check_task_exit.return_value = (False, None)
    assigned = ClientTask(Client("site-1", "private-session-token"), task)
    assigned.props["___job_id"] = "job-1"
    wf._client_task_map[assigned.id] = assigned
    locks.extend([wf._controller_lock, wf._task_lock, task.cb_lock])
    result = Shareable({"payload": "private-result-content"})
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    result.set_header(ReservedHeaderKey.AUDIT_EVENT_ID, "private-unchecked-reference")
    return wf, assigned, ctx, result


def _request_context(ctx, assigned):
    return build_context(
        ctx,
        workflow_id="wf-1",
        task_name=assigned.task.name,
        task_id=assigned.id,
        attempt_id=assigned.attempt_id,
        site=assigned.client.name,
        reference="verified-client-result",
    )


def test_complete_receipt_and_callback_are_separate_once_only_facts(audit_file):
    path, _, locks = audit_file
    wf, assigned, ctx, result = _assigned(audit_file)
    outer_lock = threading.Lock()
    locks.append(outer_lock)
    with use_context(_request_context(ctx, assigned)), collect():
        with outer_lock:
            wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
            assert path.read_text() == ""
    wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
    events = _records(path)
    assert [event["action"] for event in events] == [
        "task.result.received",
        "task.result_manager.returned",
        "task.result_callback.returned",
        "task.result.duplicate",
    ]
    for event in events[:3]:
        assert event["reference"] == "verified-client-result"
        assert event["job"] == "job-1" and event["scope"] == "research cohort"
        assert f'task_id="{assigned.id}"' in event["message"]
        assert f'attempt_id="{assigned.attempt_id}"' in event["message"]
    assigned.task.result_received_cb.assert_called_once()
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED


@pytest.mark.parametrize("failure", ["manager", "callback"])
def test_processing_error_does_not_erase_receipt_or_replay_work(audit_file, failure):
    path, _, _ = audit_file
    wf, assigned, ctx, result = _assigned(audit_file)
    callback = assigned.task.result_received_cb
    manager = assigned.task.props["___mgr"]
    error = RuntimeError("private-application-error")
    if failure == "manager":
        manager.check_task_result.side_effect = error
        with pytest.raises(RuntimeError) as raised:
            wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
        assert raised.value is error
    else:
        callback.side_effect = error
        wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
        assert assigned.task.completion_status == TaskCompletionStatus.ERROR
    wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
    actions = [event["action"] for event in _records(path)]
    assert actions[0] == "task.result.received"
    assert f"task.result_{failure}.error" in actions
    assert actions[-1] == "task.result.duplicate"
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    manager.check_task_result.assert_called_once()
    assert callback.call_count == (failure == "callback")


@pytest.mark.parametrize("mismatch", ["client", "task", "attempt", "unknown", "malformed"])
def test_rejected_assignment_never_records_unapproved_identity_or_reference(audit_file, mismatch):
    path, _, _ = audit_file
    wf, assigned, ctx, result = _assigned(audit_file)
    client, name, task_id = assigned.client, "train", assigned.id
    if mismatch == "client":
        client = Client("private-forged-client", "private-token")
    elif mismatch == "task":
        name = "private-forged-task"
    elif mismatch == "attempt":
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "private-forged-attempt")
    elif mismatch == "unknown":
        task_id = "private-forged-task-id"
    else:
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, 7)
    wf.process_submission(client, name, task_id, result, ctx)
    events = _records(path)
    assert len(events) == 1
    event = events[0]
    assert event["action"] == "task.result.rejected"
    assert "reference" not in event
    assert "task_id=" not in event["message"] and "site=" not in event["message"]
    assigned.task.result_received_cb.assert_not_called()
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED


@pytest.mark.parametrize("failure", [False, True])
def test_first_late_result_is_received_once_even_if_late_hook_raises(audit_file, failure):
    path, _, _ = audit_file
    wf, assigned, ctx, result = _assigned(audit_file)
    assigned.task.completion_status = TaskCompletionStatus.TIMEOUT
    wf._remember_completed_client_task(assigned)
    wf._client_task_map.pop(assigned.id)
    if failure:
        wf.controller.process_result_of_unknown_task.side_effect = RuntimeError("private-late-error")
        with pytest.raises(RuntimeError):
            wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
    else:
        wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
    wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
    actions = [event["action"] for event in _records(path)]
    assert actions == [
        "task.result.received",
        "task.late_result.error" if failure else "task.late_result.returned",
        "task.result.duplicate",
    ]
    wf.controller.process_result_of_unknown_task.assert_called_once()
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED


def test_unfenced_unknown_hook_omits_supplied_task_identity(audit_file):
    path, _, _ = audit_file
    wf, assigned, ctx, _ = _assigned(audit_file)
    wf.process_submission(assigned.client, "private-task-name", "private-task-id", Shareable(), ctx)
    events = _records(path)
    assert [event["action"] for event in events] == ["task.result.received", "task.unknown_result.returned"]
    assert "reason=legacy_unknown" in events[0]["message"]
    assert all("task_id=" not in event.get("message", "") and "reference" not in event for event in events)
    wf.controller.process_result_of_unknown_task.assert_called_once()


@pytest.mark.parametrize("failure", [False, True])
def test_monitor_retirement_fact_precedes_later_done_callback_error(audit_file, failure):
    path, _, _ = audit_file
    wf, assigned, _, _ = _assigned(audit_file)
    task = assigned.task
    task.completion_status = TaskCompletionStatus.OK
    task.task_done_cb = Mock(side_effect=RuntimeError("private-done-error") if failure else None)
    task.is_standing = True
    wf._tasks = [task]
    wf.check_tasks()
    events = _records(path)
    assert [event["action"] for event in events] == [
        "task.retired",
        "task.done_callback.error" if failure else "task.done_callback.returned",
    ]
    assert "status=ok" in events[0]["message"]
    assert events[0]["job"] == "job-1" and events[0]["scope"] == "research cohort"
    assert task.completion_status == (TaskCompletionStatus.ERROR if failure else TaskCompletionStatus.OK)
    assert not wf._tasks and not task.is_standing
    task.task_done_cb.assert_called_once()


def test_actual_timeout_removal_is_recorded_without_callback_success_claim(audit_file, monkeypatch):
    path, _, _ = audit_file
    wf, assigned, _, _ = _assigned(audit_file)
    task = assigned.task
    task.timeout, task.schedule_time = 1, 0.0
    wf._tasks = [task]
    monkeypatch.setattr(subject, "time", SimpleNamespace(time=lambda: 2.0))
    wf.check_tasks()
    (event,) = _records(path)
    assert event["action"] == "task.retired" and "status=timeout" in event["message"]
    assert task.completion_status == TaskCompletionStatus.TIMEOUT


def test_finalization_records_cancellation_before_releasing_task_resources(audit_file, monkeypatch):
    path, _, _ = audit_file
    wf, assigned, ctx, _ = _assigned(audit_file)
    task = assigned.task
    wf._tasks = [task]
    monkeypatch.setattr(wf, "_cleanup_inflight_tensor_downloads", lambda _: None)
    wf.finalize_run(ctx)
    (event,) = _records(path)
    assert event["action"] == "task.retired"
    assert "status=cancelled" in event["message"]
    assert 'task="train"' in event["message"]
    assert event["job"] == "job-1"
    assert task.data is None and task.client_tasks == [] and not wf._tasks


def test_cleanup_failure_preserves_all_already_removed_task_facts(audit_file, monkeypatch):
    path, _, _ = audit_file
    wf, assigned, ctx, _ = _assigned(audit_file)
    second = Task("evaluate", Shareable())
    wf._tasks = [assigned.task, second]
    monkeypatch.setattr(wf, "_release_task_resources", Mock(side_effect=RuntimeError("private-cleanup-error")))
    with pytest.raises(RuntimeError):
        wf._clear_standing_tasks(fl_ctx=ctx)
    events = _records(path)
    assert [event["action"] for event in events] == ["task.retired", "task.cleanup.error", "task.retired"]
    assert "status=cancelled" in events[0]["message"]
    assert "status=unknown" in events[2]["message"]
    assert 'task="evaluate"' in events[2]["message"]
    assert second.completion_status is None
    assert not wf._tasks
    wf._release_task_resources.assert_called_once()


def test_new_audit_failure_does_not_change_receipt_or_callback_count(audit_file, monkeypatch):
    _, writer, _ = audit_file
    wf, assigned, ctx, result = _assigned(audit_file)
    monkeypatch.setattr(writer, "add_event", Mock(side_effect=OSError("private-writer-error")))
    wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assigned.task.result_received_cb.assert_called_once()
    assert assigned.result is result


def test_matched_request_refines_only_its_scoped_approved_context(audit_file):
    path, _, _ = audit_file
    wf, assigned, ctx, result = _assigned(audit_file)
    with use_context(build_context(ctx, workflow_id="wf-1")):
        assert wf.check_submission(assigned.client, "train", assigned.id, result, ctx)
        approved = current_context()
        assert approved.task_id == assigned.id and approved.task_name == "train"
        assert approved.attempt_id == assigned.attempt_id and approved.site == "site-1"
        assert approved.reference == ""
    assert current_context().job_id == ""
    assert _records(path) == []


def test_done_callback_cannot_replace_approved_task_name_in_later_record(audit_file):
    path, _, _ = audit_file
    wf, assigned, _, _ = _assigned(audit_file)
    task = assigned.task
    task.completion_status = TaskCompletionStatus.OK
    wf._tasks = [task]

    def callback(**kwargs):
        kwargs["task"].name = "private-mutated-task-name"

    task.task_done_cb = callback
    wf.check_tasks()
    events = _records(path)
    assert [event["action"] for event in events] == ["task.retired", "task.done_callback.returned"]
    assert all('task="train"' in event["message"] for event in events)


@pytest.mark.parametrize("approved_source", ["caller", "communicator"])
@pytest.mark.parametrize("mutate_job", [False, True])
def test_result_owner_keeps_approved_job_scope_and_workflow_after_context_mutation(
    audit_file, approved_source, mutate_job
):
    path, _, _ = audit_file
    wf, assigned, ctx, result = _assigned(audit_file)
    approved = build_context(
        ctx,
        task_name=assigned.task.name,
        task_id=assigned.id,
        attempt_id=assigned.attempt_id,
        site=assigned.client.name,
        reference="verified-client-result",
        # Empty caller workflow must retain the communicator's initialized name.
        workflow_id="",
    )
    ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "private-replacement-scope", private=True, sticky=False)
    if mutate_job:
        ctx.set_prop(ReservedKey.RUN_NUM, "private-replacement-job", private=False, sticky=True)
        assert ctx.get_job_id() == "private-replacement-job"
    with use_context(approved if approved_source == "caller" else build_context()):
        wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
    events = _records(path)
    assert [event["action"] for event in events] == [
        "task.result.received",
        "task.result_manager.returned",
        "task.result_callback.returned",
    ]
    for event in events:
        assert event["job"] == "job-1" and event["scope"] == "research cohort"
        assert 'workflow="wf-1"' in event["message"]
        assert f'task_id="{assigned.id}"' in event["message"]
        if approved_source == "caller":
            assert event["reference"] == "verified-client-result"
        else:
            assert "reference" not in event
    assigned.task.props["___mgr"].check_task_result.assert_called_once()
    assigned.task.result_received_cb.assert_called_once()
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED


@pytest.mark.parametrize("later_callback_error", [False, True])
def test_done_callbacks_keep_each_retirement_snapshot_after_earlier_callback_mutates_context(
    audit_file, later_callback_error
):
    path, _, _ = audit_file
    wf, assigned, ctx, _ = _assigned(audit_file)
    first = assigned.task
    second = Task("evaluate", Shareable())
    first.completion_status = second.completion_status = TaskCompletionStatus.OK
    wf._tasks = [first, second]

    def first_done(**kwargs):
        kwargs["fl_ctx"].set_prop(
            FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "private-replacement-scope", private=True, sticky=False
        )
        second.name = "private-replacement-next-task-name"

    first.task_done_cb = Mock(side_effect=first_done)
    second.task_done_cb = Mock(
        side_effect=RuntimeError("private-later-callback-error") if later_callback_error else None
    )
    wf.check_tasks()
    events = _records(path)
    assert [event["action"] for event in events] == [
        "task.retired",
        "task.retired",
        "task.done_callback.returned",
        "task.done_callback.error" if later_callback_error else "task.done_callback.returned",
    ]
    assert [
        "task=" + json.dumps(name) in event["message"]
        for name, event in zip(["train", "evaluate", "train", "evaluate"], events)
    ] == [True] * 4
    assert all(event["job"] == "job-1" and event["scope"] == "research cohort" for event in events)
    assert all('workflow="wf-1"' in event["message"] for event in events)
    first.task_done_cb.assert_called_once()
    second.task_done_cb.assert_called_once()
    assert second.completion_status == (TaskCompletionStatus.ERROR if later_callback_error else TaskCompletionStatus.OK)
    assert not wf._tasks
