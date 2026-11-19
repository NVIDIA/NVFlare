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

"""Direct auxiliary task observations, with controlled transport boundaries."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.impl import task_controller as subject
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.fuel.sec.task_audit import collect
from tests.unit_test.fl_context_helper import make_fl_context

FIELDS = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    engine = MagicMock()
    engine.all_clients = [Client(f"site-{index}", f"private-token-{index}") for index in range(1, 5)]
    engine.validate_targets.return_value = (engine.all_clients, [])
    engine.send_aux_request.return_value = {"site-1": make_reply(ReturnCode.OK)}
    ctx = make_fl_context(engine=engine, identity_name="local-site", run_num="job-1")
    ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "research cohort α", private=True, sticky=False)
    ctx.set_prop(FLContextKey.RUNNER, SimpleNamespace(task_data_filters={}, task_result_filters={}), private=True)
    controller = subject.TaskController()
    controller.start_controller(ctx)
    task = Task(name="train", data=Shareable({"payload": "private-task-data"}), timeout=10)
    yield controller, ctx, engine, task, writer, path
    writer.close()


def _records(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        assert set(record) <= FIELDS
        assert all(isinstance(value, str) for value in record.values())
        assert record["timestamp"].endswith("+00:00")
        assert record["user"] == "unknown"
        assert record["job"] == "job-1"
        assert record["scope"] == "research cohort α"
        assert "reference" not in record
        assert "task_id=" not in record.get("message", "")
        assert "attempt_id=" not in record.get("message", "")
        assert "private-" not in json.dumps(record)
    return records


def _actions(path, suffix):
    return [record for record in _records(path) if record["action"] == f"task.auxiliary.{suffix}"]


def test_actual_replies_and_callbacks_are_distinct_from_completion(setup):
    controller, ctx, engine, task, _, path = setup
    result_cb = Mock()
    done_cb = Mock()
    task.result_received_cb, task.task_done_cb = result_cb, done_cb
    reply = make_reply(ReturnCode.OK)
    reply["payload"] = "private-result-data"
    reply.set_header(ReservedHeaderKey.AUDIT_EVENT_ID, "private-forged-reference")
    reply.set_header(ReservedHeaderKey.TASK_ID, "private-forged-task")
    engine.send_aux_request.return_value = {"site-1": reply}

    replies = controller.broadcast_and_wait(task, ctx, targets=["site-1"])

    assert replies == {"site-1": reply}
    engine.send_aux_request.assert_called_once()
    result_cb.assert_called_once()
    done_cb.assert_called_once()
    records = _records(path)
    assert [r["action"] for r in records] == [
        "task.auxiliary.prepared",
        "task.auxiliary.dispatched",
        "task.auxiliary.reply_received",
        "task.auxiliary.callback_returned",
        "task.auxiliary.callback_returned",
    ]
    assert "status=OK" in records[2]["message"]
    assert "stage=result_callback" in records[3]["message"]
    assert "stage=task_done_callback" in records[4]["message"]
    assert task.completion_status is None  # No new audit changes the owner's completion decision.


def test_mixed_remote_results_missing_reply_and_unrequested_reply(setup):
    controller, ctx, engine, task, _, path = setup
    replies = {
        "site-1": make_reply(ReturnCode.OK),
        "site-2": make_reply(ReturnCode.EXECUTION_EXCEPTION),
        "site-3": make_reply("private-custom-return-code"),
        "private-unrequested-site": make_reply(ReturnCode.OK),
    }
    engine.send_aux_request.return_value = replies

    actual = controller.broadcast_and_wait(task, ctx, targets=[f"site-{i}" for i in range(1, 5)])

    assert actual["site-1"] is replies["site-1"]
    assert actual["site-2"].get_return_code() == ReturnCode.ERROR
    assert actual["site-3"].get_return_code() == ReturnCode.ERROR
    assert actual["site-4"] is None
    assert len(_actions(path, "prepared")) == len(_actions(path, "dispatched")) == 4
    reported = _actions(path, "reply_received")
    assert len(reported) == 3
    assert any("status=custom_return_code" in r["message"] for r in reported)
    assert any("status=EXECUTION_EXCEPTION" in r["message"] for r in reported)
    missing = _actions(path, "no_reply")
    assert len(missing) == 1 and 'site="site-4"' in missing[0]["message"]


@pytest.mark.parametrize("stage", ["task_data", "task_result"])
def test_filter_failure_keeps_existing_error_reply_and_does_not_call_result_callback(setup, monkeypatch, stage):
    controller, ctx, engine, task, _, path = setup
    task.result_received_cb = Mock()
    count = 0

    def filtering(*args, **kwargs):
        nonlocal count
        count += 1
        if count == (1 if stage == "task_data" else 2):
            raise RuntimeError("private-filter-exception")
        return args[1]

    monkeypatch.setattr(subject, "apply_filters", filtering)
    replies = controller.broadcast_and_wait(task, ctx, targets=["site-1"])

    expected = ReturnCode.TASK_DATA_FILTER_ERROR if stage == "task_data" else ReturnCode.TASK_RESULT_FILTER_ERROR
    assert replies["site-1"].get_return_code() == expected
    assert engine.send_aux_request.call_count == (stage == "task_result")
    task.result_received_cb.assert_not_called()
    events = _actions(path, "filter_error")
    assert len(events) == 1 and f"stage={stage}" in events[0]["message"]
    assert not _actions(path, "handling_error")


@pytest.mark.parametrize("stage", ["result_callback", "task_done_callback"])
def test_callback_error_is_recorded_at_owner_and_preserves_error_result(setup, stage):
    controller, ctx, engine, task, writer, path = setup
    callback = Mock(side_effect=RuntimeError("private-callback-exception"))
    if stage == "result_callback":
        task.result_received_cb = callback
    else:
        task.task_done_cb = callback
    lock_states = []
    original = writer.add_event

    def recording(*args, **kwargs):
        lock_states.append(task.cb_lock.locked())
        return original(*args, **kwargs)

    writer.add_event = recording
    replies = controller.broadcast_and_wait(task, ctx, targets=["site-1"])

    callback.assert_called_once()
    assert replies["site-1"].get_return_code() == ReturnCode.ERROR
    assert task.completion_status == TaskCompletionStatus.ERROR
    errors = _actions(path, "callback_error")
    assert len(errors) == 1 and f"stage={stage}" in errors[0]["message"]
    assert not _actions(path, "handling_error")
    assert lock_states and not any(lock_states)


def test_transport_exception_is_rethrown_without_claiming_a_reply_or_replaying(setup):
    controller, ctx, engine, task, _, path = setup
    failure = RuntimeError("private-transport-exception")
    engine.send_aux_request.side_effect = failure
    with pytest.raises(RuntimeError) as raised:
        controller.broadcast_and_wait(task, ctx, targets=["site-1"])
    assert raised.value is failure
    engine.send_aux_request.assert_called_once()
    assert len(_actions(path, "dispatched")) == 1
    assert len(_actions(path, "handling_error")) == 1
    assert not _actions(path, "reply_received")


@pytest.mark.parametrize("invalid", ["targets", "timeout"])
def test_setup_rejection_preserves_exception_and_omits_unapproved_target(setup, invalid):
    controller, ctx, engine, task, _, path = setup
    if invalid == "targets":
        engine.validate_targets.return_value = ([], ["private-invalid-target"])
    else:
        task.timeout = 0
    with pytest.raises(ValueError):
        controller.broadcast_and_wait(
            task, ctx, targets=["private-invalid-target"] if invalid == "targets" else ["site-1"]
        )
    engine.send_aux_request.assert_not_called()
    assert len(_actions(path, "rejected")) == 1
    assert not _actions(path, "dispatched")


def test_no_callback_function_does_not_invent_callback_outcome(setup):
    controller, ctx, engine, task, _, path = setup
    controller.broadcast_and_wait(task, ctx, targets=["site-1"])
    assert len(_records(path)) == 3
    assert not _actions(path, "callback_returned")
    assert not _actions(path, "callback_error")


def test_task_name_and_scope_snapshot_survive_filter_mutation(setup, monkeypatch):
    controller, ctx, _, task, _, path = setup
    task.name = 'train "quoted"\nα'
    original_name = task.name

    def mutate(*args, **kwargs):
        ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "private-mutated-scope", private=True)
        task.name = "private-mutated-task"
        return args[1]

    monkeypatch.setattr(subject, "apply_filters", mutate)
    controller.broadcast_and_wait(task, ctx, targets=["site-1"])
    events = _records(path)
    assert len(events) == 3
    for event in events:
        assert "task=" + json.dumps(original_name, ensure_ascii=True) in event["message"]


@pytest.mark.parametrize("failure", ["closed", "write", "prepare"])
def test_new_record_failures_do_not_replay_or_block_operation(setup, monkeypatch, failure):
    controller, ctx, engine, task, writer, path = setup
    callback = Mock()
    task.result_received_cb = callback
    if failure == "closed":
        writer.close()
    elif failure == "write":
        monkeypatch.setattr(writer, "add_event", Mock(side_effect=OSError("private-writer-exception")))
    else:
        monkeypatch.setattr(subject, "build_context", Mock(side_effect=RuntimeError("private-preparation-exception")))
    result = controller.broadcast_and_wait(task, ctx, targets=["site-1"])
    assert result["site-1"].get_return_code() == ReturnCode.OK
    callback.assert_called_once()
    engine.send_aux_request.assert_called_once()
    assert path.read_text() == ""


def test_callback_collector_defers_writer_until_outer_lock_and_callback_lock_release(setup):
    controller, ctx, engine, task, writer, path = setup
    outer_lock = threading.Lock()
    client = engine.all_clients[0]
    task.client_tasks.append(ClientTask(task, client))
    seen = []
    original = writer.add_event

    def recording(*args, **kwargs):
        seen.append((outer_lock.locked(), task.cb_lock.locked()))
        return original(*args, **kwargs)

    writer.add_event = recording
    callback = Mock()
    with collect():
        with outer_lock:
            assert controller._call_task_cb(callback, client, task, ctx) is False
            assert path.read_text() == ""
            assert not seen
    callback.assert_called_once()
    assert seen == [(False, False)]
    assert len(_actions(path, "callback_returned")) == 1


@pytest.mark.parametrize("method", ["send", "send_and_wait"])
@pytest.mark.parametrize("targets", [[], ["site-1", "site-2"], ["private-invalid-target"]])
def test_direct_send_target_rejections_before_broadcast_are_recorded_once(setup, method, targets):
    controller, ctx, engine, task, _, path = setup
    if targets == ["private-invalid-target"]:
        engine.validate_targets.return_value = ([], targets)
    with pytest.raises(ValueError):
        getattr(controller, method)(task, ctx, targets=targets)
    engine.send_aux_request.assert_not_called()
    errors = _actions(path, "rejected")
    assert len(errors) == 1
    assert "site=" not in errors[0]["message"]


def test_auditing_does_not_read_custom_reply_code_twice_or_change_callback_decision(setup):
    controller, ctx, engine, task, _, path = setup

    class ChangingReply(Shareable):
        reads = 0

        def get_return_code(self, default=None):
            self.reads += 1
            return ReturnCode.OK if self.reads == 1 else ReturnCode.ERROR

    reply = ChangingReply()
    engine.send_aux_request.return_value = {"site-1": reply}
    callback = Mock()
    task.result_received_cb = callback
    result = controller.broadcast_and_wait(task, ctx, targets=["site-1"])
    assert result["site-1"] is reply
    assert reply.reads == 1
    callback.assert_called_once()
    assert "status=OK" in _actions(path, "reply_received")[0]["message"]
