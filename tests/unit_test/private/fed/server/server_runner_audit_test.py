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
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode, TaskResultReceipt
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.fuel.sec import task_audit
from nvflare.fuel.sec.audit import Auditor, AuditService
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.server import server_runner as subject

FIELDS = {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}


@pytest.fixture
def audit_file(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    yield writer, path
    writer.close()


def _new_records(path):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events = [event for event in events if event.get("action", "").startswith(("task.", "workflow."))]
    for event in events:
        assert set(event) <= FIELDS
        assert all(isinstance(value, str) for value in event.values())
        assert event["user"] == "unknown"
        assert event["job"] == "job-1"
        assert event["timestamp"].endswith("+00:00")
        assert "private-" not in json.dumps(event)
    return events


def _legacy_result_receipt(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    (receipt,) = [record for record in records if record.get("message") == "received result from client 'site-1'"]
    return receipt


def _runner():
    runner = subject.ServerRunner.__new__(subject.ServerRunner)
    runner.job_id = "job-1"
    runner.status = "started"
    runner.wf_lock = threading.RLock()
    runner._processing_tasks_lock = threading.Lock()
    runner._processing_tasks = {}
    runner.abort_signal = Signal()
    runner.current_wf_index = 0
    for name in ("log_info", "log_debug", "log_warning", "log_error", "log_exception", "fire_event", "system_panic"):
        setattr(runner, name, Mock())
    runner._report_client_active = Mock()
    engine = Mock(spec=ServerEngineSpec)
    manager = FLContextManager(engine=engine, identity_name="server", job_id="job-1")

    def new_context():
        context = manager.new_context()
        context.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "research cohort", private=True, sticky=False)
        return context

    engine.new_context.side_effect = new_context
    runner.engine = engine
    runner.config = SimpleNamespace(
        task_request_interval=0.1, task_data_filters={}, task_result_filters={}, workflows=[]
    )
    runner.current_wf = SimpleNamespace(id="workflow-1", controller=Mock())
    context = new_context()
    context.set_peer_context(FLContextManager(identity_name="site-1", job_id="job-1").new_context())
    return runner, context


def _assignment():
    runner, context = _runner()
    client = Client("site-1", "private-client-token")
    task = Task("train", Shareable(), result_received_cb=Mock())
    task.props["___mgr"] = Mock()
    assigned = ClientTask(client, task)
    assigned.props["___job_id"] = "job-1"
    communicator = WFCommServer()
    communicator._client_task_map[assigned.id] = assigned
    communicator.controller = Mock()
    runner.current_wf.controller = SimpleNamespace(communicator=communicator)
    return runner, assigned, context


def _result(assigned):
    result = make_reply(ReturnCode.OK)
    result["weights"] = "private-model-payload"
    result.add_cookie(ReservedHeaderKey.TASK_ID, assigned.id)
    result.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    result.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow-1")
    result.set_header(ReservedHeaderKey.AUDIT_EVENT_ID, "client-result-event")
    return result


def _outgoing(runner, assigned):
    payload = Shareable({"weights": "private-model-payload"})
    payload.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    payload.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow-1")
    runner._try_to_get_task = Mock(return_value=("train", assigned.id, payload))
    return payload


def test_assignment_prepared_keeps_legacy_header_and_pre_filter_identity(audit_file, monkeypatch):
    writer, path = audit_file
    runner, assigned, context = _assignment()
    _outgoing(runner, assigned)
    original_add = writer.add_event

    def write(*args, **kwargs):
        if kwargs.get("action", "").startswith("task."):
            assert not runner.wf_lock._is_owned()
            assert not runner._processing_tasks_lock.locked()
        return original_add(*args, **kwargs)

    monkeypatch.setattr(writer, "add_event", write)

    def replacement(*args, **kwargs):
        context.set_prop(FLContextKey.TASK_ID, "private-filter-id", private=True, sticky=False)
        return Shareable({"data": "private-replacement-payload"})

    monkeypatch.setattr(subject, "apply_filters", replacement)
    name, identifier, payload = runner.process_task_request(assigned.client, context)
    assert name == "train" and identifier == assigned.id
    assert payload.get_cookie(ReservedHeaderKey.WORKFLOW) == "workflow-1"
    (event,) = _new_records(path)
    assert event["action"] == "task.assignment.prepared"
    assert event["reference"] == payload.get_header(ReservedHeaderKey.AUDIT_EVENT_ID)
    assert f"task_id={json.dumps(assigned.id)}" in event["message"]
    assert "research cohort" == event["scope"]
    assert task_audit.current_context() == task_audit.TaskAuditContext()


def test_assignment_filter_failure_preserves_retry_and_tracker_cleanup(audit_file, monkeypatch):
    _, path = audit_file
    runner, assigned, context = _assignment()
    _outgoing(runner, assigned)
    runner.current_wf.controller.communicator.handle_exception = Mock()
    monkeypatch.setattr(subject, "apply_filters", Mock(side_effect=ValueError("private-filter-error")))
    name, identifier, _ = runner.process_task_request(assigned.client, context)
    assert name == SpecialTaskName.TRY_AGAIN and identifier == ""
    assert runner._processing_tasks == {}
    (event,) = _new_records(path)
    assert event["action"] == "task.assignment.error"
    assert "reason=data_filter_error" in event["message"]
    assert "reference" not in event


@pytest.mark.parametrize("state", ["init", "done", "empty"])
def test_routine_assignment_polling_does_not_add_records(audit_file, state):
    _, path = audit_file
    runner, context = _runner()
    if state == "empty":
        runner._try_to_get_task = Mock(return_value=("", "", None))
    else:
        runner.status = state
    runner.process_task_request(Client("site-1", "private-token"), context)
    assert _new_records(path) == []


@pytest.mark.parametrize("case", ["closed", "type", "peer", "job", "identity", "attempt", "workflow"])
def test_rejected_results_use_local_context_only_once(audit_file, case):
    _, path = audit_file
    runner, assigned, context = _assignment()
    result = _result(assigned)
    result.set_header(ReservedHeaderKey.AUDIT_EVENT_ID, "private-forged-ref")
    if case == "closed":
        runner.status = "done"
    elif case == "type":
        result = {"private-payload": "private-model"}
    elif case == "peer":
        context.set_peer_context(None)
    elif case == "job":
        context.set_peer_context(FLContextManager(identity_name="site-1", job_id="private-foreign-job").new_context())
    elif case == "identity":
        context.set_peer_context(FLContextManager(identity_name="private-forged-actor", job_id="job-1").new_context())
    elif case == "attempt":
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "")
    else:
        result.add_cookie(ReservedHeaderKey.WORKFLOW, "private-foreign-workflow")
    runner.process_submission(assigned.client, "train", assigned.id, result, context)
    events = _new_records(path)
    assert len(events) == 1
    event = events[0]
    assert event["action"] == "task.result.rejected"
    assert "reference" not in event
    assert "task_id=" not in event["message"] and "site=" not in event["message"]
    assigned.task.result_received_cb.assert_not_called()
    assert context.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED


@pytest.mark.parametrize("failure", [None, "filter", "callback", "communicator"])
def test_result_handling_outcomes_preserve_receipt_and_write_outside_gate(audit_file, monkeypatch, failure):
    writer, path = audit_file
    runner, assigned, context = _assignment()
    result = _result(assigned)
    if failure == "callback":
        assigned.task.result_received_cb.side_effect = RuntimeError("private-callback-error")
    if failure == "filter":
        monkeypatch.setattr(subject, "apply_filters", Mock(side_effect=RuntimeError("private-filter-error")))
    if failure == "communicator":
        runner.current_wf.controller.communicator.process_submission = Mock(side_effect=RuntimeError("private-error"))
    original = writer.add_event

    def write(*args, **kwargs):
        if kwargs.get("action", "").startswith("task."):
            assert not runner.wf_lock._is_owned()
            assert not assigned.task.cb_lock.locked()
        return original(*args, **kwargs)

    monkeypatch.setattr(writer, "add_event", write)
    runner.process_submission(assigned.client, "train", assigned.id, result, context)
    events = _new_records(path)
    action = "task.result.handling_error" if failure == "communicator" else "task.result.handling_returned"
    observed = [event for event in events if event["action"] == action]
    assert len(observed) == 1
    legacy_receipt = _legacy_result_receipt(path)
    assert legacy_receipt["reference"] == "client-result-event"
    assert observed[0]["reference"] == legacy_receipt["event_id"]
    assert ("task.result.filter_error" in [e["action"] for e in events]) == (failure == "filter")
    assert context.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    if failure == "callback":
        assert assigned.task.completion_status == TaskCompletionStatus.ERROR
    assert task_audit.current_context() == task_audit.TaskAuditContext()


@pytest.mark.parametrize(
    "peer_reference", ["private-forged-result-reference", {"value": "private-forged-result-reference"}]
)
def test_matched_result_uses_local_receipt_without_copying_peer_reference(audit_file, peer_reference):
    _, path = audit_file
    runner, assigned, context = _assignment()
    result = _result(assigned)
    result.set_header(ReservedHeaderKey.AUDIT_EVENT_ID, peer_reference)

    runner.process_submission(assigned.client, "train", assigned.id, result, context)

    assert context.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assigned.task.result_received_cb.assert_called_once()
    assert result.get_header(ReservedHeaderKey.AUDIT_EVENT_ID) == peer_reference
    legacy_receipt = _legacy_result_receipt(path)
    assert legacy_receipt["reference"] == str(peer_reference)
    events = _new_records(path)
    assert {event["action"] for event in events} == {
        "task.result.received",
        "task.result_manager.returned",
        "task.result_callback.returned",
        "task.result.handling_returned",
    }
    assert all(event["reference"] == legacy_receipt["event_id"] for event in events)
    assert all(f"task_id={json.dumps(assigned.id)}" in event["message"] for event in events)


def test_matched_result_keeps_approved_attribution_without_local_receipt(audit_file, monkeypatch):
    _, path = audit_file
    runner, assigned, context = _assignment()
    monkeypatch.setattr(subject, "add_job_audit_event", Mock(return_value=""))

    runner.process_submission(assigned.client, "train", assigned.id, _result(assigned), context)

    assigned.task.result_received_cb.assert_called_once()
    events = _new_records(path)
    assert "task.result.handling_returned" in {event["action"] for event in events}
    assert all("reference" not in event for event in events)
    assert all(f"task_id={json.dumps(assigned.id)}" in event["message"] for event in events)


@pytest.mark.parametrize("task_name,task_id", [("private-unknown-task", "private-unknown-id"), ("", "")])
def test_unfenced_unknown_result_does_not_gain_unverified_reference(audit_file, task_name, task_id):
    _, path = audit_file
    runner, assigned, context = _assignment()
    result = Shareable()
    result.set_header(ReservedHeaderKey.AUDIT_EVENT_ID, "private-unverified-reference")
    runner.process_submission(assigned.client, task_name, task_id, result, context)
    events = _new_records(path)
    assert "task.result.handling_returned" in {event["action"] for event in events}
    assert all("reference" not in event and "task_id=" not in event.get("message", "") for event in events)
    runner.current_wf.controller.communicator.controller.process_result_of_unknown_task.assert_called_once()


@pytest.mark.parametrize("failure", [None, "initialize", "control", "stop", "finalize", "abort"])
def test_workflow_lifecycle_separates_return_abort_and_finalization(audit_file, failure):
    writer, path = audit_file
    runner, _ = _runner()
    workflow = runner.current_wf
    runner.config.workflows = [workflow]
    runner.current_wf = None
    if failure in {"initialize", "control", "stop"}:
        name = {"initialize": "initialize", "control": "control_flow", "stop": "stop_controller"}[failure]
        getattr(workflow.controller, name).side_effect = RuntimeError("private-workflow-error")
    elif failure == "finalize":
        workflow.controller.communicator.finalize_run.side_effect = RuntimeError("private-finalize-error")
    elif failure == "abort":
        workflow.controller.control_flow.side_effect = lambda signal, context: signal.trigger(True)
    original = writer.add_event

    def write(*args, **kwargs):
        assert not runner.wf_lock._is_owned()
        return original(*args, **kwargs)

    writer.add_event = write
    runner._execute_run()
    events = _new_records(path)
    actions = [event["action"] for event in events]
    assert ("workflow.execution.initialized" in actions) == (failure != "initialize")
    assert ("workflow.control.returned" in actions) == (failure not in {"initialize", "control"})
    assert ("workflow.execution.error" in actions) == (failure in {"initialize", "control"})
    assert ("workflow.control.abort_observed" in actions) == (failure == "abort")
    assert ("workflow.finalization.error" in actions) == (failure in {"stop", "finalize"})
    assert ("workflow.finalization.returned" in actions) == (failure not in {"stop", "finalize"})
    assert runner.current_wf is None
    assert EventType.END_WORKFLOW in [call.args[0] for call in runner.fire_event.call_args_list]
    if failure != "initialize":
        origin = next(e for e in events if e["action"] == "workflow.execution.initialized")["event_id"]
        assert all(e.get("reference") == origin for e in events if e["action"] != "workflow.execution.initialized")
    else:
        assert all("reference" not in e for e in events)


@pytest.mark.parametrize("mutation_point", ["initialize", "start_workflow"])
def test_workflow_audit_scope_survives_initialization_callbacks(audit_file, mutation_point):
    _, path = audit_file
    runner, _ = _runner()
    workflow = runner.current_wf
    runner.config.workflows = [workflow]
    runner.current_wf = None

    def change_scope(fl_ctx):
        fl_ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "private-callback-scope", private=True, sticky=False)

    def change_scope_on_start(event, fl_ctx):
        if event == EventType.START_WORKFLOW:
            change_scope(fl_ctx)

    if mutation_point == "initialize":
        workflow.controller.initialize.side_effect = change_scope
    else:
        runner.fire_event.side_effect = change_scope_on_start

    runner._execute_run()

    initialized, controlled, finalized = _new_records(path)
    assert [event["action"] for event in (initialized, controlled, finalized)] == [
        "workflow.execution.initialized",
        "workflow.control.returned",
        "workflow.finalization.returned",
    ]
    assert all(event["scope"] == "research cohort" for event in (initialized, controlled, finalized))
    assert controlled["reference"] == finalized["reference"] == initialized["event_id"]


def test_new_recording_failure_does_not_replay_assignment(audit_file, monkeypatch):
    writer, path = audit_file
    runner, assigned, context = _assignment()
    _outgoing(runner, assigned)
    original = writer.add_event

    def write(*args, **kwargs):
        if kwargs.get("action", "").startswith("task."):
            assert not runner.wf_lock._is_owned()
            raise OSError("private-audit-path")
        return original(*args, **kwargs)

    monkeypatch.setattr(writer, "add_event", write)
    name, identifier, payload = runner.process_task_request(assigned.client, context)
    assert name == "train" and identifier == assigned.id
    assert payload.get_header(ReservedHeaderKey.AUDIT_EVENT_ID)
    runner._try_to_get_task.assert_called_once()
    assert _new_records(path) == []


def test_audit_context_preparation_failure_does_not_block_result(audit_file, monkeypatch):
    _, path = audit_file
    runner, assigned, context = _assignment()
    result = _result(assigned)
    original = context.get_prop

    def get_prop(key, *args, **kwargs):
        if key == FLContextKey.EFFECTIVE_JOB_SCOPE_NAME:
            raise ValueError("private-scope-preparation")
        return original(key, *args, **kwargs)

    monkeypatch.setattr(context, "get_prop", get_prop)
    monkeypatch.setattr(subject, "add_job_audit_event", Mock(return_value="legacy-event"))
    runner.process_submission(assigned.client, "train", assigned.id, result, context)
    assigned.task.result_received_cb.assert_called_once()
    assert context.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assert _new_records(path) == []


def test_activity_prevalidation_failure_is_observed_without_changing_propagation(audit_file):
    _, path = audit_file
    runner, assigned, context = _assignment()
    del runner._report_client_active
    context.set_peer_context(None)
    with pytest.raises(AssertionError):
        runner.process_submission(
            assigned.client, "private-untrusted-task", "private-untrusted-id", _result(assigned), context
        )
    (event,) = _new_records(path)
    assert event["action"] == "task.result.handling_error"
    assert "stage=admission_or_cleanup" in event["message"]
    assert "reference" not in event and "task_id=" not in event["message"]
    assigned.task.result_received_cb.assert_not_called()


def test_workflow_context_entry_failure_keeps_local_job_and_runs_finalization(audit_file):
    _, path = audit_file
    runner, _ = _runner()
    workflow = runner.current_wf
    runner.config.workflows = [workflow]
    original = runner.engine.new_context.side_effect
    calls = []

    def new_context():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("private-context-error")
        return original()

    runner.engine.new_context.side_effect = new_context
    runner._execute_run()
    events = _new_records(path)
    assert [event["action"] for event in events] == ["workflow.execution.error", "workflow.finalization.returned"]
    assert "scope" not in events[0]
    assert "stage=initialize" in events[0]["message"]
    workflow.controller.control_flow.assert_not_called()
    workflow.controller.communicator.finalize_run.assert_called_once()


@pytest.mark.parametrize("filter_error", [False, True])
def test_result_filter_cannot_replace_approved_scope_or_workflow(audit_file, monkeypatch, filter_error):
    _, path = audit_file
    runner, assigned, context = _assignment()
    communicator = runner.current_wf.controller.communicator
    # Exercise the real initialization snapshot while controlling its worker
    # boundary: this test verifies policy within one interpreter.
    monkeypatch.setattr(communicator._task_monitor, "start", Mock())
    context.set_prop(FLContextKey.WORKFLOW, "workflow-1", private=True, sticky=True)
    communicator.initialize_run(context)
    result = _result(assigned)

    def filter_result(filter_name, value, fl_ctx, *args, **kwargs):
        fl_ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "private-replacement-scope", private=True, sticky=False)
        if filter_error:
            raise RuntimeError("private-filter-exception")
        return value

    monkeypatch.setattr(subject, "apply_filters", filter_result)
    runner.process_submission(assigned.client, "train", assigned.id, result, context)
    events = _new_records(path)
    owners = {
        "task.result.received",
        "task.result_manager.returned",
        "task.result_callback.returned",
        "task.result.handling_returned",
    }
    observed = [event for event in events if event["action"] in owners]
    assert {event["action"] for event in observed} == owners
    assert all(event["scope"] == "research cohort" for event in events)
    assert all('workflow="workflow-1"' in event.get("message", "") for event in observed)
    assert all(event["reference"] == _legacy_result_receipt(path)["event_id"] for event in observed)
    assert context.get_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME) == "private-replacement-scope"
    assert context.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assigned.task.result_received_cb.assert_called_once()
    assigned.task.props["___mgr"].check_task_result.assert_called_once()
    assert assigned.result.get_return_code() == (ReturnCode.TASK_RESULT_FILTER_ERROR if filter_error else ReturnCode.OK)
    assert ("task.result.filter_error" in [event["action"] for event in events]) == filter_error
