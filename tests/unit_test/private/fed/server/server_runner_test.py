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

import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode, ServerCommandKey
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.private.fed.server.server_commands import GetTaskCommand, SubmitUpdateCommand
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_runner import ServerRunner


def _make_engine():
    args = SimpleNamespace(set=[])
    engine = ServerEngine(
        server=MagicMock(),
        args=args,
        client_manager=MagicMock(),
        snapshot_persistor=MagicMock(),
    )
    engine.logger = MagicMock()
    return engine


class TestServerEngineGetCell:
    def test_returns_parent_cell_even_when_run_manager_cell_present(self):
        engine = _make_engine()
        parent_cell = MagicMock(name="parent_cell")
        run_cell = MagicMock(name="run_cell")
        engine.cell = parent_cell
        engine.run_manager = SimpleNamespace(cell=run_cell)

        assert engine.get_cell() is parent_cell

    def test_falls_back_to_parent_cell_when_run_cell_missing(self):
        engine = _make_engine()
        parent_cell = MagicMock(name="parent_cell")
        engine.cell = parent_cell
        engine.run_manager = SimpleNamespace(cell=None)

        assert engine.get_cell() is parent_cell

    def test_returns_none_when_no_cells_available(self):
        engine = _make_engine()
        engine.cell = None
        engine.run_manager = None

        assert engine.get_cell() is None


def _make_server_runner_for_submission(status="started"):
    runner = ServerRunner.__new__(ServerRunner)
    runner.wf_lock = threading.RLock()
    runner.status = status
    runner.current_wf = MagicMock()
    runner.log_info = MagicMock()
    runner._report_client_active = MagicMock()
    runner._result_receipts = OrderedDict()
    return runner


class TestLateSubmissionAdmission:
    def test_submission_after_terminal_state_does_not_touch_run_state(self):
        runner = _make_server_runner_for_submission(status="done")
        fl_ctx = MagicMock()

        with patch.object(runner, "_process_submission") as process_submission:
            runner.process_submission(MagicMock(name="client"), "train", "task-1", Shareable(), fl_ctx)

        process_submission.assert_not_called()
        runner._report_client_active.assert_not_called()
        fl_ctx.set_prop.assert_not_called()

    def test_submission_queued_behind_teardown_is_dropped(self):
        runner = _make_server_runner_for_submission()
        fl_ctx = MagicMock()
        finished = threading.Event()

        def submit():
            runner.process_submission(MagicMock(name="client"), "train", "task-1", Shareable(), fl_ctx)
            finished.set()

        with runner.wf_lock:
            thread = threading.Thread(target=submit)
            thread.start()
            runner.status = "done"
            runner.current_wf = None

        assert finished.wait(timeout=1.0)
        thread.join(timeout=1.0)
        runner._report_client_active.assert_not_called()
        fl_ctx.set_prop.assert_not_called()


def _submission_command_context(runner):
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.RUNNER, runner, private=True, sticky=False)
    data = Shareable()
    data.set_peer_context(FLContext())
    data.set_header(ServerCommandKey.FL_CLIENT, Client("site-1", "token"))
    data.set_header(FLContextKey.TASK_NAME, "train")
    data.add_cookie(FLContextKey.TASK_ID, "task-1")
    data.add_cookie(FLContextKey.TASK_ATTEMPT_ID, "attempt-1")
    return fl_ctx, data


def test_submit_update_does_not_acknowledge_a_result_dropped_after_task_check():
    runner = _make_server_runner_for_submission(status="done")
    fl_ctx, data = _submission_command_context(runner)
    reply = SubmitUpdateCommand().process(data, fl_ctx)
    assert reply.get_header(ReservedHeaderKey.TASK_RESULT_ACCEPTED) is False
    runner._report_client_active.assert_not_called()


@pytest.mark.parametrize("accepted", [False, True])
def test_submit_update_acknowledgement_requires_workflow_admission(accepted):
    runner = MagicMock()
    fl_ctx, data = _submission_command_context(runner)
    runner.process_submission.side_effect = lambda *_args: fl_ctx.set_prop(
        FLContextKey.TASK_RESULT_ACCEPTED, accepted, private=True, sticky=False
    )
    reply = SubmitUpdateCommand().process(data, fl_ctx)
    assert reply.get_header(ReservedHeaderKey.TASK_RESULT_ACCEPTED) is accepted
    assert reply.get_header(ReservedHeaderKey.TASK_ID) == "task-1"
    assert reply.get_header(ReservedHeaderKey.TASK_ATTEMPT_ID) == "attempt-1"


def _fenced_runner():
    runner = _make_server_runner_for_submission()
    runner.job_id = "job-1"
    runner.config = SimpleNamespace(task_result_filters={})
    runner.abort_signal = Signal()
    runner.log_debug = MagicMock()
    runner.log_warning = MagicMock()
    runner.fire_event = MagicMock()
    runner.system_panic = MagicMock()
    runner.log_exception = MagicMock()
    client = Client("site-1", "token")
    task = Task("train", Shareable(), result_received_cb=MagicMock())
    task.props["___mgr"] = MagicMock()
    assignment = ClientTask(client, task)
    communicator = WFCommServer()
    communicator._client_task_map[assignment.id] = assignment
    runner.current_wf = SimpleNamespace(id="workflow", controller=SimpleNamespace(communicator=communicator))
    fl_ctx = FLContextManager(identity_name="server", job_id="job-1").new_context()
    fl_ctx.set_peer_context(FLContextManager(identity_name="site-1", job_id="job-1").new_context())
    return runner, assignment, fl_ctx


@pytest.mark.parametrize("attempt_id", [None, "stale", ""])
def test_attempt_fence_is_checked_before_fatal_return_codes_filters_or_callbacks(attempt_id):
    runner, assignment, fl_ctx = _fenced_runner()
    result = make_reply(ReturnCode.UNSAFE_JOB)
    if attempt_id is not None:
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, attempt_id)
    with (
        patch("nvflare.private.fed.server.server_runner.apply_filters") as filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
    filters.assert_not_called()
    runner.system_panic.assert_not_called()
    assignment.task.result_received_cb.assert_not_called()


def test_result_filter_replacement_preserves_admitted_attempt_and_duplicate_skips_filter():
    runner, assignment, fl_ctx = _fenced_runner()
    result = Shareable({"original": True})
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    replacement = Shareable({"filtered": True})
    with (
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=replacement) as filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
    filters.assert_called_once()
    assignment.task.result_received_cb.assert_called_once()
    assert assignment.result is replacement
    assert replacement.get_task_attempt_id() == assignment.attempt_id
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is True


@pytest.mark.parametrize("return_code", [ReturnCode.EXECUTION_EXCEPTION, ReturnCode.TASK_ABORTED, "custom.failure"])
def test_server_filter_non_ok_result_cannot_acknowledge_state_admission(return_code):
    runner, assignment, fl_ctx = _fenced_runner()
    result = Shareable()
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    result.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    failure = make_reply(return_code)
    with (
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=failure) as filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
        assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is False
        runner.current_wf = None
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is False
    filters.assert_called_once()
    assignment.task.result_received_cb.assert_called_once()


def test_manager_exception_records_rejection_for_lost_ack_across_workflow_teardown():
    runner, assignment, fl_ctx = _fenced_runner()
    manager = assignment.task.props["___mgr"]

    def fail_after_teardown(*_args):
        runner.current_wf = None
        raise RuntimeError("manager side effect failed")

    manager.check_task_result.side_effect = fail_after_teardown
    result = Shareable()
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    result.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    with (
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=result) as filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
        assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is False
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
    key = runner._result_receipt_key("site-1", "train", assignment.id, assignment.attempt_id, "workflow")
    assert runner._result_receipts[key] is False
    filters.assert_called_once()
    manager.check_task_result.assert_called_once()
    assignment.task.result_received_cb.assert_not_called()
    runner.log_exception.assert_called_once()
    assert "manager side effect failed" in runner.log_exception.call_args.args[1]


def test_get_task_cookies_preserve_attempt_for_existing_job_based_clients():
    runner = MagicMock()
    fl_ctx, request = _submission_command_context(runner)
    assigned = Shareable()
    assigned.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt-1")
    runner.process_task_request.return_value = ("train", "task-1", assigned)
    reply = GetTaskCommand().process(request, fl_ctx)
    # Existing clients only need their original cookie-jar echo behavior.
    legacy_result = Shareable()
    legacy_result.set_cookie_jar(reply.get_cookie_jar())
    assert legacy_result.get_task_attempt_id() == "attempt-1"
    assert legacy_result.get_cookie(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED) is True


@pytest.mark.parametrize("filter_workflow", [None, "filter-replacement-workflow"])
def test_fresh_task_data_filter_preserves_issued_workflow_for_lost_ack_after_transition(filter_workflow):
    runner, assignment, fl_ctx = _fenced_runner()
    runner._processing_tasks = {}
    runner._processing_tasks_lock = threading.Lock()
    runner.config.task_data_filters = {}
    runner.config.task_request_interval = 1.0
    fl_ctx.set_prop(ReservedKey.ENGINE, MagicMock(spec=ServerEngineSpec), private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.RUNNER, runner, private=True, sticky=False)
    request = Shareable()
    request.set_peer_context(fl_ctx.get_peer_context())
    request.set_header(ServerCommandKey.FL_CLIENT, assignment.client)
    assigned = Shareable({"assigned": True})
    assigned.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    filtered = Shareable({"filtered": True})
    if filter_workflow is not None:
        filtered.add_cookie(ReservedHeaderKey.WORKFLOW, filter_workflow)
    communicator = runner.current_wf.controller.communicator
    with (
        patch.object(communicator, "process_task_request", return_value=("train", assignment.id, assigned)),
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=filtered) as filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        reply = GetTaskCommand().process(request, fl_ctx)
    filters.assert_called_once()
    assert reply is filtered
    assert reply.get_cookie(ReservedHeaderKey.WORKFLOW) == "workflow"
    assert reply.get_task_attempt_id() == assignment.attempt_id
    assert FLContextKey.WORKFLOW not in fl_ctx.get_all_public_props()

    # Both existing job clients and task clients echo the assignment cookie jar.
    # This must retain the original issuing workflow after the server moves on.
    result = Shareable()
    result.set_cookie_jar(reply.get_cookie_jar())
    with (
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=result) as result_filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
        assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is True
        runner.current_wf = SimpleNamespace(id="next", controller=MagicMock())
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is True
    result_filters.assert_called_once()
    assignment.task.result_received_cb.assert_called_once()
    runner.current_wf.controller.communicator.process_submission.assert_not_called()


def test_missing_required_attempt_does_not_downgrade_to_legacy_unknown_handler():
    runner, assignment, fl_ctx = _fenced_runner()
    result = Shareable()
    result.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED, True)
    runner.current_wf.controller.communicator.controller = MagicMock()
    with patch("nvflare.private.fed.server.server_runner.add_job_audit_event"):
        runner.process_submission(assignment.client, "train", "unknown", result, fl_ctx)
    runner.current_wf.controller.communicator.controller.process_result_of_unknown_task.assert_not_called()


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("runner_state", ["next_workflow", "between_workflows", "done"])
def test_job_local_receipt_replays_lost_ack_across_workflow_teardown_without_application_code(accepted, runner_state):
    runner, assignment, fl_ctx = _fenced_runner()
    result = Shareable()
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    result.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    filtered = Shareable() if accepted else make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR)
    with (
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=filtered) as filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
        if runner_state == "next_workflow":
            runner.current_wf = SimpleNamespace(id="next", controller=MagicMock())
        else:
            runner.current_wf = None
            if runner_state == "done":
                runner.status = "done"
        # The wire result is immutable per attempt. Even a conflicting duplicate
        # must not rerun a callback, filter or fatal return-code handler.
        result.set_return_code(ReturnCode.UNSAFE_JOB)
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is accepted
    assignment.task.result_received_cb.assert_called_once()
    runner.system_panic.assert_not_called()
    filters.assert_called_once()
    if runner.current_wf is not None:
        runner.current_wf.controller.communicator.process_submission.assert_not_called()


@pytest.mark.parametrize("field", ["client_name", "task_name", "task_id", "attempt_id", "workflow_id", "job_id"])
def test_job_local_receipt_requires_exact_peer_and_all_assignment_fields(field):
    runner, assignment, fl_ctx = _fenced_runner()
    runner._remember_result_receipt(assignment.client, "train", assignment.id, assignment.attempt_id, "workflow", True)
    values = dict(
        client_name="site-1",
        task_name="train",
        task_id=assignment.id,
        attempt_id=assignment.attempt_id,
        workflow_id="workflow",
    )
    if field == "job_id":
        fl_ctx.set_peer_context(FLContextManager(identity_name="site-1", job_id="other-job").new_context())
    else:
        values[field] = "different"
    assert not runner._replay_result_receipt(**values, fl_ctx=fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is None


def test_completed_task_check_reaches_job_local_receipt_after_workflow_teardown():
    runner, assignment, fl_ctx = _fenced_runner()
    runner._remember_result_receipt(assignment.client, "train", assignment.id, assignment.attempt_id, "workflow", True)
    runner.current_wf = None
    request = Shareable()
    request.set_header(ReservedHeaderKey.TASK_NAME, "train")
    request.set_header(ReservedHeaderKey.TASK_ID, assignment.id)
    request.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    request.set_header(ReservedHeaderKey.WORKFLOW, "workflow")
    reply = runner._handle_task_check("task_check", request, fl_ctx)
    assert reply.get_return_code() == ReturnCode.OK
    request.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "stale")
    reply = runner._handle_task_check("task_check", request, fl_ctx)
    assert reply.get_return_code() == ReturnCode.TASK_UNKNOWN


def test_job_result_receipts_are_bounded_and_never_replace_a_recorded_decision(monkeypatch):
    runner, assignment, _ = _fenced_runner()
    monkeypatch.setattr("nvflare.private.fed.server.server_runner._MAX_TASK_RESULT_RECEIPTS", 2)
    for attempt in ("first", "second", "third"):
        runner._remember_result_receipt(assignment.client, "train", assignment.id, attempt, "workflow", False)
    assert len(runner._result_receipts) == 2
    assert [key[3] for key in runner._result_receipts] == ["second", "third"]
    runner._remember_result_receipt(assignment.client, "train", assignment.id, "second", "workflow", True)
    key = runner._result_receipt_key("site-1", "train", assignment.id, "second", "workflow")
    assert runner._result_receipts[key] is False


@pytest.mark.parametrize("failure_phase", ["late_hook", "before_process"])
def test_failed_late_result_receipt_survives_workflow_transition_only_after_hook_claim(failure_phase):
    runner, assignment, fl_ctx = _fenced_runner()
    communicator = runner.current_wf.controller.communicator
    # Model an issued assignment swept before its first result arrived.
    assignment.props["___job_id"] = "job-1"
    retired = communicator._remember_completed_client_task(assignment)
    communicator._client_task_map.pop(assignment.id)
    communicator.fire_event = MagicMock()
    communicator.controller = MagicMock()
    hook = communicator.controller.process_result_of_unknown_task
    hook.side_effect = RuntimeError("late hook failed after side effects")
    if failure_phase == "before_process":

        def fail_before_process(event, *_args):
            if event == EventType.BEFORE_PROCESS_SUBMISSION:
                raise RuntimeError("before late hook")

        runner.fire_event.side_effect = fail_before_process

    result = Shareable({"model": "unchanged"})
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    result.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    key = runner._result_receipt_key("site-1", "train", assignment.id, assignment.attempt_id, "workflow")
    with (
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=result) as filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
        assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is False
        if failure_phase == "before_process":
            assert retired.accepted is None
            assert key not in runner._result_receipts
            hook.assert_not_called()
            return
        assert retired.accepted is False
        assert runner._result_receipts[key] is False
        runner.current_wf = SimpleNamespace(id="next", controller=MagicMock())
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)

    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is False
    filters.assert_called_once()
    hook.assert_called_once()
    runner.current_wf.controller.communicator.process_submission.assert_not_called()


@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_first_late_result_reaches_hook_once_and_replays_after_workflow_transition(swept, accepted):
    runner, assignment, fl_ctx = _fenced_runner()
    communicator = runner.current_wf.controller.communicator
    assignment.props["___job_id"] = "job-1"
    assignment.task.completion_status = TaskCompletionStatus.TIMEOUT
    if swept:
        communicator._remember_completed_client_task(assignment)
        communicator._client_task_map.pop(assignment.id)
    communicator.fire_event = MagicMock()
    communicator.controller = MagicMock()

    def late_hook(*_args):
        if accepted:
            fl_ctx.set_prop(FLContextKey.TASK_RESULT_ACCEPTED, True, private=True, sticky=False)

    hook = communicator.controller.process_result_of_unknown_task
    hook.side_effect = late_hook
    request = Shareable()
    request.set_header(ReservedHeaderKey.TASK_NAME, "train")
    request.set_header(ReservedHeaderKey.TASK_ID, assignment.id)
    request.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    request.set_header(ReservedHeaderKey.WORKFLOW, "workflow")
    assert runner._handle_task_check("task_check", request, fl_ctx).get_return_code() == ReturnCode.OK

    result = Shareable({"original": True})
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    result.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    replacement = Shareable({"filtered": True})
    with (
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=replacement) as filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
        assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is accepted
        assert replacement.get_task_attempt_id() == assignment.attempt_id
        runner.current_wf = SimpleNamespace(id="next", controller=MagicMock())
        result.set_return_code(ReturnCode.UNSAFE_JOB)
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
        assert runner._handle_task_check("task_check", request, fl_ctx).get_return_code() == ReturnCode.OK

    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is accepted
    filters.assert_called_once()
    hook.assert_called_once_with(assignment.client, "train", assignment.id, replacement, fl_ctx)
    assignment.task.result_received_cb.assert_not_called()
    runner.system_panic.assert_not_called()
    runner.current_wf.controller.communicator.process_submission.assert_not_called()
