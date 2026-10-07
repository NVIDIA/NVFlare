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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import (
    FilterKey,
    FLContextKey,
    ReservedKey,
    ReturnCode,
    ServerCommandKey,
    TaskResultReceipt,
)
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.core_cell import MessageHeaderKey
from nvflare.fuel.f3.cellnet.core_cell import ReturnCode as CellReturnCode
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef
from nvflare.private.defs import new_cell_message
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from nvflare.private.fed.client.client_runner import ClientRunner, ClientRunnerConfig, TaskRouter
from nvflare.private.fed.client.communicator import Communicator
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
    return runner


class TestLateSubmissionAdmission:
    def test_submission_after_terminal_state_does_not_touch_run_state(self):
        runner = _make_server_runner_for_submission(status="done")
        fl_ctx = MagicMock()

        with patch.object(runner, "_process_submission") as process_submission:
            runner.process_submission(MagicMock(name="client"), "train", "task-1", Shareable(), fl_ctx)

        process_submission.assert_not_called()
        runner._report_client_active.assert_not_called()
        fl_ctx.set_prop.assert_called_once_with(
            FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.TASK_CLOSED, private=True, sticky=False
        )

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
        fl_ctx.set_prop.assert_called_once_with(
            FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.TASK_CLOSED, private=True, sticky=False
        )


def _submission_command_context(runner):
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.RUNNER, runner, private=True, sticky=False)
    data = Shareable()
    data.set_peer_context(FLContext())
    data.set_header(ServerCommandKey.FL_CLIENT, Client("site-1", "token"))
    data.set_header(FLContextKey.TASK_NAME, "train")
    data.add_cookie(FLContextKey.TASK_ID, "task-1")
    data.add_cookie(FLContextKey.TASK_ATTEMPT_ID, "attempt-1")
    data.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    return fl_ctx, data


def test_submit_update_does_not_acknowledge_a_result_dropped_after_task_check():
    runner = _make_server_runner_for_submission(status="done")
    fl_ctx, data = _submission_command_context(runner)
    reply = SubmitUpdateCommand().process(data, fl_ctx)
    assert reply.get_header(ReservedHeaderKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
    runner._report_client_active.assert_not_called()


@pytest.mark.parametrize(
    "receipt", [TaskResultReceipt.RECEIVED, TaskResultReceipt.TASK_CLOSED, TaskResultReceipt.RETRY]
)
def test_submit_update_reports_explicit_receipt_with_assignment_identity(receipt):
    runner = MagicMock()
    fl_ctx, data = _submission_command_context(runner)
    runner.process_submission.side_effect = lambda *_args: fl_ctx.set_prop(
        FLContextKey.TASK_RESULT_RECEIPT, receipt, private=True, sticky=False
    )
    reply = SubmitUpdateCommand().process(data, fl_ctx)
    assert reply.get_header(ReservedHeaderKey.TASK_RESULT_RECEIPT) == receipt
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
    assignment.props["___job_id"] = "job-1"
    runner.log_error = MagicMock()
    communicator = WFCommServer()
    communicator._client_task_map[assignment.id] = assignment
    runner.current_wf = SimpleNamespace(id="workflow", controller=SimpleNamespace(communicator=communicator))
    fl_ctx = FLContextManager(identity_name="server", job_id="job-1").new_context()
    fl_ctx.set_peer_context(FLContextManager(identity_name="site-1", job_id="job-1").new_context())
    return runner, assignment, fl_ctx


def _result(assignment, rc=ReturnCode.OK):
    result = make_reply(rc)
    result.add_cookie(ReservedHeaderKey.TASK_ID, assignment.id)
    result.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    result.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    return result


@pytest.mark.parametrize(
    "field", ["client", "task", "task_id", "attempt", "workflow", "missing_attempt", "conflicting_attempt", "job"]
)
@pytest.mark.parametrize("stage", ["active", "retired", "swept"])
def test_forged_result_cannot_run_filters_callbacks_late_hooks_or_fatal_effects(field, stage):
    runner, assigned, ctx = _fenced_runner()
    comm = runner.current_wf.controller.communicator
    comm.controller = MagicMock()
    if stage != "active":
        assigned.task.completion_status = TaskCompletionStatus.TIMEOUT
        if stage == "swept":
            comm._remember_completed_client_task(assigned)
            comm._client_task_map.pop(assigned.id)
    result = _result(assigned, ReturnCode.UNSAFE_JOB)
    client, name, task_id = assigned.client, "train", assigned.id
    if field == "client":
        client = Client("other", "token")
        ctx.set_peer_context(FLContextManager(identity_name="other", job_id="job-1").new_context())
    elif field == "job":
        ctx.set_peer_context(FLContextManager(identity_name="site-1", job_id="other").new_context())
    elif field == "task":
        name = "other"
    elif field == "task_id":
        task_id = "other"
    elif field == "attempt":
        result.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, "other")
    elif field == "missing_attempt":
        result.set_cookie_jar({ReservedHeaderKey.TASK_ID: assigned.id, ReservedHeaderKey.WORKFLOW: "workflow"})
    elif field == "conflicting_attempt":
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "other")
    else:
        result.add_cookie(ReservedHeaderKey.WORKFLOW, "other")
    with (
        patch("nvflare.private.fed.server.server_runner.apply_filters") as filters,
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
    ):
        runner.process_submission(client, name, task_id, result, ctx)
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
    filters.assert_not_called()
    runner.system_panic.assert_not_called()
    assigned.task.result_received_cb.assert_not_called()
    comm.controller.process_result_of_unknown_task.assert_not_called()


@pytest.mark.parametrize("rc", ServerRunner.ABORT_RETURN_CODES)
@pytest.mark.parametrize("stage", ["active", "retired", "swept"])
def test_fatal_complete_result_records_receipt_before_panic_and_never_repeats(rc, stage):
    runner, assigned, ctx = _fenced_runner()
    comm = runner.current_wf.controller.communicator
    if stage != "active":
        assigned.task.completion_status = TaskCompletionStatus.TIMEOUT
        if stage == "swept":
            comm._remember_completed_client_task(assigned)
            comm._client_task_map.pop(assigned.id)
    result = _result(assigned, rc)

    def panic(**kwargs):
        assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
        raise RuntimeError("panic after receipt")

    runner.system_panic.side_effect = panic
    with patch("nvflare.private.fed.server.server_runner.add_job_audit_event"):
        with pytest.raises(RuntimeError, match="panic after receipt"):
            runner.process_submission(assigned.client, "train", assigned.id, result, ctx)
        runner.process_submission(assigned.client, "train", assigned.id, result, ctx)
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    runner.system_panic.assert_called_once()
    assigned.task.result_received_cb.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["callback_false", "callback_throw", "manager_throw", "filter_throw", "before_filter", "before_process"]
)
def test_complete_result_receipt_is_independent_of_application_failures(failure):
    runner, assigned, ctx = _fenced_runner()
    result = _result(assigned)
    if failure == "callback_false":
        assigned.task.result_received_cb.return_value = False
    elif failure == "callback_throw":
        assigned.task.result_received_cb.side_effect = RuntimeError("callback failed")
    elif failure == "manager_throw":
        assigned.task.props["___mgr"].check_task_result.side_effect = RuntimeError("manager failed")

    def fire(event, *_args):
        if event == (
            EventType.BEFORE_TASK_RESULT_FILTER
            if failure == "before_filter"
            else EventType.BEFORE_PROCESS_SUBMISSION if failure == "before_process" else None
        ):
            raise RuntimeError("event failed")

    runner.fire_event.side_effect = fire
    with (
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
        patch(
            "nvflare.private.fed.server.server_runner.apply_filters",
            side_effect=RuntimeError("filter failed") if failure == "filter_throw" else None,
            return_value=result,
        ) as filters,
    ):
        runner.process_submission(assigned.client, "train", assigned.id, result, ctx)
        assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
        runner.process_submission(assigned.client, "train", assigned.id, result, ctx)
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assert assigned.result_received_time is not None
    assert filters.call_count == (0 if failure == "before_filter" else 1)
    assert assigned.task.result_received_cb.call_count <= 1


@pytest.mark.parametrize("consumer", ["filter", "callback"])
def test_submit_command_receipt_keeps_identity_when_application_consumes_input(consumer):
    runner, assigned, ctx = _fenced_runner()
    ctx.set_prop(FLContextKey.RUNNER, runner, private=True, sticky=False)
    result = _result(assigned)
    result.set_header(ServerCommandKey.FL_CLIENT, assigned.client)
    result.set_header(FLContextKey.TASK_NAME, "train")
    result.set_peer_context(ctx.get_peer_context())

    class ConsumingFilter(Filter):
        def process(self, shareable, fl_ctx):
            shareable.clear()
            return Shareable({"filtered": True})

    if consumer == "filter":
        runner.config.task_result_filters = {"train" + FilterKey.DELIMITER + FilterKey.IN: [ConsumingFilter()]}
    else:
        assigned.task.result_received_cb.side_effect = lambda client_task, fl_ctx: client_task.result.clear()

    with patch("nvflare.private.fed.server.server_runner.add_job_audit_event"):
        reply = SubmitUpdateCommand().process(result, ctx)
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assert result.get_cookie(ReservedHeaderKey.WORKFLOW) is None
    assert reply.get_task_result_receipt(assigned.id, assigned.attempt_id, "workflow") == TaskResultReceipt.RECEIVED
    assigned.task.result_received_cb.assert_called_once()


def test_unresolved_stream_reference_requires_retry_before_receipt_or_side_effects():
    runner, assigned, ctx = _fenced_runner()
    result = _result(assigned)
    result["model"] = LazyDownloadRef("client", "transfer", "tensor")
    with (
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=result) as filters,
    ):
        runner.process_submission(assigned.client, "train", assigned.id, result, ctx)
        assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RETRY
        assert assigned.result_received_time is None
        assert not assigned.props.get("___result_received")
        filters.assert_not_called()
        result["model"] = b"complete downloaded model"
        runner.process_submission(assigned.client, "train", assigned.id, result, ctx)
        assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
        result["model"] = LazyDownloadRef("client", "transfer", "tensor")
        runner.process_submission(assigned.client, "train", assigned.id, result, ctx)
        assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assigned.task.result_received_cb.assert_called_once()


@pytest.mark.parametrize("placement", ["cookie", "header_and_cookie"])
def test_filter_replacement_rebinds_complete_assignment_and_retries_skip_filter(placement):
    runner, assigned, ctx = _fenced_runner()
    original = _result(assigned)
    replacement = Shareable({"filtered": True})
    stale = {
        ReservedHeaderKey.TASK_ID: "prior-task",
        ReservedHeaderKey.TASK_ATTEMPT_ID: "prior-attempt",
        ReservedHeaderKey.TASK_ATTEMPT_REQUIRED: "bad",
        ReservedHeaderKey.WORKFLOW: "prior-workflow",
    }
    replacement.set_cookie_jar({**stale, "application-cookie": "preserved"})
    if placement == "header_and_cookie":
        for key, value in stale.items():
            replacement.set_header(key, value)
    with (
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=replacement) as filters,
    ):
        runner.process_submission(assigned.client, "train", assigned.id, original, ctx)
        runner.process_submission(assigned.client, "train", assigned.id, original, ctx)
    assert assigned.result is replacement
    assert replacement.get_task_attempt_id() == assigned.attempt_id
    assert replacement.get_cookie(ReservedHeaderKey.TASK_ID) == assigned.id
    assert replacement.get_cookie(ReservedHeaderKey.WORKFLOW) == "workflow"
    assert replacement.get_cookie("application-cookie") == "preserved"
    filters.assert_called_once()
    assigned.task.result_received_cb.assert_called_once()


@pytest.mark.parametrize("state", ["next_workflow", "between_workflows", "done"])
def test_old_result_is_closed_after_workflow_end_without_cross_workflow_receipt_cache(state):
    runner, assigned, ctx = _fenced_runner()
    result = _result(assigned)
    with (
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
        patch("nvflare.private.fed.server.server_runner.apply_filters", return_value=result) as filters,
    ):
        runner.process_submission(assigned.client, "train", assigned.id, result, ctx)
        assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
        comm = runner.current_wf.controller.communicator
        comm._clear_standing_tasks()
        assert not comm._completed_client_task_map
        runner.current_wf = SimpleNamespace(id="next", controller=MagicMock()) if state == "next_workflow" else None
        if state == "done":
            runner.status = "done"
        result.set_return_code(ReturnCode.UNSAFE_JOB)
        runner.process_submission(assigned.client, "train", assigned.id, result, ctx)
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
    assert not hasattr(runner, "_result_receipts")
    filters.assert_called_once()
    assigned.task.result_received_cb.assert_called_once()
    runner.system_panic.assert_not_called()


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
        assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
        runner.current_wf = SimpleNamespace(id="next", controller=MagicMock())
        runner.process_submission(assignment.client, "train", assignment.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
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


@pytest.mark.parametrize(
    "result_path",
    ["active_rejected", "active_accepted", "swept_late_failure", "before_process_failure"],
)
@pytest.mark.parametrize("teardown", [False, True])
def test_persisted_result_lost_ack_never_reexecutes_or_repeats_application_work(
    monkeypatch, tmp_path, result_path, teardown
):
    server, assignment, server_ctx = _fenced_runner()
    assignment.task.result_received_cb.return_value = result_path == "active_accepted"
    workflow_communicator = server.current_wf.controller.communicator
    late_effects = []
    if not result_path.startswith("active"):
        assignment.props["___job_id"] = "job-1"
        assignment.task.client_tasks.append(assignment)
        workflow_communicator._tasks.append(assignment.task)
        workflow_communicator._engine = MagicMock()
        workflow_communicator._engine.new_context.return_value = server_ctx
        workflow_communicator.fire_event = MagicMock()
        workflow_communicator.controller = MagicMock()

        def late_hook(*_args):
            late_effects.append("applied")
            raise RuntimeError("late hook failed after a side effect")

        workflow_communicator.controller.process_result_of_unknown_task.side_effect = late_hook
        if result_path == "before_process_failure":

            def before_process(event, *_args):
                if event == EventType.BEFORE_PROCESS_SUBMISSION:
                    raise RuntimeError("failure before the late hook claimed the result")

            server.fire_event.side_effect = before_process
    client_ctx = FLContextManager(identity_name=assignment.client.name, job_id="job-1").new_context()
    client_ctx.set_peer_context(FLContextManager(identity_name="server", job_id="job-1").new_context())
    client_ctx.set_prop(FLContextKey.SSID, "session", private=True, sticky=False)
    server_ctx.set_prop(FLContextKey.RUNNER, server, private=True, sticky=False)

    class CountingExecutor(Executor):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def execute(self, task_name, shareable, fl_ctx, abort_signal):
            self.calls += 1
            saved = tmp_path / "result"
            saved.write_text(str(self.calls))
            return Shareable({"execution": self.calls, "saved_result": str(saved)})

    executor = CountingExecutor()
    router = TaskRouter()
    router.add_executor(["train"], executor)
    client_engine = MagicMock()
    monkeypatch.setattr(ClientRunner, "get_positive_float_var", lambda _self, _name, default: default)
    client = ClientRunner({}, ClientRunnerConfig(router, {}, {}), "job-1", client_engine)
    client.fire_event = MagicMock()
    client.task_check_interval = 0.0
    communicator = Communicator(client_config={"client_name": assignment.client.name})
    communicator.cell = MagicMock()
    monkeypatch.setattr("nvflare.private.fed.client.communicator.determine_parent_fqcn", lambda *_args: "server")
    transport_calls = []
    readiness = []

    def transport(**kwargs):
        message = kwargs["request"]
        message.set_header(MessageHeaderKey.PAYLOAD_LEN, 0)
        data = message.payload
        data.set_header(ServerCommandKey.FL_CLIENT, assignment.client)
        reply = SubmitUpdateCommand().process(data, server_ctx)
        transport_calls.append(reply)
        if len(transport_calls) == 1:
            # The result reached the application, but its ACK was lost. The
            # workflow can disappear before the client checks and resubmits.
            if teardown:
                workflow_communicator._clear_standing_tasks()
                server.current_wf = None
            return new_cell_message({MessageHeaderKey.RETURN_CODE: CellReturnCode.TIMEOUT}, None)
        return new_cell_message({MessageHeaderKey.RETURN_CODE: CellReturnCode.OK}, reply)

    def send_result(result, fl_ctx, **_kwargs):
        rc = communicator.submit_update("project", "token", "session", fl_ctx, "site-1", result, "train")
        return rc == CellReturnCode.OK

    def check_task(**kwargs):
        reply = server._handle_task_check(kwargs["topic"], kwargs["request"], server_ctx)
        readiness.append(reply.get_return_code())
        return {"server": reply}

    def filter_result(_name, result, *_args, **_kwargs):
        if not result_path.startswith("active"):
            assignment.task.completion_status = TaskCompletionStatus.TIMEOUT
            workflow_communicator.check_tasks()
            assert assignment.id not in workflow_communicator._client_task_map
        return result

    communicator.cell.send_request.side_effect = transport
    client_engine.send_task_result.side_effect = send_result
    client_engine.send_aux_request.side_effect = check_task
    data = Shareable({"model": 1})
    data.add_cookie(ReservedHeaderKey.TASK_ID, assignment.id)
    data.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, assignment.attempt_id)
    data.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED, True)
    data.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    task = TaskAssignment("train", assignment.id, data)
    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit"),
        patch("nvflare.private.fed.server.server_runner.add_job_audit_event"),
        patch(
            "nvflare.private.fed.server.server_runner.apply_filters",
            side_effect=filter_result,
        ) as filters,
    ):
        result = client._process_task(task, client_ctx)
        assert client._send_task_result(result, assignment.id, client_ctx) is (not teardown)

    assert executor.calls == 1
    assert (tmp_path / "result").read_text() == "1"
    assert len(transport_calls) == 1
    assert readiness == [ReturnCode.OK, ReturnCode.TASK_UNKNOWN if teardown else ReturnCode.OK]
    assert client_ctx.get_prop(FLContextKey.TASK_RESULT_SUBMISSION_ATTEMPTED) is True
    assert client_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == (
        TaskResultReceipt.TASK_CLOSED if teardown else TaskResultReceipt.RECEIVED
    )
    assert transport_calls[0].get_header(ReservedHeaderKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    filters.assert_called_once()
    if result_path.startswith("active"):
        assignment.task.result_received_cb.assert_called_once()
    else:
        assignment.task.result_received_cb.assert_not_called()
        assert late_effects == ([] if result_path == "before_process_failure" else ["applied"])
        server.log_exception.assert_called_once()
    server.system_panic.assert_not_called()
