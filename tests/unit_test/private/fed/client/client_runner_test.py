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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import (
    FilterKey,
    FLContextKey,
    ReservedKey,
    ReservedTopic,
    ReturnCode,
    ServerCommandKey,
    ServerCommandNames,
    TaskResultReceipt,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import TaskExecutionInterrupted
from nvflare.apis.utils.reliable_message import ReliableMessage
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.private.defs import RequestHeader, SpecialTaskName, TaskConstant
from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from nvflare.private.fed.client.client_runner import (
    _TASK_CHECK_RESULT_OK,
    _TASK_CHECK_RESULT_TASK_GONE,
    _TASK_CHECK_RESULT_TRY_AGAIN,
    ClientRunner,
    ClientRunnerConfig,
    TaskRouter,
)
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.json_configer import ConfigError
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector
from tests.timing_utils import isolate_time


def _runner():
    runner = ClientRunner.__new__(ClientRunner)
    runner.task_router = TaskRouter()
    runner.task_data_filters = {}
    runner.task_result_filters = {}
    runner.default_task_fetch_interval = 0.5
    runner.parent_target = "server"
    runner.job_id = "job-1"
    runner.engine = MagicMock()
    runner.run_abort_signal = Signal()
    runner._run_abort_requested = False
    runner.task_lock = threading.Lock()
    runner.running_tasks = {}
    runner.task_check_timeout = 5.0
    runner.task_check_interval = 0.0
    runner.get_task_timeout = 5.0
    runner.submit_task_result_timeout = 5.0
    runner.log_debug = MagicMock()
    runner.log_info = MagicMock()
    runner.log_error = MagicMock()
    runner.log_exception = MagicMock()
    runner.fire_event = MagicMock()
    return runner


@pytest.mark.parametrize("initialization", ["ready", "exception", "aborted", "notification-failed"])
def test_runner_notifies_readiness_only_after_successful_initialization(initialization):
    runner = _runner()
    ready = MagicMock()
    if initialization == "notification-failed":
        ready.side_effect = RuntimeError("notification failed")
    runner.end_run_events_sequence = MagicMock()

    def initialize(*_):
        ready.assert_not_called()
        if initialization == "exception":
            raise RuntimeError("initialization failed")
        if initialization == "aborted":
            runner.run_abort_signal.trigger(True)

    runner.init_run = initialize
    runner._try_run = MagicMock()
    with (
        patch("nvflare.private.fed.client.client_runner.ReliableMessage.shutdown") as reliable_shutdown,
        patch("nvflare.private.fed.client.client_runner.DownloadService.shutdown") as download_shutdown,
    ):
        if initialization == "exception":
            with pytest.raises(RuntimeError, match="initialization failed"):
                runner.run("/app", SimpleNamespace(), on_ready=ready)
        else:
            runner.run("/app", SimpleNamespace(), on_ready=ready)
    assert ready.call_count == (1 if initialization in ("ready", "notification-failed") else 0)
    if initialization != "exception":
        runner.end_run_events_sequence.assert_called_once()
        reliable_shutdown.assert_called_once()
        runner.engine.shutdown_streamer.assert_called_once()
        download_shutdown.assert_called_once()
    if initialization == "notification-failed":
        runner._try_run.assert_called_once()
        runner.log_exception.assert_called_once()


def test_cj_status_notifications_preserve_startup_abort_and_retry_without_crashing(monkeypatch):
    app = ClientAppRunner()
    runner = MagicMock()
    app.create_client_runner = MagicMock(return_value=runner)
    app.start_command_agent = MagicMock()
    client = MagicMock()
    client.cell.core_cell.get_fqcn.return_value = "site-1.job-1"
    clock = isolate_time(monkeypatch, "nvflare.private.fed.client.client_app_runner")
    now = [0]
    clock.time = lambda: now[0]
    clock.sleep = lambda _: None
    statuses = []

    def send(**kwargs):
        status = kwargs["request"].payload.get_header(RequestHeader.JOB_STATUS)
        statuses.append(status)
        now[0] += 6
        # Lose every READY acknowledgement and the first STOPPED delivery.
        rc = CellReturnCode.OK
        if status == ClientStatus.READY or (status == ClientStatus.STOPPED and statuses.count(status) == 1):
            rc = CellReturnCode.TIMEOUT
        return CellMessage({MessageHeaderKey.RETURN_CODE: rc})

    def run(*_, on_ready):
        assert statuses == [ClientStatus.STARTED]
        assert client.status == ClientStatus.STARTED  # Commands available during initialization.
        on_ready()

    client.cell.send_request.side_effect = send
    runner.run.side_effect = run
    app.start_run(
        "/app", SimpleNamespace(job_id="job-1", sp_target="server", sp_scheme="grpc"), "config", client, True, []
    )
    assert statuses == [ClientStatus.STARTED] + [ClientStatus.READY] * 3 + [ClientStatus.STOPPED] * 2
    assert client.status == ClientStatus.STOPPED
    client.stop_cell.assert_called_once()


@pytest.mark.parametrize(
    "return_code,payload,finished",
    [
        (CellReturnCode.OK, {ServerCommandKey.JOB_ID: "job-1", ServerCommandKey.CLIENTS: None}, True),
        (CellReturnCode.OK, {ServerCommandKey.JOB_ID: "job-1", ServerCommandKey.CLIENTS: {"site": "client"}}, False),
        (CellReturnCode.OK, {ServerCommandKey.JOB_ID: "job-1", ServerCommandKey.CLIENTS: {}}, False),
        (CellReturnCode.OK, {ServerCommandKey.JOB_ID: "other-job", ServerCommandKey.CLIENTS: None}, False),
        (CellReturnCode.OK, {ServerCommandKey.JOB_ID: "job-1"}, False),
        (CellReturnCode.OK, None, False),
        (CellReturnCode.TIMEOUT, {ServerCommandKey.JOB_ID: "job-1", ServerCommandKey.CLIENTS: None}, False),
        (CellReturnCode.UNAUTHENTICATED, {ServerCommandKey.JOB_ID: "job-1", ServerCommandKey.CLIENTS: None}, False),
    ],
)
def test_sync_failure_requires_parent_confirmation_before_finishing(monkeypatch, return_code, payload, finished):
    runner = _runner()
    ready = MagicMock()
    runner._try_run = MagicMock()
    runner.end_run_events_sequence = MagicMock()
    # A replacement whose server job has ended never enables ReliableMessage.
    monkeypatch.setattr(ReliableMessage, "_executor", None)
    monkeypatch.setattr(ReliableMessage, "_shutdown_asked", False)
    runner.get_positive_float_var = MagicMock(return_value=1)
    runner.engine.send_aux_request.return_value = {}
    cell = runner.engine.get_cell.return_value
    cell.send_request.return_value = CellMessage({MessageHeaderKey.RETURN_CODE: return_code}, payload)
    clock = isolate_time(monkeypatch, "nvflare.private.fed.client.client_runner")
    ticks = iter([0, 2])
    clock.time = lambda: next(ticks)
    clock.sleep = lambda _: None
    with (
        patch("nvflare.private.fed.client.client_runner.DownloadService.initialize"),
        patch("nvflare.private.fed.client.client_runner.DownloadService.shutdown") as download_shutdown,
    ):
        if finished:
            runner.run("/app", SimpleNamespace(), on_ready=ready)
            runner.engine.shutdown_streamer.assert_called_once()
            download_shutdown.assert_called_once()
        else:
            with pytest.raises(RuntimeError, match="cannot sync"):
                runner.init_run("/app", SimpleNamespace())
    assert runner.run_abort_signal.triggered is finished
    assert cell.send_request.call_args.kwargs["topic"] == ServerCommandNames.GET_CLIENTS
    ready.assert_not_called()
    runner._try_run.assert_not_called()
    runner.end_run_events_sequence.assert_not_called()
    runner.fire_event.assert_not_called()


def _task(name="train", task_id="task-1", data=None):
    return TaskAssignment(name=name, task_id=task_id, data=data or Shareable())


@pytest.mark.parametrize("failure", ["worker-lost", "application-error", "abort"])
def test_worker_loss_exits_cj_without_reporting_a_failed_task_or_stopped_status(failure):
    runner = _runner()
    context, peer = FLContext(), FLContext()
    peer.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    context.set_peer_context(peer)
    runner.engine.get_task_assignment.return_value = _task()
    runner.init_run = MagicMock()
    runner.end_run_events_sequence = MagicMock()
    runner._send_task_result = MagicMock(return_value=True)
    runner._try_run = lambda: runner.fetch_and_run_one_task(context)

    def execute(_name, _data, _ctx, abort_signal):
        if failure == "application-error":
            raise ValueError("bad input")
        if failure == "abort":
            abort_signal.trigger(True)
        raise TaskExecutionInterrupted("worker timed out")

    executor = MagicMock(execute=MagicMock(side_effect=execute))
    runner.task_router.add_executor(["train"], executor)
    app = ClientAppRunner()
    app.create_client_runner = MagicMock(return_value=runner)
    app.start_command_agent = MagicMock()
    app.sync_up_parents_process = MagicMock()
    app.notify_job_status = MagicMock()
    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event"),
        patch("nvflare.private.fed.client.client_runner.ReliableMessage.shutdown"),
        patch("nvflare.private.fed.client.client_runner.DownloadService.shutdown"),
    ):
        args = (
            "/app",
            SimpleNamespace(job_id="job-1", sp_target="server", sp_scheme="grpc"),
            "config",
            MagicMock(),
            True,
            [],
        )
        if failure == "worker-lost":
            with pytest.raises(TaskExecutionInterrupted, match="worker timed out"):
                app.start_run(*args)
            runner._send_task_result.assert_not_called()
        else:
            app.start_run(*args)
            expected = ReturnCode.TASK_ABORTED if failure == "abort" else ReturnCode.EXECUTION_EXCEPTION
            assert runner._send_task_result.call_args.args[0].get_return_code() == expected
    statuses = [call.args[2] for call in app.notify_job_status.call_args_list]
    assert statuses == [ClientStatus.STARTED, ClientStatus.READY] + (
        [] if failure == "worker-lost" else [ClientStatus.STOPPED]
    )
    runner.end_run_events_sequence.assert_called_once()
    runner.engine.shutdown_streamer.assert_called_once()


def test_task_router_prefers_exact_match_then_patterns():
    exact = MagicMock()
    wildcard = MagicMock()
    router = TaskRouter()
    router.add_executor(["train*"], wildcard)
    router.add_executor(["train-model"], exact)

    assert router.route("train-model") is exact
    assert router.route("train-evaluate") is wildcard
    assert router.route("validate") is None
    with pytest.raises(ConfigError, match="multiple executors"):
        router.add_executor(["train-model"], MagicMock())


def test_client_runner_config_initializes_and_validates_components():
    config = ClientRunnerConfig(TaskRouter(), {}, {}, handlers=None, components=None)
    component = FLComponent()

    config.add_component("component", component)

    assert config.components == {"component": component}
    assert config.handlers == [component]
    with pytest.raises(TypeError, match="component id must be str"):
        config.add_component(1, object())
    with pytest.raises(ValueError, match="duplicate component id"):
        config.add_component("component", object())


def test_client_runner_init_registers_aux_and_event_handlers(monkeypatch):
    engine = MagicMock()
    config = ClientRunnerConfig(TaskRouter(), {}, {})
    monkeypatch.setattr(ClientRunner, "get_positive_float_var", lambda _self, _name, default: default)
    monkeypatch.setattr(ClientRunner, "register_event_handler", MagicMock())

    runner = ClientRunner({}, config, "job-1", engine)

    assert runner.parent_target == "server"
    topics = {call.kwargs["topic"] for call in engine.register_aux_message_handler.call_args_list}
    assert topics == {ReservedTopic.END_RUN, ReservedTopic.DO_TASK}
    event_types = {call.args[0] for call in ClientRunner.register_event_handler.call_args_list}
    assert {EventType.TASK_ASSIGNMENT_SENT, EventType.TASK_RESULT_RECEIVED}.issubset(event_types)


def test_init_run_registers_download_source_failure_receiver_before_start(monkeypatch):
    runner = _runner()
    cell = MagicMock()
    runner.engine.get_cell.return_value = cell
    runner.engine.send_aux_request.return_value = {"server": make_reply(ReturnCode.OK)}
    runner.engine.new_context.return_value.__enter__.return_value = FLContext()
    monkeypatch.setattr(runner, "get_positive_float_var", lambda var_name, default: default)
    monkeypatch.setattr("nvflare.private.fed.client.client_runner.time.sleep", MagicMock())
    monkeypatch.setattr("nvflare.private.fed.client.client_runner.ReliableMessage.enable", MagicMock())
    initialize = MagicMock()
    monkeypatch.setattr("nvflare.private.fed.client.client_runner.DownloadService.initialize", initialize)

    runner.init_run("/app", SimpleNamespace())

    initialize.assert_called_once_with(cell)
    assert runner.fire_event.call_args_list[-1].args[0] == EventType.START_RUN


def test_process_task_preserves_cookie_and_assignment_headers():
    runner = _runner()
    task = _task()
    task.data.set_cookie_jar({"cookie": "value"})
    runner._do_process_task = MagicMock(return_value=Shareable())

    reply = runner._process_task(task, FLContext())

    assert reply.get_cookie_jar() == {"cookie": "value"}
    assert reply.get_header(ReservedHeaderKey.TASK_NAME) == "train"
    assert reply.get_header(ReservedHeaderKey.TASK_ID) == "task-1"


def test_task_assignment_and_result_preserve_authority_issued_attempt():
    runner = _runner()
    data = Shareable()
    data.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "server-attempt")
    task = _task(data=data)
    runner._do_process_task = MagicMock(return_value=Shareable())
    reply = runner._process_task(task, FLContext())
    assert task.attempt_id == "server-attempt"
    assert reply.get_task_attempt_id() == "server-attempt"
    assert reply.get_header(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED) is True


def test_task_assignment_rejects_missing_required_authority_attempt():
    data = Shareable()
    data.set_header(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED, True)
    with pytest.raises(ValueError, match="requires a task attempt ID"):
        _task(data=data)


def test_process_task_preserves_assignment_cookies_when_data_filter_replaces_shareable():
    runner = _runner()
    data = Shareable()
    data.set_cookie_jar(
        {ReservedHeaderKey.TASK_ID: "task-1", ReservedHeaderKey.TASK_ATTEMPT_ID: "server-attempt", "cookie": "value"}
    )
    task = _task(data=data)

    def replace_data(_task, _fl_ctx):
        _task.data = Shareable({"filtered": True})
        return Shareable()

    runner._do_process_task = MagicMock(side_effect=replace_data)
    reply = runner._process_task(task, FLContext())
    assert reply.get_cookie(ReservedHeaderKey.TASK_ID) == "task-1"
    assert reply.get_cookie("cookie") == "value"
    assert reply.get_task_attempt_id() == "server-attempt"


def test_do_process_task_short_circuits_unsafe_job():
    runner = _runner()
    fl_ctx = FLContext()
    fl_ctx.set_job_is_unsafe()

    reply = runner._do_process_task(_task(), fl_ctx)

    assert reply.get_return_code() == ReturnCode.UNSAFE_JOB


@pytest.mark.parametrize("result", [RuntimeError("failed"), object()])
def test_do_process_task_converts_executor_failures_to_shareable(result):
    runner = _runner()
    if isinstance(result, Exception):
        runner._do_task = MagicMock(side_effect=result)
    else:
        runner._do_task = MagicMock(return_value=result)

    reply = runner._do_process_task(_task(), FLContext())

    assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert runner.running_tasks == {}


@pytest.mark.parametrize(
    "task_data, peer_ctx, peer_job_id, expected",
    [
        (object(), None, None, ReturnCode.BAD_TASK_DATA),
        (Shareable(), None, None, ReturnCode.MISSING_PEER_CONTEXT),
        (Shareable(), object(), None, ReturnCode.BAD_PEER_CONTEXT),
        (Shareable(), FLContext(), "other-job", ReturnCode.RUN_MISMATCH),
        (Shareable(), FLContext(), "job-1", ReturnCode.TASK_UNKNOWN),
    ],
)
def test_do_task_validates_assignment_context(task_data, peer_ctx, peer_job_id, expected):
    runner = _runner()
    fl_ctx = FLContext()
    if peer_ctx is not None:
        if isinstance(peer_ctx, FLContext):
            peer_ctx.set_prop(ReservedKey.RUN_NUM, peer_job_id, private=False, sticky=False)
        fl_ctx.set_peer_context(peer_ctx)

    with patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"):
        reply = runner._do_task(_task(data=task_data), fl_ctx, Signal())

        assert reply.get_return_code() == expected


@pytest.mark.parametrize("filter_direction", [FilterKey.IN, FilterKey.OUT])
def test_do_task_reports_abort_during_filter_materialization(filter_direction):
    runner = _runner()
    executor = MagicMock()
    executor.execute.return_value = Shareable()
    runner.task_router.add_executor(["train"], executor)
    peer_ctx = FLContext()
    peer_ctx.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    fl_ctx = FLContext()
    fl_ctx.set_peer_context(peer_ctx)
    abort_signal = Signal()

    def apply_or_abort(_filter_name, filter_data, _fl_ctx, _filters, _task_name, direction, **_kwargs):
        if direction == filter_direction:
            abort_signal.trigger(True)
            raise RuntimeError("download aborted")
        return filter_data

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner.apply_filters", side_effect=apply_or_abort),
    ):
        reply = runner._do_task(_task(), fl_ctx, abort_signal)

    assert reply.get_return_code() == ReturnCode.TASK_ABORTED
    runner.log_exception.assert_not_called()


@pytest.mark.parametrize(
    "filter_failure, expected",
    [
        ("exception", ReturnCode.TASK_RESULT_FILTER_ERROR),
        ("invalid_result", ReturnCode.TASK_RESULT_FILTER_ERROR),
        ("abort", ReturnCode.TASK_ABORTED),
        ("unsafe", ReturnCode.UNSAFE_JOB),
    ],
)
def test_process_task_binds_final_failure_reply_after_successful_executor_result(filter_failure, expected):
    runner = _runner()
    success = Shareable({"successful_execution": True})
    executor = MagicMock()
    executor.execute.return_value = success
    runner.task_router.add_executor(["train"], executor)
    peer_ctx = FLContext()
    peer_ctx.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    fl_ctx = FLContext()
    fl_ctx.set_peer_context(peer_ctx)

    def fail_result_filter(_filter_name, data, _fl_ctx, _filters, _task_name, direction, **kwargs):
        if direction == FilterKey.IN:
            return data
        # Reproduce the stale-OK context specifically, rather than mocking the
        # entire task processor: these paths return before AFTER_RESULT_FILTER.
        assert data is success
        assert fl_ctx.get_prop(FLContextKey.TASK_RESULT) is success
        if filter_failure == "invalid_result":
            return object()
        if filter_failure == "abort":
            kwargs["abort_signal"].trigger(True)
        elif filter_failure == "unsafe":
            raise UnsafeJobError("unsafe result filter")
        raise RuntimeError("result filter failed")

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner.apply_filters", side_effect=fail_result_filter),
    ):
        reply = runner._process_task(_task(), fl_ctx)

    assert reply.get_return_code() == expected
    assert reply is not success
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT) is reply
    assert runner.running_tasks == {}


@pytest.mark.parametrize(
    "assignment, expected",
    [
        (None, (0.5, False)),
        (_task(SpecialTaskName.END_RUN), (0.5, False)),
        (_task(SpecialTaskName.TRY_AGAIN), (0.5, False)),
    ],
)
def test_fetch_and_run_handles_no_work_and_control_tasks(assignment, expected):
    runner = _runner()
    runner.engine.get_task_assignment.return_value = assignment

    assert runner.fetch_and_run_one_task(FLContext()) == expected


@pytest.mark.parametrize("send_success", [True, False])
def test_fetch_and_run_processes_task_and_exposes_publication_outcome(send_success):
    runner = _runner()
    task = _task()
    task.data.set_header(TaskConstant.WAIT_TIME, 2.0)
    runner.engine.get_task_assignment.return_value = task
    runner._process_task = MagicMock(return_value=Shareable())
    runner._send_task_result = MagicMock(return_value=send_success)
    observed = []
    runner.fire_event = lambda event_type, fl_ctx: observed.append(
        (event_type, fl_ctx.get_prop(FLContextKey.TASK_RESULT_SEND_SUCCESS))
    )
    fl_ctx = FLContext()

    assert runner.fetch_and_run_one_task(fl_ctx) == (2.0, True)
    runner._send_task_result.assert_called_once()
    assert observed[-1] == (EventType.AFTER_SEND_TASK_RESULT, send_success)


@pytest.mark.parametrize(
    "response, expected",
    [
        ({"server": make_reply(ReturnCode.OK)}, _TASK_CHECK_RESULT_OK),
        ({"server": make_reply(ReturnCode.COMMUNICATION_ERROR)}, _TASK_CHECK_RESULT_TRY_AGAIN),
        ({"server": make_reply(ReturnCode.SERVER_NOT_READY)}, _TASK_CHECK_RESULT_TRY_AGAIN),
        ({"server": make_reply(ReturnCode.TASK_UNKNOWN)}, _TASK_CHECK_RESULT_TASK_GONE),
        ({"server": make_reply(ReturnCode.EXECUTION_EXCEPTION)}, _TASK_CHECK_RESULT_OK),
        ({"server": object()}, _TASK_CHECK_RESULT_TRY_AGAIN),
        (None, _TASK_CHECK_RESULT_TRY_AGAIN),
    ],
)
def test_check_task_once_classifies_server_replies(response, expected):
    runner = _runner()
    runner.engine.send_aux_request.return_value = response

    assert runner._check_task_once("task-1", FLContext()) == expected


@pytest.mark.parametrize(
    "reply_case",
    [
        "received",
        "closed",
        "open",
        "missing_receipt",
        "wrong_task",
        "wrong_attempt",
        "wrong_workflow",
        "malformed_headers",
    ],
)
def test_fenced_readiness_requires_assignment_bound_receipt(reply_case):
    runner = _runner()
    ctx = FLContext()
    ctx.set_prop(FLContextKey.TASK_ATTEMPT_ID, "attempt", private=True, sticky=False)
    ctx.set_prop(FLContextKey.TASK_NAME, "train", private=True, sticky=False)
    ctx.set_prop(FLContextKey.WORKFLOW, "workflow", private=True, sticky=False)
    reply = make_reply(ReturnCode.TASK_UNKNOWN if reply_case == "closed" else ReturnCode.OK)
    for key, value, field in (
        (ReservedHeaderKey.TASK_ID, "task", "wrong_task"),
        (ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt", "wrong_attempt"),
        (ReservedHeaderKey.WORKFLOW, "workflow", "wrong_workflow"),
    ):
        reply.set_header(key, "other" if reply_case == field else value)
    receipt = (
        TaskResultReceipt.TASK_CLOSED
        if reply_case == "closed"
        else TaskResultReceipt.RETRY if reply_case == "open" else TaskResultReceipt.RECEIVED
    )
    if reply_case != "missing_receipt":
        reply.set_header(ReservedHeaderKey.TASK_RESULT_RECEIPT, receipt)
    if reply_case == "malformed_headers":
        reply[ReservedHeaderKey.HEADERS] = "malformed"
    runner.engine.send_aux_request.return_value = {"server": reply}
    expected = (
        _TASK_CHECK_RESULT_OK
        if reply_case in ("received", "open")
        else _TASK_CHECK_RESULT_TASK_GONE if reply_case == "closed" else _TASK_CHECK_RESULT_TRY_AGAIN
    )
    assert runner._check_task_once("task", ctx) == expected


def test_explicit_retry_keeps_same_saved_result_until_received(tmp_path):
    runner = _runner()
    runner._check_task_once = MagicMock(return_value=_TASK_CHECK_RESULT_OK)
    ctx = FLContext()
    saved = tmp_path / "result"
    saved.write_bytes(b"finished task output")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"matching committed state")
    committed_at = checkpoint.stat().st_mtime_ns
    result = Shareable({"result_file": str(saved)})
    result.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt")
    observed = []

    def send(current, fl_ctx, **kwargs):
        observed.append(current)
        assert saved.read_bytes() == b"finished task output"
        assert checkpoint.read_bytes() == b"matching committed state"
        assert checkpoint.stat().st_mtime_ns == committed_at
        receipt = TaskResultReceipt.RETRY if len(observed) == 1 else TaskResultReceipt.RECEIVED
        fl_ctx.set_prop(FLContextKey.TASK_RESULT_RECEIPT, receipt, private=True, sticky=False)
        return True

    runner.engine.send_task_result.side_effect = send
    assert runner._send_task_result(result, "task", ctx)
    assert observed == [result, result]
    assert all(current is result for current in observed)
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED


def test_try_send_result_once_sends_after_task_check(monkeypatch):
    runner = _runner()
    runner._check_task_once = MagicMock(return_value=_TASK_CHECK_RESULT_OK)
    runner.engine.send_task_result.return_value = True
    result = Shareable()
    monkeypatch.setattr("nvflare.private.fed.client.client_runner.delete_msg_root", MagicMock())

    assert runner._try_send_result_once(result, "task-1", FLContext()) == _TASK_CHECK_RESULT_OK
    assert result.get_header(ReservedHeaderKey.MSG_ROOT_ID)


def test_send_task_result_retries_after_transport_failure(monkeypatch):
    runner = _runner()
    call_order = []
    send_results = iter([False, True])

    def check_task(*_args, **_kwargs):
        call_order.append("check")
        return _TASK_CHECK_RESULT_OK

    def send_result(*_args, **_kwargs):
        call_order.append("send")
        return next(send_results)

    runner._check_task_once = MagicMock(side_effect=check_task)
    # Cell._send_one_request maps a failed StreamFuture to a timeout reply,
    # which send_task_result reports as False.
    runner.engine.send_task_result.side_effect = send_result
    monkeypatch.setattr("nvflare.private.fed.client.client_runner.time.sleep", MagicMock())
    monkeypatch.setattr("nvflare.private.fed.client.client_runner.delete_msg_root", MagicMock())
    result = Shareable()

    assert runner._send_task_result(result, "task-1", FLContext()) is True

    assert runner._check_task_once.call_count == 2
    assert runner.engine.send_task_result.call_count == 2
    assert call_order == ["check", "send", "check", "send"]
    assert all(call.args[0] is result for call in runner.engine.send_task_result.call_args_list)
    runner.log_error.assert_called_once()


@pytest.mark.parametrize("reason", ["task_gone", "abort"])
@pytest.mark.parametrize("submitted", [False, True])
def test_send_task_result_distinguishes_no_submit_from_lost_ack(monkeypatch, reason, submitted):
    runner = _runner()
    fl_ctx = FLContext()
    # Facts from an earlier result must not leak into a new publication.
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_SUBMISSION_ATTEMPTED, True, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_RESULT_RECEIPT, True, private=True, sticky=False)
    checks = [_TASK_CHECK_RESULT_OK, _TASK_CHECK_RESULT_TASK_GONE] if submitted else [_TASK_CHECK_RESULT_TASK_GONE]
    runner._check_task_once = MagicMock(side_effect=checks)

    def lose_ack(*_args, **_kwargs):
        assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_SUBMISSION_ATTEMPTED) is True
        if reason == "abort":
            runner.run_abort_signal.trigger(True)
        return False

    runner.engine.send_task_result.side_effect = lose_ack
    if reason == "abort" and not submitted:
        runner.run_abort_signal.trigger(True)
    monkeypatch.setattr("nvflare.private.fed.client.client_runner.time.sleep", MagicMock())

    assert runner._send_task_result(Shareable(), "task-1", fl_ctx) is False
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_SUBMISSION_ATTEMPTED) is submitted
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) is None
    assert runner.engine.send_task_result.call_count == int(submitted)


def test_abort_during_successful_readiness_check_does_not_handoff_result():
    runner = _runner()
    fl_ctx = FLContext()

    def ready_then_abort(*_args):
        runner.run_abort_signal.trigger(True)
        return _TASK_CHECK_RESULT_OK

    runner._check_task_once = MagicMock(side_effect=ready_then_abort)
    assert runner._send_task_result(Shareable(), "task-1", fl_ctx) is False
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_SUBMISSION_ATTEMPTED) is False
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) is None
    runner.engine.send_task_result.assert_not_called()
    runner._check_task_once.assert_called_once()


def test_task_check_and_control_handlers():
    runner = _runner()
    fl_ctx = FLContext()

    assert runner._handle_sync_runner("topic", Shareable(), fl_ctx).get_return_code() == ReturnCode.OK
    assert runner._handle_job_heartbeat("topic", Shareable(), fl_ctx).get_return_code() == ReturnCode.OK
    assert runner._handle_task_check("topic", Shareable(), fl_ctx).get_return_code() == ReturnCode.BAD_REQUEST_DATA

    request = Shareable()
    request.set_header(ReservedHeaderKey.TASK_ID, "task-1")
    assert runner._handle_task_check("topic", request, fl_ctx).get_return_code() == ReturnCode.TASK_UNKNOWN
    runner.running_tasks["task-1"] = _task()
    assert runner._handle_task_check("topic", request, fl_ctx).get_return_code() == ReturnCode.OK

    assert runner._handle_end_run("topic", Shareable(), fl_ctx).get_return_code() == ReturnCode.OK
    assert runner.run_abort_signal.triggered


def test_explicit_abort_fires_abort_task_without_a_running_task():
    runner = _runner()
    fl_ctx = FLContext()
    runner.engine.new_context.return_value.__enter__.return_value = fl_ctx
    runner.check_end_run_readiness = MagicMock()

    runner.abort("abort job")
    runner.end_run_events_sequence()

    abort_calls = [call for call in runner.fire_event.call_args_list if call.args[0] == EventType.ABORT_TASK]
    assert len(abort_calls) == 1
    assert abort_calls[0].args[1].get_prop(FLContextKey.RUN_ABORT_REQUESTED) is True


def test_handle_event_reports_running_tasks_and_aborts_on_fatal_error():
    runner = _runner()
    runner.running_tasks = {"task-1": _task("train"), "task-2": _task("validate", "task-2")}
    collector = GroupInfoCollector()
    fl_ctx = FLContext()
    fl_ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, collector, private=True, sticky=False)

    runner.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, fl_ctx)

    assert collector.info["ClientRunner"] == {
        "job_id": "job-1",
        "current_tasks": ["train", "validate"],
    }

    fl_ctx.set_prop(FLContextKey.EVENT_DATA, "fatal", private=True, sticky=False)
    runner.handle_event(EventType.FATAL_SYSTEM_ERROR, fl_ctx)
    assert runner.run_abort_signal.triggered


@pytest.fixture(autouse=True)
def _isolate_module_clocks(monkeypatch):
    isolate_time(
        monkeypatch,
        "nvflare.private.fed.client.client_runner",
    )


@pytest.mark.parametrize("filter_kind", ["none", "data", "result", "site-data"])
def test_task_resolver_precedes_events_and_release_follows_result_filters(tmp_path, filter_kind):
    import torch
    from safetensors.torch import save_file

    from nvflare.app_common.widgets.job_stats_reporter import _estimate_payload_mb
    from nvflare.app_opt.pt.lazy_tensor_dict import safetensors_refs
    from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef

    runner = _runner()
    context, peer = FLContext(), FLContext()
    peer.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    context.set_peer_context(peer)
    file = tmp_path / "received.safetensors"
    save_file({"w": torch.ones(4)}, str(file))
    ref = safetensors_refs(str(file))["w"]
    expected_eager = filter_kind in ("data", "site-data")
    release = MagicMock()
    executor = MagicMock()

    def resolve(data, fl_ctx, *, abort_signal):
        assert not expected_eager
        data["w"] = ref
        return data, release

    def resolve_eager(data, cell, abort_signal):
        assert expected_eager
        data["w"] = torch.ones(4)
        return data

    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: None))

    executor.resolve_task_data.side_effect = resolve
    executor.execute.return_value = Shareable({"w": torch.ones(4)})
    runner.task_router.add_executor(["train"], executor)
    input_filter = MagicMock()
    input_filter.process.side_effect = lambda data, ctx: data
    result_filter = MagicMock()

    def filter_result(data, ctx):
        assert isinstance(data["w"], torch.Tensor)
        release.assert_not_called()
        data["w"].add_(2)
        return data

    result_filter.process.side_effect = filter_result
    if filter_kind == "data":
        runner.task_data_filters["tra*" + FilterKey.DELIMITER + FilterKey.IN] = [input_filter]
    elif filter_kind == "site-data":
        context.set_prop(
            FLContextKey.SCOPE_OBJECT,
            SimpleNamespace(task_data_filters={FilterKey.IN: [input_filter]}, task_result_filters={}),
        )
    elif filter_kind == "result":
        runner.task_result_filters["train" + FilterKey.DELIMITER + FilterKey.OUT] = [result_filter]
    events = []

    def event(event_type, ctx):
        if event_type in (
            EventType.BEFORE_TASK_DATA_FILTER,
            EventType.AFTER_TASK_DATA_FILTER,
            EventType.BEFORE_TASK_EXECUTION,
        ):
            data = ctx.get_prop(FLContextKey.TASK_DATA)
            assert isinstance(data["w"], torch.Tensor) if expected_eager else data["w"] is ref
            assert _estimate_payload_mb(data) > 0
            release.assert_not_called()
            events.append(event_type)

    runner.fire_event.side_effect = event
    task = _task(data=Shareable({"w": LazyDownloadRef("server", "tx", "T0")}))
    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event"),
        patch("nvflare.private.fed.client.client_runner.materialize_lazy_download_refs", side_effect=resolve_eager),
    ):
        result = runner._do_task(task, context, Signal())
    assert len(events) == 3
    if expected_eager:
        release.assert_not_called()
        executor.resolve_task_data.assert_not_called()
    else:
        release.assert_called_once_with()
    assert torch.equal(result["w"], torch.full((4,), 3.0 if filter_kind == "result" else 1.0))


@pytest.mark.parametrize("failure", ["abort", "invalid"])
@pytest.mark.parametrize("data_filter", [False, True])
def test_task_resolution_maps_failure_before_executor(failure, data_filter):
    from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef

    runner = _runner()
    context, peer = FLContext(), FLContext()
    peer.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    context.set_peer_context(peer)
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: None))
    executor = MagicMock()
    runner.task_router.add_executor(["train"], executor)
    if data_filter:
        runner.task_data_filters["train" + FilterKey.DELIMITER + FilterKey.IN] = [MagicMock()]

    def resolve(data, ctx, abort_signal):
        if failure == "abort":
            abort_signal.trigger(True)
        raise RuntimeError("resolution failed")

    executor.resolve_task_data.side_effect = resolve
    task = _task(data=Shareable({"w": LazyDownloadRef("server", "tx", "T0")}))
    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event"),
        patch("nvflare.private.fed.client.client_runner.materialize_lazy_download_refs", side_effect=resolve),
    ):
        result = runner._do_task(task, context, Signal())
    expected = {"abort": ReturnCode.TASK_ABORTED, "invalid": ReturnCode.BAD_TASK_DATA}
    assert result.get_return_code() == expected[failure]
    executor.execute.assert_not_called()
    runner.fire_event.assert_not_called()


@pytest.mark.parametrize("phase", ["event", "filter", "executor"])
def test_task_resolver_releases_input_on_later_failure(phase):
    from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef

    runner = _runner()
    context, peer = FLContext(), FLContext()
    peer.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
    context.set_peer_context(peer)
    release, executor = MagicMock(), MagicMock()
    executor.resolve_task_data.return_value = (Shareable({"w": 1}), release)
    executor.execute.return_value = Shareable()
    runner.task_router.add_executor(["train"], executor)
    if phase == "event":
        runner.fire_event.side_effect = RuntimeError("hook failed")
    elif phase == "filter":
        filter_object = MagicMock()
        filter_object.process.side_effect = RuntimeError("filter failed")
        runner.task_result_filters["train" + FilterKey.DELIMITER + FilterKey.OUT] = [filter_object]
    else:
        executor.execute.side_effect = RuntimeError("executor failed")
    task = _task(data=Shareable({"w": LazyDownloadRef("server", "tx", "T0")}))
    with patch("nvflare.private.fed.client.client_runner.add_job_audit_event"):
        if phase == "event":
            with pytest.raises(RuntimeError, match="hook failed"):
                runner._do_task(task, context, Signal())
        else:
            assert runner._do_task(task, context, Signal()).get_return_code() != ReturnCode.OK
    release.assert_called_once_with()
