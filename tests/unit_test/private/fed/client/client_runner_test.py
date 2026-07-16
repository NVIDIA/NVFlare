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
from nvflare.apis.fl_constant import FilterKey, FLContextKey, ReservedKey, ReservedTopic, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef
from nvflare.private.defs import SpecialTaskName, TaskConstant
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from nvflare.private.fed.client.client_runner import (
    _TASK_CHECK_RESULT_OK,
    _TASK_CHECK_RESULT_TASK_GONE,
    _TASK_CHECK_RESULT_TRY_AGAIN,
    ClientRunner,
    ClientRunnerConfig,
    TaskRouter,
    _materialize_lazy_download_refs,
)
from nvflare.private.json_configer import ConfigError
from nvflare.private.privacy_manager import Scope
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


def _runner():
    runner = ClientRunner.__new__(ClientRunner)
    runner.task_router = TaskRouter()
    runner.task_data_filters = {}
    runner.task_result_filters = {}
    runner.default_task_fetch_interval = 0.5
    runner.parent_target = "server"
    runner.job_id = "job-1"
    runner.engine = MagicMock()
    runner.engine.get_cell.return_value.get_fqcn.return_value = "site-1.job-1"
    runner.run_abort_signal = Signal()
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


def _task(name="train", task_id="task-1", data=None):
    return TaskAssignment(name=name, task_id=task_id, data=data or Shareable())


def _lazy_payload(item_id="T0"):
    return Shareable({"payload": LazyDownloadRef("source", "ref-1", item_id, 1)})


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
    config = ClientRunnerConfig(TaskRouter(), {}, {}, handlers=[FLComponent()])
    monkeypatch.setattr(ClientRunner, "get_positive_float_var", lambda _self, _name, default: default)
    monkeypatch.setattr(ClientRunner, "register_event_handler", MagicMock())

    runner = ClientRunner({}, config, "job-1", engine)

    assert runner.parent_target == "server"
    topics = {call.kwargs["topic"] for call in engine.register_aux_message_handler.call_args_list}
    assert topics == {ReservedTopic.END_RUN, ReservedTopic.DO_TASK}
    event_types = {call.args[0] for call in ClientRunner.register_event_handler.call_args_list}
    assert {EventType.TASK_ASSIGNMENT_SENT, EventType.TASK_RESULT_RECEIVED}.issubset(event_types)


def test_process_task_preserves_cookie_and_assignment_headers():
    runner = _runner()
    task = _task()
    task.data.set_cookie_jar({"cookie": "value"})
    runner._do_process_task = MagicMock(return_value=Shareable())

    reply = runner._process_task(task, FLContext())

    assert reply.get_cookie_jar() == {"cookie": "value"}
    assert reply.get_header(ReservedHeaderKey.TASK_NAME) == "train"
    assert reply.get_header(ReservedHeaderKey.TASK_ID) == "task-1"


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


def _task_context_for_job(job_id="job-1"):
    fl_ctx = FLContext()
    peer_ctx = FLContext()
    peer_ctx.set_prop(ReservedKey.RUN_NUM, job_id, private=False, sticky=False)
    fl_ctx.set_peer_context(peer_ctx)
    return fl_ctx


def test_do_task_keeps_payload_lazy_for_pass_through_executor_without_filters():
    runner = _runner()
    executor = MagicMock()
    executor.supports_payload_pass_through.return_value = True
    executor.execute.return_value = Shareable()
    runner.task_router.add_executor(["train"], executor)

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner._materialize_lazy_download_refs") as materialize,
    ):
        reply = runner._do_task(_task(data=_lazy_payload()), _task_context_for_job(), Signal())

    assert reply.get_return_code() == ReturnCode.OK
    materialize.assert_not_called()


def test_do_task_keeps_result_lazy_for_pass_through_executor_without_filters():
    runner = _runner()
    executor = MagicMock()
    executor.supports_payload_pass_through.return_value = True
    lazy_result = _lazy_payload("result")
    executor.execute.return_value = lazy_result
    runner.task_router.add_executor(["train"], executor)

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner._materialize_lazy_download_refs") as materialize,
    ):
        reply = runner._do_task(_task(), _task_context_for_job(), Signal())

    assert reply is lazy_result
    materialize.assert_not_called()
    executed_task = executor.execute.call_args.args[1]
    assert executed_task.get_header(FOBSContextKey.RECEIVER_IDS) is None


def test_materialize_lazy_download_refs_propagates_task_abort_signal_to_fobs():
    lazy = LazyDownloadRef("server", "ref-1", "T0", 1)
    # Sets and mapping keys are both valid FOBS graph positions. Materialization must not
    # depend on a hand-written traversal that only notices ordinary dict values.
    data = Shareable({"set": {lazy}, lazy: "mapping-key"})
    fl_ctx = MagicMock()
    engine = MagicMock()
    cell = MagicMock()
    decode_ctx = {"cell": cell}
    abort_signal = Signal()
    fl_ctx.get_engine.return_value = engine
    engine.get_cell.return_value = cell
    cell.get_fobs_context.return_value = decode_ctx
    expected = Shareable()

    with (
        patch("nvflare.fuel.utils.fobs.dumps", return_value=b"encoded") as dumps,
        patch("nvflare.fuel.utils.fobs.loads", return_value=expected) as loads,
    ):
        result = _materialize_lazy_download_refs(data, fl_ctx, abort_signal)

    assert result is expected
    dumps.assert_called_once_with(data)
    cell.get_fobs_context.assert_called_once_with(
        props={FOBSContextKey.PASS_THROUGH: False, FOBSContextKey.ABORT_SIGNAL: abort_signal}
    )
    loads.assert_called_once_with(b"encoded", fobs_ctx=decode_ctx)


def test_do_task_materializes_lazy_payload_for_cj_local_executor_without_filters():
    runner = _runner()
    executor = MagicMock()
    executor.supports_payload_pass_through.return_value = False
    executor.execute.return_value = Shareable()
    runner.task_router.add_executor(["train"], executor)

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch(
            "nvflare.private.fed.client.client_runner._materialize_lazy_download_refs",
            side_effect=lambda data, _fl_ctx, _abort_signal: data,
        ) as materialize,
    ):
        reply = runner._do_task(_task(data=_lazy_payload()), _task_context_for_job(), Signal())

    assert reply.get_return_code() == ReturnCode.OK
    materialize.assert_called_once()


def test_do_task_does_not_reserialize_concrete_payload_for_cj_local_executor():
    runner = _runner()
    executor = MagicMock()
    executor.supports_payload_pass_through.return_value = False
    executor.execute.return_value = Shareable()
    runner.task_router.add_executor(["train"], executor)

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner._materialize_lazy_download_refs") as materialize,
    ):
        reply = runner._do_task(_task(data=Shareable({"payload": "concrete"})), _task_context_for_job(), Signal())

    assert reply.get_return_code() == ReturnCode.OK
    materialize.assert_not_called()


def test_do_task_materializes_before_applicable_data_and_result_filters():
    runner = _runner()
    executor = MagicMock()
    executor.supports_payload_pass_through.return_value = True
    executor.execute.return_value = _lazy_payload("result")
    runner.task_router.add_executor(["train"], executor)

    data_filter = MagicMock()
    data_filter.process.side_effect = lambda data, _ctx: data
    result_filter = MagicMock()
    result_filter.process.side_effect = lambda data, _ctx: data
    fl_ctx = _task_context_for_job()
    fl_ctx.set_prop(
        FLContextKey.SCOPE_OBJECT,
        SimpleNamespace(
            **{
                Scope.TASK_DATA_FILTERS_NAME: {FilterKey.IN: [data_filter]},
                Scope.TASK_RESULT_FILTERS_NAME: {FilterKey.OUT: [result_filter]},
            }
        ),
        private=True,
        sticky=False,
    )

    materialized = []

    def materialize(data, _fl_ctx, _abort_signal):
        materialized.append(data)
        return data

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner._materialize_lazy_download_refs", side_effect=materialize),
    ):
        reply = runner._do_task(_task(data=_lazy_payload("task")), fl_ctx, Signal())

    assert reply.get_return_code() == ReturnCode.OK
    assert len(materialized) == 2
    data_filter.process.assert_called_once()
    result_filter.process.assert_called_once()
    executed_task = executor.execute.call_args.args[1]
    assert executed_task.get_header(FOBSContextKey.RECEIVER_IDS) == ["site-1.job-1"]


def test_do_task_does_not_reserialize_concrete_result_for_result_filter():
    runner = _runner()
    executor = MagicMock()
    executor.supports_payload_pass_through.return_value = True
    executor.execute.return_value = Shareable({"payload": "concrete"})
    runner.task_router.add_executor(["train"], executor)
    result_filter = MagicMock()
    result_filter.process.side_effect = lambda data, _ctx: data
    runner.task_result_filters["train" + FilterKey.DELIMITER + FilterKey.OUT] = [result_filter]

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner._materialize_lazy_download_refs") as materialize,
    ):
        reply = runner._do_task(_task(), _task_context_for_job(), Signal())

    assert reply.get_return_code() == ReturnCode.OK
    materialize.assert_not_called()
    result_filter.process.assert_called_once()


def test_after_task_execution_sees_materialized_result_when_result_filter_requires_it():
    runner = _runner()
    executor = MagicMock()
    executor.supports_payload_pass_through.return_value = True
    lazy_result = _lazy_payload("result")
    concrete_result = Shareable({"payload": "concrete"})
    executor.execute.return_value = lazy_result
    runner.task_router.add_executor(["train"], executor)
    result_filter = MagicMock()
    result_filter.process.side_effect = lambda data, _ctx: data
    runner.task_result_filters["train" + FilterKey.DELIMITER + FilterKey.OUT] = [result_filter]
    observed_after_execution = []

    def fire_event(event_type, fl_ctx):
        if event_type == EventType.AFTER_TASK_EXECUTION:
            observed_after_execution.append(fl_ctx.get_prop(FLContextKey.TASK_RESULT))

    runner.fire_event.side_effect = fire_event

    def materialize(data, _fl_ctx, _abort_signal):
        return concrete_result if data is lazy_result else data

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner._materialize_lazy_download_refs", side_effect=materialize),
    ):
        reply = runner._do_task(_task(), _task_context_for_job(), Signal())

    assert reply is concrete_result
    assert observed_after_execution == [concrete_result]
    executed_task = executor.execute.call_args.args[1]
    assert executed_task.get_header(FOBSContextKey.RECEIVER_IDS) == ["site-1.job-1"]


def test_do_task_maps_abort_during_task_data_materialization_to_task_aborted():
    runner = _runner()
    executor = MagicMock()
    executor.supports_payload_pass_through.return_value = True
    runner.task_router.add_executor(["train"], executor)
    data_filter = MagicMock()
    runner.task_data_filters["train" + FilterKey.DELIMITER + FilterKey.IN] = [data_filter]
    abort_signal = Signal()

    def materialize(_data, _fl_ctx, signal):
        assert signal is abort_signal
        signal.trigger(True)
        raise RuntimeError("download aborted")

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner._materialize_lazy_download_refs", side_effect=materialize),
    ):
        reply = runner._do_task(_task(data=_lazy_payload()), _task_context_for_job(), abort_signal)

    assert reply.get_return_code() == ReturnCode.TASK_ABORTED
    executor.execute.assert_not_called()
    data_filter.process.assert_not_called()


def test_do_task_maps_abort_during_task_result_materialization_to_task_aborted():
    runner = _runner()
    executor = MagicMock()
    executor.supports_payload_pass_through.return_value = True
    executor.execute.return_value = _lazy_payload("result")
    runner.task_router.add_executor(["train"], executor)
    result_filter = MagicMock()
    runner.task_result_filters["train" + FilterKey.DELIMITER + FilterKey.OUT] = [result_filter]
    abort_signal = Signal()

    def materialize(_data, _fl_ctx, signal):
        assert signal is abort_signal
        signal.trigger(True)
        raise RuntimeError("download aborted")

    with (
        patch("nvflare.private.fed.client.client_runner.add_job_audit_event", return_value="audit-id"),
        patch("nvflare.private.fed.client.client_runner._materialize_lazy_download_refs", side_effect=materialize),
    ):
        reply = runner._do_task(_task(), _task_context_for_job(), abort_signal)

    assert reply.get_return_code() == ReturnCode.TASK_ABORTED
    executor.execute.assert_called_once()
    result_filter.process.assert_not_called()


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


def test_fetch_and_run_processes_task_and_uses_requested_interval():
    runner = _runner()
    task = _task()
    task.data.set_header(TaskConstant.WAIT_TIME, 2.0)
    runner.engine.get_task_assignment.return_value = task
    runner._process_task = MagicMock(return_value=Shareable())
    runner._send_task_result = MagicMock()

    assert runner.fetch_and_run_one_task(FLContext()) == (2.0, True)
    runner._send_task_result.assert_called_once()


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


def test_try_send_result_once_sends_after_task_check(monkeypatch):
    runner = _runner()
    runner._check_task_once = MagicMock(return_value=_TASK_CHECK_RESULT_OK)
    runner.engine.send_task_result.return_value = True
    result = Shareable()
    monkeypatch.setattr("nvflare.private.fed.client.client_runner.delete_msg_root", MagicMock())

    assert runner._try_send_result_once(result, "task-1", FLContext()) == _TASK_CHECK_RESULT_OK
    assert result.get_header(ReservedHeaderKey.MSG_ROOT_ID)


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
