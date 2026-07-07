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

from unittest.mock import MagicMock, patch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.ccwf.client_controller_executor import ClientControllerExecutor
from nvflare.app_common.ccwf.common import Constant
from tests.unit_test.fl_context_helper import make_fl_context


class _Persistor(LearnablePersistor):
    def __init__(self):
        super().__init__()
        self.model = Learnable({"weights": 1})
        self.saved = []

    def load(self, fl_ctx):
        return self.model

    def save(self, learnable, fl_ctx):
        self.saved.append(learnable)


def _executor():
    executor = ClientControllerExecutor(["controller-1"], task_name_prefix="test")
    executor.log_debug = MagicMock()
    executor.log_info = MagicMock()
    executor.log_warning = MagicMock()
    executor.log_error = MagicMock()
    executor.system_panic = MagicMock()
    executor.fire_event = MagicMock()
    executor.fire_fed_event = MagicMock()
    return executor


def _context(engine=None):
    return make_fl_context(engine=engine, identity_name="site-1")


def test_get_config_prop_and_initialize():
    executor = _executor()
    assert executor.get_config_prop("missing", "default") == "default"
    executor.config = {"value": 1}
    assert executor.get_config_prop("value") == 1

    fl_ctx = _context()
    executor.initialize(fl_ctx)
    assert fl_ctx.get_prop(Constant.EXECUTOR) is executor
    executor.fire_event.assert_called_once_with(Constant.EXECUTOR_INITIALIZED, fl_ctx)


def test_start_run_validates_engine_runner_and_persistor():
    executor = _executor()
    no_engine = _context()
    executor.start_run(no_engine)
    executor.system_panic.assert_called_with("no engine", no_engine)

    engine = MagicMock()
    no_runner = _context(engine)
    executor.start_run(no_runner)
    executor.system_panic.assert_called_with("no client runner", no_runner)

    fl_ctx = _context(engine)
    fl_ctx.set_prop(FLContextKey.RUNNER, object(), private=True, sticky=False)
    engine.get_component.return_value = object()
    executor.initialize = MagicMock()
    executor.start_run(fl_ctx)
    assert executor.me == "site-1"
    assert executor.persistor is None
    executor.initialize.assert_called_once_with(fl_ctx)

    engine.get_component.return_value = _Persistor()
    executor.start_run(fl_ctx)
    assert isinstance(executor.persistor, _Persistor)


def test_initialize_controller_configures_communicator():
    executor = _executor()
    executor.engine = MagicMock()
    controller = MagicMock()
    executor.engine.get_component.return_value = controller
    executor.config = {"workflow": "wf"}

    result = executor.initialize_controller("controller-1", _context())

    assert result is controller
    controller.set_communicator.assert_called_once()
    assert controller.config == executor.config
    controller.initialize.assert_called_once()


def test_handle_event_reports_status_finalizes_and_forwards_fatal_error():
    executor = _executor()
    executor.workflow_id = "wf"
    executor.current_status.timestamp = 1.0
    fl_ctx = _context()
    fl_ctx.set_prop(Constant.STATUS_REPORTS, {"wf": {"old": True}}, private=False, sticky=False)

    executor.handle_event(EventType.BEFORE_PULL_TASK, fl_ctx)
    assert Constant.STATUS_REPORTS in fl_ctx.props
    assert fl_ctx.get_prop(Constant.STATUS_REPORTS)["wf"][Constant.TIMESTAMP] == 1.0

    executor.finalize = MagicMock()
    executor.handle_event(EventType.ABORT_TASK, fl_ctx)
    assert executor.asked_to_stop
    executor.finalize.assert_called_once_with(fl_ctx)

    executor.is_starting_client = True
    executor.handle_event(EventType.FATAL_SYSTEM_ERROR, fl_ctx)
    assert executor.fatal_system_error
    executor.fire_fed_event.assert_called_once()


def test_finalize_is_idempotent():
    executor = _executor()
    executor.workflow_id = "wf"
    fl_ctx = _context()

    executor.finalize(fl_ctx)
    executor.finalize(fl_ctx)

    assert executor.workflow_done
    executor.fire_event.assert_called_once_with(Constant.EXECUTOR_FINALIZED, fl_ctx)


def test_execute_configures_workflow_and_rejects_bad_tasks():
    executor = _executor()
    executor.engine = MagicMock()
    fl_ctx = _context()
    config = Shareable({Constant.CONFIG: {FLContextKey.WORKFLOW: "wf"}})

    reply = executor.execute(executor.configure_task_name, config, fl_ctx, Signal())
    assert reply.get_return_code() == ReturnCode.OK
    assert executor.workflow_id == "wf"
    executor.engine.register_aux_message_handler.assert_called_once()

    missing = Shareable({Constant.CONFIG: {}})
    assert executor.execute(executor.configure_task_name, missing, fl_ctx, Signal()).get_return_code() == (
        ReturnCode.BAD_REQUEST_DATA
    )
    assert executor.execute("unknown", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN

    executor.workflow_done = True
    assert (
        executor.execute(executor.start_task_name, Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.ERROR
    )


def test_execute_runs_controllers_and_broadcasts_final_result():
    executor = _executor()
    executor.engine = MagicMock()
    executor.workflow_id = "wf"
    controller = MagicMock()
    controller.name = "controller"
    controller.persistor = _Persistor()
    executor.initialize_controller = MagicMock(return_value=controller)
    executor.broadcast_final_result = MagicMock()

    reply = executor.execute(executor.start_task_name, Shareable(), _context(), Signal())

    assert reply.get_return_code() == ReturnCode.OK
    controller.control_flow.assert_called_once()
    controller.stop_controller.assert_called_once()
    executor.broadcast_final_result.assert_called_once()
    assert executor.current_status.all_done


def test_execute_handles_stop_abort_and_controller_exception():
    executor = _executor()
    executor.engine = MagicMock()
    executor.asked_to_stop = True
    assert (
        executor.execute(executor.start_task_name, Shareable(), _context(), Signal()).get_return_code() == ReturnCode.OK
    )

    executor.asked_to_stop = False
    controller = MagicMock()
    controller.name = "controller"
    executor.initialize_controller = MagicMock(return_value=controller)
    abort_signal = Signal()
    abort_signal.trigger(True)
    assert executor.execute(executor.start_task_name, Shareable(), _context(), abort_signal).get_return_code() == (
        ReturnCode.TASK_ABORTED
    )

    abort_signal = Signal()
    controller.control_flow.side_effect = RuntimeError("failed")
    with patch.object(executor, "broadcast_final_result"):
        executor.execute(executor.start_task_name, Shareable(), _context(), abort_signal)
    executor.system_panic.assert_called_once()


def test_status_updates_preserve_completion_and_latest_round():
    executor = _executor()
    assert executor._get_status_report() is None
    executor.update_status(last_round=2, action="training", error="warning", all_done=True)
    executor.update_status(last_round=1, all_done=False)
    report = executor._get_status_report()
    assert report.last_round == 2
    assert report.action == "training"
    assert report.error == "warning"
    assert report.all_done


def test_process_final_result_validates_and_persists():
    executor = _executor()
    fl_ctx = _context()
    assert executor._process_final_result(Shareable(), fl_ctx).get_return_code() == ReturnCode.BAD_REQUEST_DATA

    peer_ctx = _context()
    fl_ctx.set_peer_context(peer_ctx)
    assert executor._process_final_result(Shareable(), fl_ctx).get_return_code() == ReturnCode.BAD_REQUEST_DATA

    bad = Shareable({Constant.RESULT: object()})
    assert executor._process_final_result(bad, fl_ctx).get_return_code() == ReturnCode.BAD_REQUEST_DATA

    model = Learnable({"weights": 2})
    request = Shareable({Constant.RESULT: model})
    assert executor._process_final_result(request, fl_ctx).get_return_code() == ReturnCode.OK
    assert fl_ctx.get_prop("global_model") is model

    executor.persistor = _Persistor()
    assert executor._process_final_result(request, fl_ctx).get_return_code() == ReturnCode.OK
    assert executor.persistor.saved == [model]


def test_end_workflow_and_task_security():
    executor = _executor()
    executor.workflow_id = "wf"
    executor.config = {Constant.PRIVATE_P2P: True}
    fl_ctx = _context()
    fl_ctx.set_prop(FLContextKey.SECURE_MODE, True, private=True, sticky=False)
    assert executor.is_task_secure(fl_ctx)

    executor.finalize = MagicMock()
    reply = executor._process_end_workflow("topic", Shareable(), fl_ctx)
    assert reply.get_return_code() == ReturnCode.OK
    assert executor.asked_to_stop
    executor.finalize.assert_called_once()


def test_broadcast_final_result_validates_targets_and_replies():
    executor = _executor()
    executor.me = "site-1"
    executor.controller = MagicMock()
    fl_ctx = _context()
    model = Learnable({"weights": 1})

    executor.config = {Constant.RESULT_CLIENTS: "site-2"}
    assert executor.broadcast_final_result(model, fl_ctx) is None

    executor.config = {Constant.RESULT_CLIENTS: ["site-1"]}
    assert executor.broadcast_final_result(model, fl_ctx) is None

    executor.config = {Constant.RESULT_CLIENTS: ["site-1", "site-2", "site-3"]}
    executor.controller.broadcast_and_wait.return_value = object()
    assert executor.broadcast_final_result(model, fl_ctx) is None

    executor.controller.broadcast_and_wait.return_value = {
        "site-2": make_reply(ReturnCode.OK),
        "site-3": make_reply(ReturnCode.EXECUTION_EXCEPTION),
    }
    assert executor.broadcast_final_result(model, fl_ctx) == 1

    executor.controller.broadcast_and_wait.return_value = {
        "site-2": make_reply(ReturnCode.OK),
        "site-3": make_reply(ReturnCode.OK),
    }
    assert executor.broadcast_final_result(model, fl_ctx) == 0
