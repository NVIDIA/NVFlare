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

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.ipc_exchanger import IPCExchanger, _TaskContext
from nvflare.client.ipc import defs
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.message import Message


def _exchanger(**kwargs):
    exchanger = IPCExchanger(**kwargs)
    exchanger.log_debug = MagicMock()
    exchanger.log_info = MagicMock()
    exchanger.log_warning = MagicMock()
    exchanger.log_error = MagicMock()
    exchanger.system_panic = MagicMock()
    return exchanger


def _context(engine=None):
    fl_ctx = FLContext()
    if engine:
        fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=False)
    return fl_ctx


def _cell_reply(rc=CellReturnCode.OK, payload=None):
    return Message(headers={MessageHeaderKey.RETURN_CODE: rc}, payload=payload)


def test_task_context_tracks_state_and_formats_name():
    context = _TaskContext("train", "task-1", FLContext())
    assert str(context) == "'train task-1'"
    assert context.send_rc is None
    assert not context.result_waiter.is_set()


def test_start_run_validates_agent_id_and_registers_handler():
    engine = MagicMock()
    cell = MagicMock()
    engine.get_cell.return_value = cell
    fl_ctx = _context(engine)
    exchanger = _exchanger()

    exchanger.handle_event(EventType.START_RUN, fl_ctx)
    exchanger.system_panic.assert_called_once()

    exchanger.system_panic.reset_mock()
    fl_ctx.set_prop(FLContextKey.JOB_META, {defs.JOB_META_KEY_AGENT_ID: 1}, private=True, sticky=False)
    exchanger.handle_event(EventType.START_RUN, fl_ctx)
    exchanger.system_panic.assert_called_once()

    exchanger.system_panic.reset_mock()
    fl_ctx.set_prop(FLContextKey.JOB_META, {defs.JOB_META_KEY_AGENT_ID: "agent"}, private=True, sticky=False)
    with patch("nvflare.app_common.executors.ipc_exchanger.threading.Thread") as thread_cls:
        exchanger.handle_event(EventType.START_RUN, fl_ctx)
    assert exchanger.agent_id == "agent"
    assert exchanger.flare_agent_fqcn == "site-1.-agent"
    cell.register_request_cb.assert_called()
    thread_cls.return_value.start.assert_called_once()


def test_end_run_says_goodbye_and_handles_error_reply():
    exchanger = _exchanger(agent_id="agent")
    exchanger.cell = MagicMock()
    exchanger.flare_agent_fqcn = "site-1.-agent"
    exchanger.cell.send_request.return_value = _cell_reply(CellReturnCode.TIMEOUT)

    exchanger.handle_event(EventType.END_RUN, _context())

    assert exchanger.is_done
    exchanger.cell.send_request.assert_called_once()
    exchanger.cell.send_request.return_value = None
    exchanger._say_goodbye()


def test_monitor_connects_on_heartbeat_and_stops_cleanly():
    exchanger = _exchanger(agent_id="agent", agent_heartbeat_interval=1.0)
    cell = Cell.__new__(Cell)
    cell.send_request = MagicMock(return_value=_cell_reply())
    exchanger.cell = cell
    exchanger.flare_agent_fqcn = "site-1.-agent"

    def stop_after_sleep(_):
        exchanger.is_done = True

    with patch("nvflare.app_common.executors.ipc_exchanger.time.time", side_effect=[2.0, 2.1, 2.2]):
        with patch("nvflare.app_common.executors.ipc_exchanger.time.sleep", side_effect=stop_after_sleep):
            exchanger._monitor()

    assert exchanger.is_connected
    assert exchanger.last_agent_ack_time == 2.1


def test_monitor_panics_when_agent_heartbeat_expires():
    exchanger = _exchanger(
        agent_id="agent", agent_heartbeat_interval=1.0, agent_connection_timeout=2.0, agent_heartbeat_timeout=3.0
    )
    cell = Cell.__new__(Cell)
    cell.send_request = MagicMock(return_value=_cell_reply(CellReturnCode.TIMEOUT))
    exchanger.cell = cell
    exchanger.engine = MagicMock()
    exchanger.engine.new_context.return_value = nullcontext(FLContext())
    exchanger.flare_agent_fqcn = "site-1.-agent"
    exchanger.last_agent_ack_time = 0.0
    exchanger.is_connected = True

    with patch("nvflare.app_common.executors.ipc_exchanger.time.time", side_effect=[10.0, 10.0]):
        exchanger._monitor()

    assert exchanger.is_done
    assert not exchanger.is_connected
    exchanger.system_panic.assert_called_once()


def test_execute_rejects_overlap_abort_and_delegates_connected_task():
    exchanger = _exchanger(agent_id="agent")
    fl_ctx = _context()
    exchanger.task_ctx = _TaskContext("old", "old-task", fl_ctx)
    assert exchanger.execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.BAD_REQUEST_DATA

    exchanger.task_ctx = None
    exchanger.is_done = True
    assert exchanger.execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.TASK_ABORTED

    exchanger.is_done = False
    exchanger.is_connected = True
    exchanger._do_execute = MagicMock(return_value=Shareable({"result": 1}))
    data = Shareable()
    data.set_header(FLContextKey.TASK_ID, "task-1")
    result = exchanger.execute("train", data, fl_ctx, Signal())
    assert result["result"] == 1
    assert exchanger.task_ctx is None


def test_send_task_handles_abort_result_success_and_rejection():
    exchanger = _exchanger(agent_id="agent", resend_task_interval=0.0)
    exchanger.cell = MagicMock()
    exchanger.flare_agent_fqcn = "site-1.-agent"
    fl_ctx = _context()
    context = _TaskContext("train", "task-1", fl_ctx)
    exchanger._ask_agent_to_abort_task = MagicMock()
    exchanger.is_done = True
    exchanger._send_task(context, Message(), Signal())
    assert context.send_rc == ReturnCode.TASK_ABORTED
    exchanger._ask_agent_to_abort_task.assert_called_once()

    exchanger.is_done = False
    context = _TaskContext("train", "task-1", fl_ctx)
    context.result_received_time = 1.0
    exchanger._send_task(context, Message(), Signal())
    assert context.send_rc == ReturnCode.OK

    exchanger.is_connected = True
    context = _TaskContext("train", "task-1", fl_ctx)
    exchanger.cell.send_request.return_value = _cell_reply(CellReturnCode.OK)
    exchanger._send_task(context, Message(), Signal())
    assert context.send_rc == ReturnCode.OK

    context = _TaskContext("train", "task-1", fl_ctx)
    exchanger.cell.send_request.return_value = _cell_reply(CellReturnCode.INVALID_REQUEST)
    exchanger._send_task(context, Message(), Signal())
    assert context.send_rc == ReturnCode.BAD_REQUEST_DATA


def test_do_execute_rejects_bad_data_and_send_failure():
    exchanger = _exchanger(agent_id="agent")
    fl_ctx = _context()
    exchanger.task_ctx = _TaskContext("train", "task-1", fl_ctx)
    assert exchanger._do_execute("train", Shareable(), fl_ctx, Signal()).get_return_code() == ReturnCode.BAD_TASK_DATA

    data = DXO(DataKind.WEIGHTS, {"weight": 1}).to_shareable()
    exchanger._send_task = MagicMock(
        side_effect=lambda context, message, signal: setattr(context, "send_rc", ReturnCode.TASK_ABORTED)
    )
    assert exchanger._do_execute("train", data, fl_ctx, Signal()).get_return_code() == ReturnCode.TASK_ABORTED


@pytest.mark.parametrize("data_kind", [DataKind.WEIGHTS, DataKind.APP_DEFINED])
def test_do_execute_builds_task_and_converts_result(data_kind):
    exchanger = _exchanger(agent_id="agent")
    fl_ctx = _context()
    context = _TaskContext("train", "task-1", fl_ctx)
    exchanger.task_ctx = context
    input_data = DXO(data_kind, {"input": 1}, {"source": "test"}).to_shareable()
    input_data.set_header(AppConstants.CURRENT_ROUND, 2)
    input_data.set_header(AppConstants.NUM_ROUNDS, 5)

    def finish(task_ctx, message, signal):
        task_ctx.send_rc = ReturnCode.OK
        task_ctx.result_rc = defs.RC.OK
        task_ctx.result = {
            defs.PayloadKey.DATA: {"output": 2},
            defs.PayloadKey.META: {defs.MetaKey.DATA_KIND: DataKind.METRICS},
        }
        task_ctx.result_waiter.set()

    exchanger._send_task = finish
    result = exchanger._do_execute("train", input_data, fl_ctx, Signal())
    dxo = from_shareable(result)
    assert dxo.data == {"output": 2}
    assert dxo.data_kind == (DataKind.APP_DEFINED if data_kind == DataKind.APP_DEFINED else DataKind.METRICS)
    assert result.get_return_code() == defs.RC.OK


def test_do_execute_returns_agent_error_result():
    exchanger = _exchanger(agent_id="agent")
    context = _TaskContext("train", "task-1", _context())
    exchanger.task_ctx = context

    def finish(task_ctx, message, signal):
        task_ctx.send_rc = ReturnCode.OK
        task_ctx.result_rc = ReturnCode.EXECUTION_EXCEPTION
        task_ctx.result_waiter.set()

    exchanger._send_task = finish
    result = exchanger._do_execute(
        "train", DXO(DataKind.WEIGHTS, {"input": 1}).to_shareable(), context.fl_ctx, Signal()
    )
    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION


def test_abort_request_and_finish_result():
    exchanger = _exchanger(agent_id="agent")
    exchanger.cell = MagicMock()
    exchanger.flare_agent_fqcn = "site-1.-agent"
    exchanger._ask_agent_to_abort_task("train", "task-1")
    message = exchanger.cell.fire_and_forget.call_args.kwargs["message"]
    assert message.get_header(defs.MsgHeader.TASK_ID) == "task-1"

    context = _TaskContext("train", "task-1", _context())
    assert IPCExchanger._finish_result(context, "ok", {"value": 1}).get_header(MessageHeaderKey.RETURN_CODE) == (
        CellReturnCode.OK
    )
    assert context.result_waiter.is_set()
    assert IPCExchanger._finish_result(context, result_is_valid=False).get_header(MessageHeaderKey.RETURN_CODE) == (
        CellReturnCode.INVALID_REQUEST
    )


def test_receive_result_validates_context_task_and_payload():
    exchanger = _exchanger(agent_id="agent")
    exchanger.logger.error = MagicMock()
    no_task = Message(headers={MessageHeaderKey.ORIGIN: "agent"})
    assert exchanger._receive_result(no_task).get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK

    unexpected = Message(headers={MessageHeaderKey.ORIGIN: "agent", defs.MsgHeader.TASK_ID: "task-1"})
    assert exchanger._receive_result(unexpected).get_header(MessageHeaderKey.RETURN_CODE) == (
        CellReturnCode.INVALID_REQUEST
    )

    context = _TaskContext("train", "task-1", _context())
    exchanger.task_ctx = context
    wrong = Message(headers={MessageHeaderKey.ORIGIN: "agent", defs.MsgHeader.TASK_ID: "other"})
    assert exchanger._receive_result(wrong).get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.INVALID_REQUEST

    duplicate = Message(headers={MessageHeaderKey.ORIGIN: "agent", defs.MsgHeader.TASK_ID: "task-1"})
    context.result_received_time = 1.0
    assert exchanger._receive_result(duplicate).get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK

    for payload in ("bad", {}, {defs.PayloadKey.DATA: {"value": 1}}):
        context.result_received_time = None
        request = Message(headers={MessageHeaderKey.ORIGIN: "agent", defs.MsgHeader.TASK_ID: "task-1"}, payload=payload)
        assert exchanger._receive_result(request).get_header(MessageHeaderKey.RETURN_CODE) == (
            CellReturnCode.INVALID_REQUEST
        )


def test_receive_result_accepts_valid_payload_and_restart_without_task_id():
    exchanger = _exchanger(agent_id="agent")
    context = _TaskContext("train", "task-1", _context())
    exchanger.task_ctx = context
    payload = {defs.PayloadKey.DATA: {"value": 1}, defs.PayloadKey.META: {defs.MetaKey.DATA_KIND: DataKind.WEIGHTS}}
    request = Message(headers={MessageHeaderKey.ORIGIN: "agent", defs.MsgHeader.RC: defs.RC.OK}, payload=payload)
    reply = exchanger._receive_result(request)
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK
    assert context.result == payload
    assert context.result_rc == defs.RC.OK
