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

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_constant import ReturnCode, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_client import WFCommClient
from nvflare.apis.shareable import Shareable, make_reply
from tests.unit_test.fl_context_helper import make_fl_context


def _context(engine):
    return make_fl_context(engine=engine)


def _engine():
    engine = MagicMock()
    engine.all_clients = [Client("site-1", "token-1"), Client("site-2", "token-2")]
    engine.validate_targets.return_value = (engine.all_clients, [])
    return engine


def _ok_reply(peer_name):
    reply = make_reply(ReturnCode.OK)
    peer_ctx = FLContext()
    peer_ctx.set_prop("peer", peer_name, private=False, sticky=False)
    reply.set_peer_context(peer_ctx)
    return reply


def test_broadcast_and_wait_processes_callbacks_filters_and_replies():
    engine = _engine()
    engine.send_aux_request.return_value = {
        "site-1": _ok_reply("site-1"),
        "site-2": make_reply(ReturnCode.EXECUTION_EXCEPTION),
    }
    engine.send_aux_request.return_value["site-2"].set_peer_context(FLContext())
    callbacks = {name: MagicMock() for name in ("before", "after", "result", "done")}
    task = Task(
        "train",
        Shareable(),
        timeout=0,
        before_task_sent_cb=callbacks["before"],
        after_task_sent_cb=callbacks["after"],
        result_received_cb=callbacks["result"],
        task_done_cb=callbacks["done"],
        secure=True,
    )
    comm = WFCommClient(max_task_timeout=30)
    comm.fire_event = MagicMock()
    comm.fire_event_with_data = MagicMock()

    with patch("nvflare.apis.impl.wf_comm_client.apply_filters", side_effect=lambda *args: args[1]):
        with patch("nvflare.apis.impl.wf_comm_client.delete_msg_root") as delete_msg_root:
            replies = comm.broadcast(task, _context(engine), targets=engine.all_clients)

    assert task.timeout == 30
    assert set(replies) == {"site-1", "site-2"}
    assert replies["site-1"].get_return_code() == ReturnCode.OK
    assert replies["site-2"].get_return_code() == ReturnCode.ERROR
    assert callbacks["before"].call_count == 2
    assert callbacks["after"].call_count == 2
    callbacks["result"].assert_called_once()
    callbacks["done"].assert_called_once()
    delete_msg_root.assert_called_once()
    assert engine.send_aux_request.call_args.kwargs["secure"] is True


def test_broadcast_uses_all_clients_and_rejects_invalid_targets():
    engine = _engine()
    engine.send_aux_request.return_value = {}
    comm = WFCommClient()
    comm.fire_event = MagicMock()
    comm.fire_event_with_data = MagicMock()

    with patch("nvflare.apis.impl.wf_comm_client.apply_filters", side_effect=lambda *args: args[1]):
        comm.broadcast_and_wait(Task("train", Shareable(), timeout=1), _context(engine), targets=None)

    assert len(engine.send_aux_request.call_args.kwargs["targets"]) == 2

    engine.validate_targets.return_value = ([], ["missing"])
    with patch("nvflare.apis.impl.wf_comm_client.apply_filters", side_effect=lambda *args: args[1]):
        with pytest.raises(ValueError, match="invalid target"):
            comm.broadcast_and_wait(Task("train", Shareable(), timeout=1), _context(engine), ["missing"])


def test_broadcast_returns_filter_error_replies():
    engine = _engine()
    comm = WFCommClient()
    comm.log_exception = MagicMock()
    comm.fire_event_with_data = MagicMock()

    with patch("nvflare.apis.impl.wf_comm_client.apply_filters", side_effect=RuntimeError("filter failed")):
        replies = comm.broadcast_and_wait(Task("train", Shareable()), _context(engine), ["site-1"])

    assert replies["site-1"].get_return_code() == ReturnCode.TASK_DATA_FILTER_ERROR


def test_broadcast_handles_result_filter_and_task_done_errors():
    engine = _engine()
    engine.send_aux_request.return_value = {"site-1": _ok_reply("site-1")}
    comm = WFCommClient()
    comm.log_exception = MagicMock()
    comm.fire_event = MagicMock()
    comm.fire_event_with_data = MagicMock()
    task = Task("train", Shareable(), timeout=1, task_done_cb=MagicMock(side_effect=RuntimeError("done")))

    calls = 0

    def filter_data(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("result filter failed")
        return args[1]

    with patch("nvflare.apis.impl.wf_comm_client.apply_filters", side_effect=filter_data):
        replies = comm.broadcast_and_wait(task, _context(engine), ["site-1"])

    assert replies["site-1"].get_return_code() == ReturnCode.ERROR
    assert task.completion_status == TaskCompletionStatus.ERROR
    assert isinstance(task.exception, RuntimeError)


@pytest.mark.parametrize("callback_name", ["before_task_sent_cb", "after_task_sent_cb", "result_received_cb"])
def test_callback_errors_mark_task_failed(callback_name):
    engine = _engine()
    engine.send_aux_request.return_value = {"site-1": _ok_reply("site-1")}
    kwargs = {callback_name: MagicMock(side_effect=RuntimeError("callback failed"))}
    task = Task("train", Shareable(), timeout=1, **kwargs)
    comm = WFCommClient()
    comm.log_exception = MagicMock()
    comm.fire_event = MagicMock()
    comm.fire_event_with_data = MagicMock()

    with patch("nvflare.apis.impl.wf_comm_client.apply_filters", side_effect=lambda *args: args[1]):
        replies = comm.broadcast_and_wait(task, _context(engine), ["site-1"])

    assert replies["site-1"].get_return_code() == ReturnCode.ERROR
    assert task.completion_status == TaskCompletionStatus.ERROR


def test_negative_timeout_is_rejected_after_task_setup():
    engine = _engine()
    task = Task("train", Shareable(), timeout=1)
    task.timeout = -1
    comm = WFCommClient()
    comm.fire_event_with_data = MagicMock()

    with patch("nvflare.apis.impl.wf_comm_client.apply_filters", side_effect=lambda *args: args[1]):
        with pytest.raises(ValueError, match="timeout must"):
            comm.broadcast_and_wait(task, _context(engine), ["site-1"])


def test_client_lookup_error_replies_and_callback_helpers():
    engine = _engine()
    comm = WFCommClient()
    task = Task("train", Shareable())
    client_task = ClientTask(engine.all_clients[0], task)
    task.client_tasks.append(client_task)

    assert comm._get_client(engine.all_clients[0], engine) is engine.all_clients[0]
    assert comm._get_client(SiteType.SERVER, engine).name == SiteType.SERVER
    assert comm._get_client("site-2", engine).name == "site-2"
    assert comm._get_client("missing", engine) is None
    assert comm._get_client_task(engine.all_clients[0], task) is client_task
    assert comm._make_error_reply(ReturnCode.ERROR, ["site-1"])["site-1"].get_return_code() == ReturnCode.ERROR

    comm.log_exception = MagicMock()
    callback = MagicMock(side_effect=RuntimeError("failed"))
    assert comm._call_task_cb(callback, engine.all_clients[0], task, FLContext())
    assert task.completion_status == TaskCompletionStatus.ERROR


def test_send_and_relay_delegate_in_order():
    engine = _engine()
    fl_ctx = _context(engine)
    task = Task("train", Shareable())
    comm = WFCommClient()
    comm._validate_target = MagicMock()
    ok = make_reply(ReturnCode.OK)
    comm.broadcast_and_wait = MagicMock(
        side_effect=[{"site-1": make_reply(ReturnCode.ERROR)}, {"site-2": ok}],
    )

    assert comm.send(task, fl_ctx, ["site-1", "site-2"]) == {"site-2": ok}
    assert comm.broadcast_and_wait.call_count == 2

    comm.broadcast_and_wait.reset_mock()
    error_replies = [{"site-1": make_reply(ReturnCode.ERROR)}, {"site-2": make_reply(ReturnCode.ERROR)}]
    comm.broadcast_and_wait.side_effect = error_replies
    failed = comm.send(task, fl_ctx, ["site-1", "site-2"])
    assert failed["site-2"].get_return_code() == ReturnCode.ERROR

    comm.broadcast_and_wait.reset_mock()
    comm.broadcast_and_wait.side_effect = [{"site-1": ok}, {"site-2": ok}]
    assert comm.relay(task, fl_ctx, ["site-1", "site-2"]) == {"site-1": ok, "site-2": ok}


def test_validate_target_rejects_empty_and_unknown_targets():
    comm = WFCommClient()
    engine = _engine()
    with pytest.raises(ValueError, match="Must provide a target"):
        comm._validate_target(engine, [])

    engine.validate_targets.return_value = ([], ["missing"])
    with pytest.raises(ValueError, match="invalid target"):
        comm._validate_target(engine, ["missing"])
