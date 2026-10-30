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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.edge.executors import hug as hug_module
from nvflare.edge.executors.hug import HierarchicalUpdateGatherer, TaskInfo
from nvflare.edge.updater import Updater
from nvflare.fuel.utils import waiter_utils
from tests.timing_utils import ManualClock, isolate_time


@pytest.fixture
def task_loop(monkeypatch):
    clock = ManualClock()
    local_time = isolate_time(monkeypatch, hug_module, waiter_utils)
    monkeypatch.setattr(local_time, "time", clock.time)
    monkeypatch.setattr(local_time, "sleep", clock.advance)
    monkeypatch.setattr(hug_module.random, "uniform", lambda low, high: 0.25)
    gatherer = HierarchicalUpdateGatherer("", "", 1)
    gatherer._parent_name = "parent"
    gatherer._task_start_time = clock.time()
    task = Shareable()
    task.set_header(ReservedKey.TASK_ID, "task-1")
    task.set_header(ReservedKey.TASK_NAME, "train")
    task.set_header(EdgeTaskHeaderKey.TASK_SEQ, 1)
    gatherer._pending_task = TaskInfo(task)
    gatherer._updater = Mock(spec=Updater)
    gatherer._updater.start_task.side_effect = lambda task, ctx: task
    gatherer._updater.prepare_update_for_parent.side_effect = lambda ctx: Shareable({"value": 1})
    ctx = FLContext()
    engine = Mock()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    engine.send_aux_request.return_value = {"parent": make_reply(ReturnCode.OK)}
    return gatherer, ctx, engine, clock, Signal(), local_time


@pytest.mark.parametrize("interval", [None, 0.05, 0.4, 1.0, 3.0])
@pytest.mark.parametrize("jitter", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("num_children", [0, 2])
def test_update_requests_respect_interval(task_loop, monkeypatch, interval, jitter, num_children):
    gatherer, ctx, engine, clock, abort, _ = task_loop
    if interval is not None:
        gatherer._pending_task.task.set_header(EdgeTaskHeaderKey.UPDATE_INTERVAL, interval)
        gatherer._pending_task = TaskInfo(gatherer._pending_task.task)
    gatherer._num_children = num_children
    monkeypatch.setattr(hug_module.random, "uniform", lambda low, high: jitter)
    sent = []

    def send(**kwargs):
        sent.append(clock.time())
        assert kwargs["targets"] == "parent"
        assert kwargs["topic"] == "SAGE__train_update"
        assert kwargs["request"].get_header(EdgeTaskHeaderKey.TASK_SEQ) == 1
        rc = ReturnCode.TASK_ABORTED if len(sent) == 3 else ReturnCode.OK
        return {"parent": make_reply(rc)}

    engine.send_aux_request.side_effect = send
    result = gatherer._do_task(ctx, abort)

    assert result["value"] == 1
    assert len(sent) == 3
    assert sent[0] == 1000.0  # The initial report is still sent immediately.
    requested_wait = (1.0 if interval is None else interval) + jitter
    for earlier, later in zip(sent, sent[1:]):
        assert requested_wait - 1e-8 <= later - earlier <= requested_wait + 0.10000001
    assert gatherer._updater.process_parent_update_reply.call_count == 2


@pytest.mark.parametrize("stop", ["set_task_done", "parent_end", "abort"])
def test_completion_and_abort_interrupt_long_wait(task_loop, stop):
    gatherer, ctx, engine, clock, abort, local_time = task_loop
    gatherer._pending_task.update_interval = 10.0

    def sleep(delay):
        clock.advance(delay)
        if clock.time() >= 1000.35:
            if stop == "abort":
                abort.trigger(True)
            elif stop == "set_task_done":
                assert gatherer.set_task_done("task-1", ctx)
            else:
                end = Shareable()
                end.set_header(EdgeTaskHeaderKey.TASK_SEQ, 1)
                gatherer._process_task_end("SAGE__train_end", end, ctx)

    local_time.sleep = sleep
    result = gatherer._do_task(ctx, abort)

    assert clock.time() == pytest.approx(1000.4)
    assert engine.send_aux_request.call_count == 1
    assert result.get_return_code() == (ReturnCode.TASK_ABORTED if stop == "abort" else ReturnCode.OK)


@pytest.mark.parametrize("stop", ["parent_abort", "next_task", "children_done"])
def test_parent_and_child_completion_do_not_wait(task_loop, stop):
    gatherer, ctx, engine, clock, abort, _ = task_loop
    reply = make_reply(ReturnCode.TASK_ABORTED if stop == "parent_abort" else ReturnCode.OK)
    if stop == "next_task":
        reply.set_header(EdgeTaskHeaderKey.TASK_SEQ, 2)
    if stop == "children_done":
        gatherer._num_children = 2
        gatherer._pending_clients = {"child-1": 999.0, "child-2": 999.0}
    engine.send_aux_request.return_value = {"parent": reply}

    result = gatherer._do_task(ctx, abort)

    assert result["value"] == 1
    assert clock.time() == 1000.0
    assert engine.send_aux_request.call_count == 1


@pytest.mark.parametrize("replies", [{}, {"parent": "invalid"}])
def test_invalid_parent_reply_still_fails_promptly(task_loop, replies):
    gatherer, ctx, engine, clock, abort, _ = task_loop
    engine.send_aux_request.return_value = replies

    result = gatherer._do_task(ctx, abort)

    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert clock.time() == 1000.0
    assert engine.send_aux_request.call_count == 1


def test_final_report_includes_updates_received_during_wait(task_loop):
    gatherer, ctx, engine, clock, abort, local_time = task_loop
    data = {"value": 1}
    gatherer._updater.prepare_update_for_parent.side_effect = lambda ctx: Shareable(data)

    def sleep(delay):
        clock.advance(delay)
        data["value"] = 2
        gatherer.set_task_done("task-1", ctx)

    local_time.sleep = sleep
    result = gatherer._do_task(ctx, abort)

    assert engine.send_aux_request.call_args.kwargs["request"]["value"] == 1
    assert result["value"] == 2
    assert engine.send_aux_request.call_count == 1


def test_execute_resets_completion_for_next_task(task_loop):
    gatherer, ctx, engine, clock, abort, local_time = task_loop
    task = gatherer._pending_task.task

    def sleep(delay):
        clock.advance(delay)
        gatherer.set_task_done("task-1", ctx)

    local_time.sleep = sleep
    for _ in range(2):
        result = gatherer.execute("train", task, ctx, abort)
        assert result["value"] == 1
        assert gatherer._pending_task is None
        assert gatherer._task_done is False
    assert engine.send_aux_request.call_count == 2
    assert gatherer._updater.end_task.call_count == 2
