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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReservedTopic, TaskResultReceipt
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.utils.event import fire_event_to_components
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.edge.executors.hug import HierarchicalUpdateGatherer, TaskInfo
from nvflare.fuel.f3.cellnet.core_cell import MessageHeaderKey, ReturnCode
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef
from nvflare.private.defs import CellMessageHeaderKeys, new_cell_message
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from nvflare.private.fed.client.client_runner import (
    _TASK_CHECK_RESULT_OK,
    _TASK_CHECK_RESULT_TASK_GONE,
    ClientRunner,
    ClientRunnerConfig,
    TaskRouter,
)
from nvflare.private.fed.client.communicator import Communicator


@pytest.fixture
def hierarchy(monkeypatch):
    engine = Mock()
    router = TaskRouter()
    gatherer = HierarchicalUpdateGatherer("learner", "updater", 5.0)
    gatherer._updater = Mock()
    # The aggregator may decline the update; the child must still stop uploading.
    gatherer._updater.process_child_update.return_value = (False, Shareable())
    router.add_executor(["train"], gatherer)
    runner = ClientRunner({}, ClientRunnerConfig(router, {}, {}), "job-1", engine)
    contexts = FLContextManager(engine=engine, identity_name="parent", job_id="job-1")
    with contexts.new_context() as ctx:
        ctx.set_prop(FLContextKey.RUNNER, runner, private=True)
    engine.new_context.side_effect = contexts.new_context
    engine.fire_event.side_effect = lambda event, ctx: fire_event_to_components(event, [runner], ctx)
    parent = Communicator(client_config={"client_name": "parent"})
    parent.engine = engine
    data = Shareable({"model": 1})
    for key, value in (
        (ReservedHeaderKey.TASK_ID, "task-1"),
        (ReservedHeaderKey.TASK_NAME, "train"),
        (ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt-1"),
        (ReservedHeaderKey.TASK_ATTEMPT_REQUIRED, True),
        (ReservedHeaderKey.WORKFLOW, "workflow"),
    ):
        data.set_header(key, value)
        data.add_cookie(key, value)
    data.set_header(ReservedKey.TASK_IS_READY, True)
    parent.pending_task = data
    runner.running_tasks["task-1"] = TaskAssignment("train", "task-1", data)
    gatherer._pending_task = TaskInfo(data)
    monkeypatch.setattr("nvflare.private.fed.client.communicator.determine_parent_fqcn", lambda *_: "parent")

    def child(name="child"):
        ctx = FLContextManager(identity_name=name, job_id="job-1").new_context()
        req = Shareable()
        req.set_peer_context(ctx)
        headers = {CellMessageHeaderKeys.CLIENT_NAME: name}
        task = parent._process_get_task(new_cell_message(headers, req)).payload
        assert task.get_task_attempt_id() == "attempt-1"
        assert task.get_header(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED) is True
        result = Shareable({"model": 2})
        result.set_cookie_jar(dict(task.get_cookie_jar()))
        result.set_header(ReservedHeaderKey.TASK_ID, "task-1")
        result.set_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA, True)
        result.set_peer_context(ctx)
        child_engine = Mock()
        child_runner = ClientRunner({}, ClientRunnerConfig(TaskRouter(), {}, {}), "job-1", child_engine)
        child_runner.parent_target = "parent"
        child_runner.task_check_interval = 0
        client = Communicator(client_config={"client_name": name})
        client.cell = Mock()
        for key, value in (
            (FLContextKey.TASK_ID, "task-1"),
            (FLContextKey.TASK_ATTEMPT_ID, "attempt-1"),
            (FLContextKey.TASK_NAME, "train"),
            (FLContextKey.WORKFLOW, "workflow"),
            (FLContextKey.SSID, "session"),
        ):
            ctx.set_prop(key, value, private=True, sticky=False)

        def check(**kwargs):
            with contexts.new_context() as parent_ctx:
                parent_ctx.set_peer_context(ctx)
                return {"parent": runner._handle_task_check(ReservedTopic.TASK_CHECK, kwargs["request"], parent_ctx)}

        def transport(**kwargs):
            message = kwargs["request"]
            message.set_header(MessageHeaderKey.PAYLOAD_LEN, 0)
            message.set_header(CellMessageHeaderKeys.CLIENT_NAME, name)
            return parent._process_submit_result(message)

        child_engine.send_aux_request.side_effect = check
        client.cell.send_request.side_effect = transport
        child_engine.send_task_result.side_effect = lambda output, fl_ctx, **_: (
            client.submit_update("project", "token", "session", fl_ctx, name, output, "train") == ReturnCode.OK
        )
        return SimpleNamespace(runner=child_runner, client=client, result=result, ctx=ctx, transport=transport)

    return SimpleNamespace(parent=parent, runner=runner, gatherer=gatherer, child=child, contexts=contexts)


def test_child_receipt_independent_of_parent_aggregation_and_replayed_on_readiness(hierarchy):
    child = hierarchy.child()
    assert child.runner._check_task_once("task-1", child.ctx) == _TASK_CHECK_RESULT_OK
    assert child.ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RETRY
    assert child.runner._send_task_result(child.result, "task-1", child.ctx) is True
    assert child.ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    # Repeated publication and repeated assignment delivery do not repeat aggregation.
    hierarchy.child()
    assert child.runner._send_task_result(child.result, "task-1", child.ctx) is True
    hierarchy.gatherer._updater.process_child_update.assert_called_once()
    child.client.cell.send_request.assert_called_once()


@pytest.mark.parametrize("close_after_receipt", [False, True])
def test_child_retry_loop_recovers_lost_ack_without_reupload(hierarchy, close_after_receipt):
    child = hierarchy.child()

    def lose_ack(**kwargs):
        child.transport(**kwargs)
        if close_after_receipt:
            with hierarchy.runner.task_lock:
                hierarchy.runner.running_tasks.pop("task-1")
        return new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.COMM_ERROR}, Shareable())

    child.client.cell.send_request.side_effect = lose_ack
    assert child.runner._send_task_result(child.result, "task-1", child.ctx) is (not close_after_receipt)
    expected = TaskResultReceipt.TASK_CLOSED if close_after_receipt else TaskResultReceipt.RECEIVED
    assert child.ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == expected
    child.client.cell.send_request.assert_called_once()
    hierarchy.gatherer._updater.process_child_update.assert_called_once()


def test_each_child_has_its_own_receipt(hierarchy):
    first, second = hierarchy.child("first"), hierarchy.child("second")
    assert first.runner._send_task_result(first.result, "task-1", first.ctx)
    assert second.runner._check_task_once("task-1", second.ctx) == _TASK_CHECK_RESULT_OK
    assert second.ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RETRY
    assert second.runner._send_task_result(second.result, "task-1", second.ctx)
    assert hierarchy.gatherer._updater.process_child_update.call_count == 2


def test_receipt_survives_event_failure_and_concurrent_duplicate(hierarchy):
    child = hierarchy.child()
    effects = []
    nested = []

    def fail_after_effect(event, ctx):
        if event == EventType.TASK_RESULT_RECEIVED:
            effects.append(ctx.get_prop(FLContextKey.TASK_RESULT))
            # Replay during event processing must not enter the event again.
            nested.append(child.transport(request=new_cell_message({}, child.result)).payload)
            raise RuntimeError("side effect already happened")

    hierarchy.parent.engine.fire_event.side_effect = fail_after_effect
    assert child.runner._send_task_result(child.result, "task-1", child.ctx)
    assert nested[0].get_task_result_receipt("task-1", "attempt-1", "workflow") == TaskResultReceipt.RECEIVED
    assert len(effects) == 1
    assert child.runner._send_task_result(child.result, "task-1", child.ctx)
    assert len(effects) == 1


@pytest.mark.parametrize("mismatch", ["attempt", "workflow", "name", "job", "child", "cookie", "stripped", "auth"])
def test_invalid_child_submission_does_not_claim_or_run_events(hierarchy, mismatch):
    child = hierarchy.child()
    if mismatch in ("attempt", "workflow", "name"):
        key = {
            "attempt": ReservedHeaderKey.TASK_ATTEMPT_ID,
            "workflow": ReservedHeaderKey.WORKFLOW,
            "name": ReservedHeaderKey.TASK_NAME,
        }[mismatch]
        child.result.add_cookie(key, "other")
        child.result.set_header(key, "other")
    elif mismatch in ("job", "child", "auth"):
        peer = FLContextManager(
            identity_name="other" if mismatch != "job" else "child", job_id="other" if mismatch == "job" else "job-1"
        ).new_context()
        child.result.set_peer_context(peer)
    elif mismatch == "cookie":
        child.result.add_cookie(ReservedHeaderKey.TASK_ID, "other")
    else:
        child.result.set_cookie_jar({})
    headers = {
        CellMessageHeaderKeys.CLIENT_NAME: (
            "child" if mismatch == "auth" else child.result.get_peer_context().get_identity_name()
        )
    }
    reply = hierarchy.parent._process_submit_result(new_cell_message(headers, child.result)).payload
    assert reply.get_header(ReservedHeaderKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
    hierarchy.gatherer._updater.process_child_update.assert_not_called()
    assert child.runner._check_task_once("task-1", child.ctx) == _TASK_CHECK_RESULT_OK
    assert child.ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RETRY


@pytest.mark.parametrize("mismatch", ["attempt", "workflow", "name", "job", "child"])
def test_invalid_child_readiness_returns_bound_closure(hierarchy, mismatch):
    child = hierarchy.child()
    key = {
        "attempt": FLContextKey.TASK_ATTEMPT_ID,
        "workflow": FLContextKey.WORKFLOW,
        "name": FLContextKey.TASK_NAME,
        "job": ReservedKey.RUN_NUM,
        "child": ReservedKey.IDENTITY_NAME,
    }[mismatch]
    child.ctx.set_prop(key, "other", private=mismatch not in ("job", "child"), sticky=mismatch == "job")
    assert child.runner._check_task_once("task-1", child.ctx) == _TASK_CHECK_RESULT_TASK_GONE
    assert child.ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED


def test_incomplete_child_payload_retries_then_confirms_complete_result(hierarchy):
    child = hierarchy.child()
    child.result["model"] = LazyDownloadRef("child.job-1", "tx", "model")
    assert (
        child.client.submit_update("project", "token", "session", child.ctx, "child", child.result, "train")
        == ReturnCode.COMM_ERROR
    )
    assert child.ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RETRY
    hierarchy.gatherer._updater.process_child_update.assert_not_called()
    child.result["model"] = 2
    assert child.runner._send_task_result(child.result, "task-1", child.ctx)
    # A replay can confirm the earlier receipt without downloading another envelope.
    child.result["model"] = LazyDownloadRef("child.job-1", "tx", "model")
    assert (
        child.client.submit_update("project", "token", "session", child.ctx, "child", child.result, "train")
        == ReturnCode.OK
    )
    hierarchy.gatherer._updater.process_child_update.assert_called_once()


@pytest.mark.parametrize("aborted", [False, True])
def test_parent_closure_stops_both_readiness_and_upload(hierarchy, aborted):
    child = hierarchy.child()
    if aborted:
        hierarchy.runner.run_abort_signal.trigger(True)
    else:
        with hierarchy.runner.task_lock:
            hierarchy.runner.running_tasks.pop("task-1")
    assert child.runner._check_task_once("task-1", child.ctx) == _TASK_CHECK_RESULT_TASK_GONE
    assert (
        child.client.submit_update("project", "token", "session", child.ctx, "child", child.result, "train")
        == ReturnCode.OK
    )
    assert child.ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
    hierarchy.gatherer._updater.process_child_update.assert_not_called()


def test_old_child_result_cannot_claim_next_parent_assignment(hierarchy):
    child = hierarchy.child()
    task = hierarchy.runner.running_tasks["task-1"]
    data = Shareable({"model": 3})
    data.set_cookie_jar({ReservedHeaderKey.WORKFLOW: "next-workflow"})
    data.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, "next-attempt")
    next_task = TaskAssignment("train", "task-1", data)
    next_task.child_result_receipts["child"] = False
    with hierarchy.runner.task_lock:
        hierarchy.runner.running_tasks["task-1"] = next_task
    assert child.runner._send_task_result(child.result, "task-1", child.ctx) is False
    assert child.ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
    assert (
        child.client.submit_update("project", "token", "session", child.ctx, "child", child.result, "train")
        == ReturnCode.OK
    )
    assert not next_task.child_result_receipts["child"]
    assert not task.child_result_receipts["child"]
    hierarchy.gatherer._updater.process_child_update.assert_not_called()


def test_child_receipt_reply_keeps_identity_when_event_consumes_payload(hierarchy):
    child = hierarchy.child()

    def consume(event, ctx):
        if event == EventType.TASK_RESULT_RECEIVED:
            ctx.get_prop(FLContextKey.TASK_RESULT).clear()

    hierarchy.parent.engine.fire_event.side_effect = consume
    reply = hierarchy.parent._process_submit_result(new_cell_message({}, child.result)).payload
    assert child.result.get_cookie(ReservedHeaderKey.WORKFLOW) is None
    assert reply.get_task_result_receipt("task-1", "attempt-1", "workflow") == TaskResultReceipt.RECEIVED


def test_unfenced_parent_keeps_legacy_event_ack(hierarchy):
    child = hierarchy.child()
    hierarchy.runner.running_tasks["task-1"] = TaskAssignment("train", "task-1", Shareable())
    legacy = Shareable({"model": 2})
    legacy.set_header(ReservedHeaderKey.TASK_ID, "task-1")
    legacy.set_header(ReservedHeaderKey.TASK_NAME, "train")
    legacy.set_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA, True)
    legacy.set_peer_context(child.ctx)
    reply = hierarchy.parent._process_submit_result(new_cell_message({}, legacy))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload.get_header(ReservedHeaderKey.TASK_RESULT_RECEIPT) is None
    hierarchy.gatherer._updater.process_child_update.assert_called_once()
