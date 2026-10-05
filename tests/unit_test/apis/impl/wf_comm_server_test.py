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

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.impl.task_manager import TaskCheckStatus
from nvflare.apis.impl.wf_comm_server import WFCommServer, _DeadClientStatus
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply


def _make_wf_comm(clients, dead_names, min_sites=1, required_sites=None):
    """Build a WFCommServer with mocked engine, enrolled clients, and pre-declared dead clients."""
    mock_engine = Mock(spec=ServerEngineSpec)
    ctx_mgr = FLContextManager(
        engine=mock_engine,
        identity_name="__mock_server",
        job_id="job_1",
        public_stickers={},
        private_stickers={},
    )
    fl_ctx = ctx_mgr.new_context()
    fl_ctx.set_prop(
        FLContextKey.JOB_META,
        {
            JobMetaKey.MIN_CLIENTS: min_sites,
            JobMetaKey.MANDATORY_CLIENTS: required_sites or [],
        },
    )
    mock_engine.new_context.return_value = fl_ctx
    mock_engine.get_clients.return_value = clients

    wf = WFCommServer()
    wf._engine = mock_engine
    for name in dead_names:
        status = _DeadClientStatus()
        status.disconnect_time = 1.0  # non-None → deemed disconnected
        wf._dead_clients[name] = status
    return wf


@pytest.mark.parametrize("failure", [None, "callback", "filter"])
def test_acceptance_preserves_processing_outcome_for_active_and_completed_retries(failure):
    callback = Mock(side_effect=RuntimeError("callback failed") if failure == "callback" else None)
    client = Client("site-1", "token")
    task = Task("train", Shareable(), result_received_cb=callback)
    task.props["___mgr"] = Mock()
    client_task = ClientTask(client, task)
    wf = WFCommServer()
    wf._client_task_map[client_task.id] = client_task
    fl_ctx = FLContextManager().new_context()
    result = make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR) if failure == "filter" else Shareable()
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf._do_process_submission(client, "train", client_task.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is (failure is None)
    # Preserve callback failure's completion status: an active-map retry must
    # keep that failed outcome even before the completed task is swept.
    wf._do_process_submission(client, "train", client_task.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is (failure is None)
    wf._remember_completed_client_task(client_task)
    wf._client_task_map.pop(client_task.id)
    wf._do_process_submission(client, "train", client_task.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is (failure is None)
    callback.assert_called_once()


def _assignment():
    client = Client("site-1", "token")
    task = Task("train", Shareable(), result_received_cb=Mock())
    task.props["___mgr"] = Mock()
    client_task = ClientTask(client, task)
    wf = WFCommServer()
    wf.controller = Mock()
    wf._client_task_map[client_task.id] = client_task
    return wf, client_task, FLContextManager().new_context()


@pytest.mark.parametrize("attempt_id", [None, "old-attempt", ""])
def test_missing_or_stale_attempt_never_invokes_result_callback(attempt_id):
    wf, client_task, fl_ctx = _assignment()
    result = Shareable()
    if attempt_id is not None:
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, attempt_id)
    wf.process_submission(client_task.client, "train", client_task.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is False
    client_task.task.result_received_cb.assert_not_called()
    assert client_task.result is None


@pytest.mark.parametrize("completed", [False, True])
def test_conflicting_duplicate_payload_does_not_replace_first_result(completed):
    wf, client_task, fl_ctx = _assignment()
    first = Shareable({"value": "first"})
    first.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf.process_submission(client_task.client, "train", client_task.id, first, fl_ctx)
    if completed:
        wf._remember_completed_client_task(client_task)
        wf._client_task_map.pop(client_task.id)
    second = Shareable({"value": "different"})
    second.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    assert not wf.check_submission(client_task.client, "train", client_task.id, second, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is True
    wf.process_submission(client_task.client, "train", client_task.id, second, fl_ctx)
    client_task.task.result_received_cb.assert_called_once()
    assert client_task.result is first


@pytest.mark.parametrize("return_code", [ReturnCode.EXECUTION_EXCEPTION, ReturnCode.TASK_ABORTED, "custom.failure"])
@pytest.mark.parametrize("completed", [False, True])
def test_handled_non_ok_result_receives_false_admission_ack_on_every_retry(return_code, completed):
    wf, client_task, fl_ctx = _assignment()
    first = make_reply(return_code)
    first.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf.process_submission(client_task.client, "train", client_task.id, first, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is False
    if completed:
        wf._remember_completed_client_task(client_task)
        wf._client_task_map.pop(client_task.id)
    duplicate = make_reply(ReturnCode.OK)
    duplicate.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf.process_submission(client_task.client, "train", client_task.id, duplicate, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is False
    client_task.task.result_received_cb.assert_called_once()
    assert client_task.result is first


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize(
    "hook, expected_accepted",
    [
        ("manager_rc", False),
        ("callback_rc", False),
        ("callback_replacement_rc", False),
        ("callback_task_error", False),
        ("callback_clear", True),
        ("callback_processed_value", True),
    ],
)
def test_final_hook_outcome_gates_state_admission_without_restricting_consumed_results(
    hook, expected_accepted, completed
):
    wf, client_task, fl_ctx = _assignment()
    first = Shareable()
    first.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    manager = client_task.task.props["___mgr"]
    if hook == "manager_rc":
        manager.check_task_result.side_effect = lambda result, *_args: result.set_return_code(ReturnCode.TASK_ABORTED)

    def callback(**kwargs):
        current = kwargs["client_task"]
        if hook == "callback_rc":
            current.result.set_return_code(ReturnCode.TASK_ABORTED)
        elif hook == "callback_replacement_rc":
            current.result = make_reply(ReturnCode.EXECUTION_EXCEPTION)
        elif hook == "callback_task_error":
            current.task.completion_status = TaskCompletionStatus.ERROR
        elif hook == "callback_clear":
            current.result = None
        elif hook == "callback_processed_value":
            current.result = {"consumed": True}

    client_task.task.result_received_cb.side_effect = callback
    wf.process_submission(client_task.client, "train", client_task.id, first, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is expected_accepted
    if completed:
        wf._remember_completed_client_task(client_task)
        wf._client_task_map.pop(client_task.id)
    duplicate = Shareable()
    duplicate.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf.process_submission(client_task.client, "train", client_task.id, duplicate, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is expected_accepted
    manager.check_task_result.assert_called_once()
    client_task.task.result_received_cb.assert_called_once()


def test_manager_exception_is_rethrown_but_attempt_retries_do_not_repeat_side_effects():
    wf, client_task, fl_ctx = _assignment()
    error = RuntimeError("manager failed after a side effect")
    manager = client_task.task.props["___mgr"]
    manager.check_task_result.side_effect = error
    result = Shareable()
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    with pytest.raises(RuntimeError, match="manager failed") as raised:
        wf.process_submission(client_task.client, "train", client_task.id, result, fl_ctx)
    assert raised.value is error
    assert client_task.result_received_time is not None
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is False
    wf.process_submission(client_task.client, "train", client_task.id, result, fl_ctx)
    manager.check_task_result.assert_called_once()
    client_task.task.result_received_cb.assert_not_called()
    assert client_task.task.completion_status is None


def test_unknown_attempt_fenced_result_cannot_use_legacy_unknown_handler():
    wf, client_task, fl_ctx = _assignment()
    result = Shareable()
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf.process_submission(client_task.client, "train", "unknown", result, fl_ctx)
    wf.controller.process_result_of_unknown_task.assert_not_called()
    wf.process_submission(client_task.client, "train", "unknown", Shareable(), fl_ctx)
    wf.controller.process_result_of_unknown_task.assert_called_once()


def test_server_resend_preserves_authority_issued_attempt_identity():
    wf, client_task, fl_ctx = _assignment()
    wf._client_task_map.clear()
    wf._tasks = [client_task.task]
    client_task.task.props["___mgr"].check_task_send.return_value = TaskCheckStatus.SEND
    _, first_id, first_data = wf.process_task_request(client_task.client, fl_ctx)
    _, resend_id, resend_data = wf.process_task_request(client_task.client, fl_ctx)
    assert first_id == resend_id
    assert first_data.get_task_attempt_id() == resend_data.get_task_attempt_id()
    assert first_data.get_header(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED) is True
    assert first_id != first_data.get_task_attempt_id()
    assert len(client_task.task.client_tasks) == 1


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("peer_name, correct_attempt", [("site-1", True), ("site-2", True), ("site-1", False)])
def test_attempt_task_check_requires_matching_peer_and_supports_lost_ack(completed, peer_name, correct_attempt):
    wf, client_task, fl_ctx = _assignment()
    if completed:
        client_task.result_received_time = 1.0
        client_task.props["___result_accepted"] = True
        wf._remember_completed_client_task(client_task)
        wf._client_task_map.pop(client_task.id)
    peer = FLContextManager(identity_name=peer_name).new_context()
    fl_ctx.set_peer_context(peer)
    fl_ctx.set_prop(
        FLContextKey.TASK_ATTEMPT_ID, client_task.attempt_id if correct_attempt else "stale", private=True, sticky=False
    )
    assert bool(wf.process_task_check(client_task.id, fl_ctx)) is (peer_name == "site-1" and correct_attempt)
    fl_ctx.set_prop(FLContextKey.TASK_ATTEMPT_ID, None, private=True, sticky=False)
    assert bool(wf.process_task_check(client_task.id, fl_ctx)) is (not completed)


class TestJobPolicyViolated:
    def test_alive_below_min_sites_aborts(self):
        """min_sites=2, 2 enrolled, 1 dead → alive=1 < min_sites=2 → abort."""
        clients = [Client("site-1", "tok-1"), Client("site-2", "tok-2")]
        wf = _make_wf_comm(clients, dead_names=["site-1"], min_sites=2)
        assert wf._job_policy_violated() is True

    def test_non_required_dead_client_no_abort(self):
        """min_sites=1, 2 enrolled, 1 non-required dead → alive=1 >= min_sites=1, not required → no abort."""
        clients = [Client("site-1", "tok-1"), Client("site-2", "tok-2")]
        wf = _make_wf_comm(clients, dead_names=["site-1"], min_sites=1, required_sites=[])
        assert wf._job_policy_violated() is False

    def test_required_dead_client_aborts(self):
        """min_sites=1, 2 enrolled, 1 required dead → required client is dead → abort."""
        clients = [Client("site-1", "tok-1"), Client("site-2", "tok-2")]
        wf = _make_wf_comm(clients, dead_names=["site-1"], min_sites=1, required_sites=["site-1"])
        assert wf._job_policy_violated() is True
