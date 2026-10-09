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

from pathlib import Path
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.dxo import DXO, DataKind, from_file
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode, TaskResultReceipt
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.impl.task_manager import TaskCheckStatus
from nvflare.apis.impl.wf_comm_server import WFCommServer, _DeadClientStatus
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather


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
def test_receipt_survives_processing_failure_for_active_and_retired_retries(failure):
    callback = Mock(side_effect=RuntimeError("callback failed") if failure == "callback" else None)
    client = Client("site-1", "token")
    task = Task("train", Shareable(), result_received_cb=callback)
    task.props["___mgr"] = Mock()
    client_task = ClientTask(client, task)
    wf = WFCommServer()
    wf._client_task_map[client_task.id] = client_task
    client_task.props["___job_id"] = "job-1"
    fl_ctx = FLContextManager(identity_name="server", job_id="job-1").new_context()
    fl_ctx.set_peer_context(FLContextManager(identity_name="site-1", job_id="job-1").new_context())
    result = make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR) if failure == "filter" else Shareable()
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf.process_submission(client, "train", client_task.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    # Preserve callback failure's completion status: an active-map retry must
    # keep that failed outcome even before the completed task is swept.
    wf.process_submission(client, "train", client_task.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    wf._remember_completed_client_task(client_task)
    wf._client_task_map.pop(client_task.id)
    wf.process_submission(client, "train", client_task.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    callback.assert_called_once()


def _assignment():
    client = Client("site-1", "token")
    task = Task("train", Shareable(), result_received_cb=Mock())
    task.props["___mgr"] = Mock()
    client_task = ClientTask(client, task)
    wf = WFCommServer()
    wf.controller = Mock()
    wf._client_task_map[client_task.id] = client_task
    client_task.props["___job_id"] = "job-1"
    ctx = FLContextManager(identity_name="server", job_id="job-1").new_context()
    ctx.set_peer_context(FLContextManager(identity_name="site-1", job_id="job-1").new_context())
    ctx.set_prop(FLContextKey.TASK_NAME, "train", private=True, sticky=False)
    return wf, client_task, ctx


@pytest.mark.parametrize("attempt_id", [None, "old-attempt", ""])
def test_missing_or_stale_attempt_never_invokes_result_callback(attempt_id):
    wf, client_task, fl_ctx = _assignment()
    result = Shareable()
    if attempt_id is not None:
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, attempt_id)
    wf.process_submission(client_task.client, "train", client_task.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
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
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    wf.process_submission(client_task.client, "train", client_task.id, second, fl_ctx)
    client_task.task.result_received_cb.assert_called_once()
    assert client_task.result is first


@pytest.mark.parametrize("return_code", [ReturnCode.EXECUTION_EXCEPTION, ReturnCode.TASK_ABORTED, "custom.failure"])
@pytest.mark.parametrize("completed", [False, True])
def test_failed_result_still_receives_receipt_on_every_retry(return_code, completed):
    wf, client_task, fl_ctx = _assignment()
    first = make_reply(return_code)
    first.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf.process_submission(client_task.client, "train", client_task.id, first, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    if completed:
        wf._remember_completed_client_task(client_task)
        wf._client_task_map.pop(client_task.id)
    duplicate = make_reply(ReturnCode.OK)
    duplicate.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf.process_submission(client_task.client, "train", client_task.id, duplicate, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    client_task.task.result_received_cb.assert_called_once()
    assert client_task.result is first


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize(
    "hook",
    [
        "manager_rc",
        "callback_rc",
        "callback_replacement_rc",
        "callback_task_error",
        "callback_clear",
        "callback_processed_value",
    ],
)
def test_application_outcome_does_not_change_receipt(hook, completed):
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
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    if completed:
        wf._remember_completed_client_task(client_task)
        wf._client_task_map.pop(client_task.id)
    duplicate = Shareable()
    duplicate.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, client_task.attempt_id)
    wf.process_submission(client_task.client, "train", client_task.id, duplicate, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    manager.check_task_result.assert_called_once()
    client_task.task.result_received_cb.assert_called_once()


@pytest.mark.parametrize("callback_result", [False, None, True])
@pytest.mark.parametrize("failure", [None, "return_code", "task_error"])
@pytest.mark.parametrize("completed", [False, True])
def test_callback_decision_does_not_change_consumed_result_receipt(callback_result, failure, completed):
    wf, assigned, fl_ctx = _assignment()
    first = Shareable()
    first.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)

    def callback(**kwargs):
        current = kwargs["client_task"]
        if failure == "return_code":
            current.result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
        elif failure == "task_error":
            current.task.completion_status = TaskCompletionStatus.ERROR
        current.result = None
        return callback_result

    assigned.task.result_received_cb.side_effect = callback
    wf.process_submission(assigned.client, "train", assigned.id, first, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    if completed:
        wf._remember_completed_client_task(assigned)
        wf._client_task_map.pop(assigned.id)
    duplicate = Shareable()
    duplicate.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    wf.process_submission(assigned.client, "train", assigned.id, duplicate, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assigned.task.result_received_cb.assert_called_once()


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
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
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


def _retired_assignment(task_name="train", status=TaskCompletionStatus.TIMEOUT, swept=True):
    wf, original, _ = _assignment()
    fl_ctx = FLContextManager(identity_name="server", job_id="job-1").new_context()
    fl_ctx.set_peer_context(FLContextManager(identity_name=original.client.name, job_id="job-1").new_context())
    wf._engine = Mock()
    wf._engine.new_context.return_value = fl_ctx
    wf._client_task_map.clear()
    original.task.name = task_name
    original.task.props["___mgr"].check_task_send.return_value = TaskCheckStatus.SEND
    wf._tasks = [original.task]
    _, task_id, data = wf.process_task_request(original.client, fl_ctx)
    assigned = wf._client_task_map[task_id]
    assigned.task.completion_status = status
    if swept:
        wf.check_tasks()
        assert wf.get_num_standing_tasks() == 0
    result = Shareable({"late": True})
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, data.get_task_attempt_id())
    return wf, assigned, fl_ctx, result


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("swept", [False, True])
def test_fatal_retry_cannot_replace_a_received_publication(accepted, swept):
    wf, assigned, fl_ctx = _assignment()
    assigned.task.result_received_cb.return_value = accepted
    result = Shareable({"first": True})
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    if swept:
        wf._remember_completed_client_task(assigned)
        wf._client_task_map.pop(assigned.id)
    fatal = make_reply(ReturnCode.UNSAFE_JOB)
    fatal.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    assert not wf.claim_submission(assigned.client, "train", assigned.id, fatal, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assert assigned.result is result
    assigned.task.result_received_cb.assert_called_once()


@pytest.mark.parametrize("stage", ["active", "retired", "swept"])
@pytest.mark.parametrize("mismatch", ["client", "task", "attempt", "missing_attempt"])
def test_invalid_submission_cannot_claim_an_assignment(stage, mismatch):
    wf, assigned, fl_ctx, result = _retired_assignment(swept=stage == "swept")
    if stage == "active":
        assigned.task.completion_status = None
    client = Client("other-site", "token") if mismatch == "client" else assigned.client
    task_name = "other-task" if mismatch == "task" else "train"
    if mismatch in ("attempt", "missing_attempt"):
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "other-attempt" if mismatch == "attempt" else None)
    assert not wf.claim_submission(client, task_name, assigned.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
    assert assigned.result_received_time is None
    completed = wf._completed_client_task_map.get(assigned.id)
    assert completed is None or not completed.received
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    assert wf.claim_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    wf.finish_submission(assigned.id, fl_ctx)
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assert assigned.result is None
    assigned.task.result_received_cb.assert_not_called()
    assigned.task.props["___mgr"].check_task_result.assert_not_called()
    wf.controller.process_result_of_unknown_task.assert_not_called()


def test_unfenced_legacy_claim_does_not_fabricate_an_assignment():
    wf, assigned, fl_ctx = _assignment()
    result = Shareable()
    assert wf.claim_submission(assigned.client, "train", "unknown", result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    assert not wf._completed_client_task_map
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    assert not wf.claim_submission(assigned.client, "train", "unknown", result, fl_ctx)
    wf.controller.process_result_of_unknown_task.assert_not_called()


@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("status", [TaskCompletionStatus.TIMEOUT, TaskCompletionStatus.CANCELLED])
def test_exact_first_late_assigned_result_reaches_unknown_hook_once(status, swept):
    wf, assigned, fl_ctx, result = _retired_assignment(status=status, swept=swept)
    assert wf.check_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    wf.controller.process_result_of_unknown_task.assert_called_once_with(
        assigned.client, "train", assigned.id, result, fl_ctx
    )
    assigned.task.result_received_cb.assert_not_called()
    # A void legacy hook does not invent successful result admission.
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    duplicate = Shareable({"replacement": True})
    duplicate.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    wf.process_submission(assigned.client, "train", assigned.id, duplicate, fl_ctx)
    if not swept:
        wf.check_tasks()
        wf.process_submission(assigned.client, "train", assigned.id, duplicate, fl_ctx)
    wf.controller.process_result_of_unknown_task.assert_called_once()
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED


@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("cookie_only", [False, True])
def test_scatter_and_gather_aggregates_first_authenticated_late_result_once(swept, cookie_only):
    wf, assigned, fl_ctx, result = _retired_assignment(swept=swept)
    if cookie_only:
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, None)
        result.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    controller = ScatterAndGather()
    controller._phase = AppConstants.PHASE_TRAIN
    controller._current_round = 0
    controller.fire_event = Mock()
    controller.aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)
    controller.aggregator.handle_event(EventType.START_RUN, fl_ctx)
    wf.controller = controller
    DXO(DataKind.WEIGHTS, {"weight": 3.0}).update_shareable(result)
    result.set_peer_props({ReservedKey.IDENTITY_NAME: assigned.client.name})
    result.add_cookie(AppConstants.CONTRIBUTION_ROUND, 0)
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    assert fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED) is True
    # The void late hook preserves its established behavior; it does not opt in
    # to a new admission ACK merely because aggregation accepted a contribution.
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    helper = controller.aggregator.dxo_aggregators[""].aggregation_helper
    assert helper.get_len() == 1
    DXO(DataKind.WEIGHTS, {"weight": 100.0}).update_shareable(result)
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    assert helper.get_len() == 1
    assert helper.get_result() == {"weight": 3.0}
    assert controller.fire_event.call_count == 2


@pytest.mark.parametrize("swept", [False, True])
def test_fedavg_first_late_result_keeps_existing_unknown_hook_behavior(swept):
    wf, assigned, fl_ctx, result = _retired_assignment(swept=swept)
    controller = FedAvg(num_clients=1)
    wf.controller = controller
    # Current FedAvg's late hook publishes TRAINING_RESULT for consumers; its
    # ordinary in-time callback is reserved for a standing assignment.
    hook = Mock(wraps=controller.process_result_of_unknown_task)
    controller.process_result_of_unknown_task = hook
    DXO(DataKind.WEIGHTS, {"weight": 3.0}).update_shareable(result)
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    assert fl_ctx.get_prop(AppConstants.TRAINING_RESULT) is result
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    retry = make_reply(ReturnCode.OK)
    retry.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    wf.process_submission(assigned.client, "train", assigned.id, retry, fl_ctx)
    assert fl_ctx.get_prop(AppConstants.TRAINING_RESULT) is result
    hook.assert_called_once()


@pytest.mark.parametrize("task_name", [AppConstants.TASK_SUBMIT_MODEL, AppConstants.TASK_VALIDATION])
@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("cookie_only", [False, True])
def test_cross_site_model_eval_stores_exact_late_assigned_contribution_once(tmp_path, task_name, swept, cookie_only):
    wf, assigned, fl_ctx, result = _retired_assignment(task_name=task_name, swept=swept)
    if cookie_only:
        # Existing job clients echo cookies without adding an attempt header.
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, None)
        result.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
        result.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED, True)
    controller = CrossSiteModelEval()
    controller.fire_event = Mock()
    controller._send_validation_task = Mock()
    controller._cross_val_models_dir = str(tmp_path)
    controller._cross_val_results_dir = str(tmp_path)
    wf.controller = controller
    data = {"weight": 1.0} if task_name == AppConstants.TASK_SUBMIT_MODEL else {"accuracy": 0.8}
    kind = DataKind.WEIGHTS if task_name == AppConstants.TASK_SUBMIT_MODEL else DataKind.METRICS
    DXO(kind, data).update_shareable(result)
    result.add_cookie(AppConstants.MODEL_OWNER, "model-1")
    wf.process_submission(assigned.client, task_name, assigned.id, result, fl_ctx)
    if task_name == AppConstants.TASK_SUBMIT_MODEL:
        path = controller._client_models[assigned.client.name]
        controller._send_validation_task.assert_called_once_with(assigned.client.name, fl_ctx)
    else:
        path = controller._val_results[assigned.client.name]["model-1"]
    assert from_file(path).data == data
    saved = Path(path).read_bytes()
    DXO(kind, {"replacement": 2.0}).update_shareable(result)
    wf.process_submission(assigned.client, task_name, assigned.id, result, fl_ctx)
    assert Path(path).read_bytes() == saved
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED


@pytest.mark.parametrize(
    "mismatch", ["peer_missing", "peer_name", "peer_job", "context_job", "client", "task_name", "task_id", "attempt"]
)
def test_late_assigned_results_require_exact_authority_and_authenticated_job(mismatch):
    wf, assigned, fl_ctx, result = _retired_assignment()
    client, task_name, task_id = assigned.client, "train", assigned.id
    if mismatch == "peer_missing":
        fl_ctx.set_peer_context(None)
    elif mismatch == "peer_name":
        fl_ctx.set_peer_context(FLContextManager(identity_name="other-site", job_id="job-1").new_context())
    elif mismatch == "peer_job":
        fl_ctx.set_peer_context(FLContextManager(identity_name=client.name, job_id="other-job").new_context())
    elif mismatch == "context_job":
        fl_ctx = FLContextManager(identity_name="server", job_id="other-job").new_context()
        fl_ctx.set_peer_context(FLContextManager(identity_name=client.name, job_id="job-1").new_context())
    elif mismatch == "client":
        client = Client("other-site", "token")
    elif mismatch == "task_name":
        task_name = "other-task"
    elif mismatch == "task_id":
        task_id = "unrecognized-assignment"
    else:
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "stale-attempt")
    wf.process_submission(client, task_name, task_id, result, fl_ctx)
    wf.controller.process_result_of_unknown_task.assert_not_called()
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED


@pytest.mark.parametrize("admitted", [False, True])
@pytest.mark.parametrize("failure", [None, "return_code", "exception"])
def test_late_hook_outcome_does_not_change_receipt_or_repeat_effects(admitted, failure):
    wf, assigned, fl_ctx, result = _retired_assignment()

    def hook(*_args):
        if failure == "return_code":
            result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
        elif failure == "exception":
            raise RuntimeError("late hook failed after a side effect")

    wf.controller.process_result_of_unknown_task.side_effect = hook
    if failure == "exception":
        with pytest.raises(RuntimeError, match="late hook failed"):
            wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    else:
        wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    duplicate = Shareable()
    duplicate.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    wf.process_submission(assigned.client, "train", assigned.id, duplicate, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    wf.controller.process_result_of_unknown_task.assert_called_once()


@pytest.mark.parametrize("peer_job, context_job", [("job-1", "job-1"), ("other", "job-1"), ("job-1", "other")])
@pytest.mark.parametrize("swept", [False, True])
def test_retired_pending_assignment_task_check_requires_issued_job_and_peer(peer_job, context_job, swept):
    wf, assigned, _, result = _retired_assignment(swept=swept)
    fl_ctx = FLContextManager(identity_name="server", job_id=context_job).new_context()
    fl_ctx.set_peer_context(FLContextManager(identity_name=assigned.client.name, job_id=peer_job).new_context())
    fl_ctx.set_prop(FLContextKey.TASK_ATTEMPT_ID, result.get_task_attempt_id(), private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_NAME, "train", private=True, sticky=False)
    retired = wf.process_task_check(assigned.id, fl_ctx)
    if peer_job == context_job == "job-1":
        assert retired is not None
        assert not retired.received
    else:
        assert retired is None


def test_late_result_without_captured_issuing_job_cannot_use_unknown_hook():
    wf, assigned, fl_ctx, result = _retired_assignment()
    wf._completed_client_task_map[assigned.id].job_id = None
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    wf.controller.process_result_of_unknown_task.assert_not_called()
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED


@pytest.mark.parametrize("attempt", [None, "", "conflicting"])
def test_missing_or_conflicting_late_attempt_cannot_downgrade_issued_assignment(attempt):
    wf, assigned, fl_ctx, result = _retired_assignment()
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED, True)
    if attempt == "conflicting":
        result.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, "other-attempt")
    else:
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, attempt)
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    wf.controller.process_result_of_unknown_task.assert_not_called()
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED


@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize(
    "event", [EventType.BEFORE_PROCESS_RESULT_OF_UNKNOWN_TASK, EventType.AFTER_PROCESS_RESULT_OF_UNKNOWN_TASK]
)
def test_late_event_failure_keeps_receipt_before_retries(event, swept):
    wf, assigned, fl_ctx, result = _retired_assignment(swept=swept)

    def fire(event_type, _fl_ctx):
        if event_type == event:
            raise RuntimeError("late event failed")

    wf.fire_event = Mock(side_effect=fire)
    with pytest.raises(RuntimeError, match="late event failed"):
        wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    assert wf.controller.process_result_of_unknown_task.call_count == (
        0 if event == EventType.BEFORE_PROCESS_RESULT_OF_UNKNOWN_TASK else 1
    )
    assert wf.fire_event.call_count == (1 if event == EventType.BEFORE_PROCESS_RESULT_OF_UNKNOWN_TASK else 2)


def test_retired_assignment_cache_is_bounded_and_evicted_fenced_results_stay_unrecognized(monkeypatch):
    monkeypatch.setattr("nvflare.apis.impl.wf_comm_server._COMPLETED_CLIENT_TASK_CACHE_SIZE", 2)
    wf, assigned, fl_ctx, result = _retired_assignment()
    for _ in range(2):
        other = ClientTask(assigned.client, Task("train", Shareable()))
        other.props["___job_id"] = "job-1"
        wf._remember_completed_client_task(other)
    assert len(wf._completed_client_task_map) == 2
    assert assigned.id not in wf._completed_client_task_map
    wf.process_submission(assigned.client, "train", assigned.id, result, fl_ctx)
    wf.controller.process_result_of_unknown_task.assert_not_called()


@pytest.mark.parametrize("capacity", [1, 2, 10000, 0, -1])
def test_retired_assignment_history_capacity_comes_from_application_config(monkeypatch, capacity):
    wf, _, ctx = _assignment()
    ctx.set_prop(ReservedKey.ENGINE, Mock(spec=ServerEngineSpec), private=True, sticky=False)

    def configured(name, conf, default):
        assert name == "task_result_history_size"
        assert default == 10000
        return capacity

    monkeypatch.setattr("nvflare.apis.impl.wf_comm_server.ConfigService.get_int_var", configured)
    wf._task_monitor = Mock()
    if capacity <= 0:
        with pytest.raises(ValueError, match="must be positive"):
            wf.initialize_run(ctx)
        wf._task_monitor.start.assert_not_called()
    else:
        wf.initialize_run(ctx)
        assert wf._completed_client_task_cache_size == capacity
        wf._task_monitor.start.assert_called_once()


def test_receipt_claim_does_not_finish_workflow_until_processing_finishes():
    wf, assigned, ctx = _assignment()
    result = Shareable()
    result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
    assert wf.claim_submission(assigned.client, "train", assigned.id, result, ctx)
    assert assigned.result_received_time is None
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED
    wf.process_submission(assigned.client, "train", assigned.id, result, ctx)
    assert assigned.result_received_time is not None
    assigned.task.result_received_cb.assert_called_once()


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


@pytest.mark.parametrize("forwarded_header", [False, True])
def test_forwarded_result_data_rebinds_assignment_cookies_without_mutating_source(forwarded_header):
    wf, client_task, fl_ctx = _assignment()
    forwarded = Shareable({"model": "forwarded"})
    forwarded.add_cookie(ReservedHeaderKey.TASK_ID, "prior-task")
    forwarded.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, "prior-attempt")
    forwarded.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED, True)
    forwarded.add_cookie("application-cookie", "preserved")
    if forwarded_header:
        forwarded.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "prior-attempt")
    client_task.task.data = forwarded
    client_task.task.props["___mgr"].check_task_send.return_value = TaskCheckStatus.SEND
    wf._client_task_map.clear()
    wf._tasks = [client_task.task]

    _, first_id, first_data = wf.process_task_request(client_task.client, fl_ctx)
    _, second_id, second_data = wf.process_task_request(Client("site-2", "token-2"), fl_ctx)
    _, resend_id, resend_data = wf.process_task_request(client_task.client, fl_ctx)

    for task_id, data in ((first_id, first_data), (second_id, second_data), (resend_id, resend_data)):
        assigned = wf._client_task_map[task_id]
        # ServerRunner reads this before GetTaskCommand can stamp wire cookies.
        assert data.get_task_attempt_id() == assigned.attempt_id
        assert data.get_cookie(ReservedHeaderKey.TASK_ID) == task_id
        assert data.get_cookie(ReservedHeaderKey.TASK_ATTEMPT_REQUIRED) is True
        assert data.get_cookie("application-cookie") == "preserved"
        assert data["model"] == "forwarded"
    assert first_id != second_id
    assert first_data.get_task_attempt_id() != second_data.get_task_attempt_id()
    assert resend_id == first_id
    assert resend_data.get_task_attempt_id() == first_data.get_task_attempt_id()
    assert forwarded.get_task_attempt_id() == "prior-attempt"
    assert forwarded.get_cookie(ReservedHeaderKey.TASK_ID) == "prior-task"
    assert forwarded.get_cookie("application-cookie") == "preserved"

    # Rebinding outbound task data must not normalize a forged incoming result.
    forged = Shareable()
    forged.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, first_data.get_task_attempt_id())
    forged.add_cookie(ReservedHeaderKey.TASK_ATTEMPT_ID, "prior-attempt")
    first_assignment = wf._client_task_map[first_id]
    wf.process_submission(first_assignment.client, "train", first_id, forged, fl_ctx)
    assert fl_ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.TASK_CLOSED
    first_assignment.task.result_received_cb.assert_not_called()


@pytest.mark.parametrize("stage", ["active", "retired", "swept"])
@pytest.mark.parametrize("received", [False, True])
@pytest.mark.parametrize("task_name", ["train", "other"])
def test_task_check_exposes_receipt_without_leaking_private_marker(stage, received, task_name, monkeypatch):
    # A private marker rename must not change the public readiness contract.
    monkeypatch.setattr("nvflare.apis.impl.wf_comm_server._CLIENT_TASK_RESULT_RECEIVED", "renamed-private-receipt")
    wf, assigned, ctx = _assignment()
    if received:
        result = Shareable()
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, assigned.attempt_id)
        assert wf.claim_submission(assigned.client, "train", assigned.id, result, ctx)
        wf.finish_submission(assigned.id, ctx)
    if stage != "active":
        assigned.task.completion_status = TaskCompletionStatus.TIMEOUT
    if stage == "swept":
        wf._remember_completed_client_task(assigned)
        wf._client_task_map.pop(assigned.id)
    ctx.set_prop(FLContextKey.TASK_ATTEMPT_ID, assigned.attempt_id, private=True, sticky=False)
    ctx.set_prop(FLContextKey.TASK_NAME, task_name, private=True, sticky=False)
    record = wf.process_task_check(assigned.id, ctx)
    if task_name == "train":
        assert record is not None
        expected = TaskResultReceipt.RECEIVED if received else TaskResultReceipt.RETRY
    else:
        assert record is None
        expected = TaskResultReceipt.TASK_CLOSED
    assert ctx.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == expected


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("peer_name, correct_attempt", [("site-1", True), ("site-2", True), ("site-1", False)])
def test_attempt_task_check_requires_matching_peer_and_supports_lost_ack(completed, peer_name, correct_attempt):
    wf, client_task, fl_ctx = _assignment()
    if completed:
        client_task.result_received_time = 1.0
        client_task.props["___result_received"] = True
        wf._remember_completed_client_task(client_task)
        wf._client_task_map.pop(client_task.id)
    peer = FLContextManager(identity_name=peer_name, job_id="job-1").new_context()
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
