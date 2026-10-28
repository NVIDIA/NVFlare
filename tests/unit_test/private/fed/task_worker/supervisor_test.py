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


import json
import os
import signal
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReturnCode, TaskResultReceipt
from nvflare.apis.shareable import Shareable
from nvflare.apis.task_launcher_spec import TaskLaunchError, TaskSettlementError
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions
from tests.task_supervisor_test_utils import RUNNING, SETTLED, UNSETTLED, _attempt, _payload, _records


@pytest.mark.parametrize("fault", ["none", "payload", "completion", "released", "unknown"])
def test_saved_result_requires_the_original_retained_validated_completion(tmp_path, fault):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    identity = outcome.identity
    if fault == "payload":
        _payload(attempt).write_bytes(b"corrupt")
    elif fault == "completion":
        path = _payload(attempt).parent / "completion.json"
        record = json.loads(path.read_text())
        record["worker_pid"] += 1
        path.write_text(json.dumps(record))
    elif fault == "released":
        supervisor.record_receipt(identity, TaskResultReceipt.RECEIVED)
    elif fault == "unknown":
        identity = replace(identity, task_name="other")
    if fault == "none":
        assert supervisor.read_saved_result(identity)["value"] == 2
    else:
        with pytest.raises((RuntimeError, ValueError)):
            supervisor.read_saved_result(identity)
    assert attempt["launcher"].launch_task.call_count == 1


def test_worker_failure_reports_original_type_message_and_diagnostics(tmp_path):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)

    def fail(request):
        handle.request = request
        attempt["store"].record_failure(
            attempt["bootstrap"].identity,
            worker_pid=os.getpid(),
            worker_ppid=os.getppid(),
            started_at=1,
            failed_at=2,
            error_type="ValueError",
            message="trainer rejected its input",
        )
        return handle

    attempt["launcher"].launch_task.side_effect = fail
    with pytest.raises(RuntimeError, match="ValueError: trainer rejected its input.*diagnostics:"):
        supervisor.run(**attempt)
    assert (_payload(attempt).parent / "failure.json").exists()


def test_launch_error_without_handle_preserves_original_exception(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    attempt["launcher"].launch_task.side_effect = TaskLaunchError("spawn denied", None)
    with pytest.raises(TaskLaunchError, match="spawn denied"):
        supervisor.run(**attempt)
    assert supervisor._active is None
    assert not supervisor.settlement_unconfirmed


def test_group_teardown_does_not_discard_a_committed_result(tmp_path):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    handle.wait_for_settlement.return_value = replace(
        SETTLED, cancel_requested=True, failure_reason="leader exited while descendants remained"
    )
    outcome = supervisor.run(**attempt)
    assert not outcome.aborted
    assert outcome.result["value"] == 2
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
    assert _payload(attempt).exists()


@pytest.mark.parametrize("receipt", [TaskResultReceipt.RECEIVED, TaskResultReceipt.TASK_CLOSED])
def test_terminal_receipt_releases_immediately_and_preserves_diagnostics(tmp_path, receipt):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    assert supervisor.record_receipt(outcome.identity, receipt)
    assert not _payload(attempt).exists()

    supervisor.end_run()
    supervisor.end_run()

    assert not _payload(attempt).exists()
    assert (_payload(attempt).parent / "completion.json").exists()
    assert _records(attempt)[-1]["event"] == "released"
    with pytest.raises(RuntimeError, match="stopping"):
        supervisor.run(**_attempt(tmp_path, attempt_id="attempt-2")[0])


@pytest.mark.parametrize("receipt", [None, TaskResultReceipt.RETRY])
def test_unknown_receipt_and_retry_preserve_saved_payload_after_end_run(tmp_path, receipt):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    supervisor.record_receipt(outcome.identity, receipt)
    supervisor.end_run()
    assert _payload(attempt).exists()
    assert supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
    assert not _payload(attempt).exists()
    attempt["launcher"].launch_task.assert_called_once()


def test_receipt_uses_full_identity_and_terminal_observation_is_monotonic(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    for name in ("job_id", "site_name", "task_id", "task_name", "attempt_id"):
        assert not supervisor.record_receipt(replace(outcome.identity, **{name: "other"}), TaskResultReceipt.RECEIVED)
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.TASK_CLOSED)
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.RETRY)
    supervisor.end_run()
    assert not _payload(attempt).exists()


def test_cleanup_does_not_block_next_assignment_or_another_receipt(tmp_path):
    supervisor = TaskSupervisor()
    first, _ = _attempt(tmp_path)
    outcome = supervisor.run(**first)
    releasing, proceed = threading.Event(), threading.Event()
    release = first["store"].release_payloads

    def blocked_release(identity):
        releasing.set()
        assert proceed.wait(5)
        release(identity)

    first["store"].release_payloads = Mock(side_effect=blocked_release)
    with ThreadPoolExecutor() as pool:
        receipting = pool.submit(supervisor.record_receipt, outcome.identity, TaskResultReceipt.RECEIVED)
        try:
            assert releasing.wait(5)
            second, _ = _attempt(tmp_path, attempt_id="attempt-2")
            next_outcome = supervisor.run(**second)
            supervisor.record_receipt(next_outcome.identity, TaskResultReceipt.RECEIVED)
            supervisor.cleanup_job_payloads()
            assert not _payload(second).exists()
            assert _payload(first).exists()
        finally:
            proceed.set()
        receipting.result(timeout=5)
    assert not _payload(first).exists()
    first["store"].release_payloads.assert_called_once()


def test_cleanup_failure_retains_diagnostics_and_does_not_block_new_work(tmp_path):
    supervisor = TaskSupervisor()
    first, _ = _attempt(tmp_path)
    outcome = supervisor.run(**first)
    first["store"].release_payloads = Mock(side_effect=OSError("release failed"))
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
    assert _payload(first).exists()
    assert supervisor.take_issues() == [("warning", "failed to release task payloads: release failed")]
    supervisor.cleanup_job_payloads()
    first["store"].release_payloads.assert_called_once()
    second, _ = _attempt(tmp_path, attempt_id="attempt-2")
    assert supervisor.run(**second).result["value"] == 2


def test_explicit_retention_preserves_all_payloads(tmp_path):
    supervisor = TaskSupervisor(retain_artifacts=True)
    attempt, _ = _attempt(tmp_path)
    outcome = supervisor.run(**attempt)
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
    supervisor.end_run()
    assert _payload(attempt).exists()


@pytest.mark.parametrize("failure", ["unsettled", "exception"])
def test_unconfirmed_settlement_retains_handle_and_forever_fences_new_work(tmp_path, failure):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    if failure == "unsettled":
        handle.wait_for_settlement.return_value = UNSETTLED
    else:
        handle.wait_for_settlement.side_effect = TaskSettlementError("lost owner", UNSETTLED)
    attempt["store"].read_result = Mock()

    with pytest.raises(TaskSettlementError):
        supervisor.run(**attempt)

    assert supervisor.settlement_unconfirmed
    assert supervisor._active.handle is handle
    attempt["store"].read_result.assert_not_called()
    supervisor.end_run()  # A later confirmed cancellation does not reset the fence.
    assert supervisor._active.handle is handle
    assert _payload(attempt).exists()
    with pytest.raises(RuntimeError, match="stopping"):
        supervisor.run(**_attempt(tmp_path, attempt_id="attempt-2")[0])


@pytest.mark.parametrize("settled", [True, False])
def test_post_spawn_launch_error_retains_and_cancels_its_owned_handle(tmp_path, settled):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    original = attempt["launcher"].launch_task.side_effect

    def failed_launch(request):
        original(request)
        raise TaskLaunchError("post-spawn failure", handle)

    attempt["launcher"].launch_task.side_effect = failed_launch
    handle.cancel.return_value = SETTLED if settled else UNSETTLED
    attempt["store"].read_result = Mock()
    with pytest.raises(TaskLaunchError if settled else TaskSettlementError):
        supervisor.run(**attempt)
    handle.cancel.assert_called_once()
    attempt["store"].read_result.assert_not_called()
    assert supervisor.settlement_unconfirmed is (not settled)
    with pytest.raises(RuntimeError, match="already owned|stopping"):
        supervisor.run(**attempt)
    assert _payload(attempt).exists()


@pytest.mark.parametrize("operation", ["cancel", "end_run"])
def test_cancellation_during_launch_takes_ownership_when_handle_arrives(tmp_path, operation):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    original = attempt["launcher"].launch_task.side_effect
    launching, proceed = threading.Event(), threading.Event()

    def blocked_launch(request):
        launching.set()
        assert proceed.wait(5)
        return original(request)

    attempt["launcher"].launch_task.side_effect = blocked_launch
    with ThreadPoolExecutor() as pool:
        future = pool.submit(supervisor.run, **attempt)
        try:
            assert launching.wait(5)
            with pytest.raises(RuntimeError, match="active"):
                supervisor.run(**_attempt(tmp_path, attempt_id="attempt-2")[0])
            (supervisor.end_run if operation == "end_run" else supervisor.cancel_active)()
        finally:
            proceed.set()
        assert future.result(timeout=5).aborted
    handle.cancel.assert_called_once()
    handle.wait_for_settlement.assert_not_called()
    assert _payload(attempt).exists()


def test_end_run_during_result_read_waits_and_does_not_publish_or_release(tmp_path):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    reading, proceed = threading.Event(), threading.Event()
    read = attempt["store"].read_result

    def blocked_read(identity):
        reading.set()
        assert proceed.wait(5)
        return read(identity)

    attempt["store"].read_result = blocked_read
    with ThreadPoolExecutor() as pool:
        future = pool.submit(supervisor.run, **attempt)
        try:
            assert reading.wait(5)
            supervisor.end_run()
            assert _payload(attempt).exists()
        finally:
            proceed.set()
        assert future.result(timeout=5).aborted
    assert _payload(attempt).exists()


@pytest.mark.parametrize("stage", ["running", "settling"])
def test_timeout_cancels_and_retains_payload_without_reading(tmp_path, stage):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    if stage == "running":
        handle.poll.return_value = RUNNING
    else:
        handle.wait_for_settlement.side_effect = TimeoutError("settlement timed out")
    attempt["options"] = TaskSupervisorOptions(worker_timeout=0.01, poll_interval=0.001)
    attempt["store"].read_result = Mock()
    with pytest.raises(TimeoutError):
        supervisor.run(**attempt)
    handle.cancel.assert_called_once()
    attempt["store"].read_result.assert_not_called()
    supervisor.end_run()
    assert _payload(attempt).exists()


@pytest.mark.parametrize("cancel_failure", [UNSETTLED, RuntimeError("cancel failed")])
def test_cancellation_failure_is_terminal_ownership_problem(tmp_path, cancel_failure):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    handle.poll.return_value = RUNNING
    if isinstance(cancel_failure, Exception):
        handle.cancel.side_effect = cancel_failure
    else:
        handle.cancel.return_value = cancel_failure
    attempt["options"] = TaskSupervisorOptions(worker_timeout=0.001, poll_interval=0.001)
    with pytest.raises((RuntimeError, TaskSettlementError)):
        supervisor.run(**attempt)
    assert supervisor.settlement_unconfirmed
    assert supervisor._active.handle is handle
    assert _payload(attempt).exists()


@pytest.mark.parametrize("corruption", ["digest", "identity", "pid", "missing_completion"])
def test_completion_validation_failures_preserve_evidence(tmp_path, corruption):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    original = attempt["launcher"].launch_task.side_effect

    def corrupt(request):
        original(request)
        completion_path = _payload(attempt).parent / "completion.json"
        record = json.loads(completion_path.read_text())
        if corruption == "digest":
            _payload(attempt).write_bytes(b"corrupt")
        elif corruption == "identity":
            record["identity"]["attempt_id"] = "stale"
            completion_path.write_text(json.dumps(record))
        elif corruption == "pid":
            handle.process_group_id = record["worker_pid"] + 1
        else:
            completion_path.unlink()
        return handle

    attempt["launcher"].launch_task.side_effect = corrupt
    with pytest.raises((ValueError, IncompleteTaskArtifactError)):
        supervisor.run(**attempt)
    supervisor.end_run()
    assert _payload(attempt).exists()
    assert _records(attempt)[-1]["event"] == "failed"


def test_application_failure_receipt_still_retains_failed_result(tmp_path):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)

    def failed_result(request):
        handle.request = request
        result = Shareable()
        result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
        attempt["store"].commit_result(attempt["bootstrap"].identity, result)
        return handle

    attempt["launcher"].launch_task.side_effect = failed_result
    outcome = supervisor.run(**attempt)
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
    supervisor.end_run()
    assert _payload(attempt).exists()


def test_worker_failure_without_completion_is_settled_before_validation(tmp_path):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    handle.wait_for_settlement.return_value = replace(SETTLED, exit_code=2)
    original = attempt["launcher"].launch_task.side_effect

    def incomplete(request):
        original(request)
        (_payload(attempt).parent / "completion.json").unlink()
        return handle

    attempt["launcher"].launch_task.side_effect = incomplete
    with pytest.raises(IncompleteTaskArtifactError):
        supervisor.run(**attempt)
    handle.wait_for_settlement.assert_called_once()
    handle.cancel.assert_not_called()
    supervisor.end_run()
    assert _payload(attempt).exists()


@pytest.mark.parametrize(
    "status",
    [replace(SETTLED, exit_code=7), replace(SETTLED, exit_code=None, termination_signal=signal.SIGKILL)],
)
def test_verified_completion_survives_later_process_failure_but_retains_evidence(tmp_path, status):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    handle.wait_for_settlement.return_value = status
    outcome = supervisor.run(**attempt)
    assert outcome.result["value"] == 2
    assert outcome.execution_status == status
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
    supervisor.end_run()
    assert _payload(attempt).exists()
    assert _records(attempt)[1]["termination_signal"] == status.termination_signal


def test_verified_completion_with_failure_diagnostic_retains_payload(tmp_path):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    original = attempt["launcher"].launch_task.side_effect

    def post_publication_failure(request):
        original(request)
        attempt["store"].record_failure(
            attempt["bootstrap"].identity,
            worker_pid=os.getpid(),
            worker_ppid=os.getppid(),
            started_at=1.0,
            failed_at=2.0,
            error_type="OSError",
            message="directory fsync failed after publication",
        )
        return handle

    attempt["launcher"].launch_task.side_effect = post_publication_failure
    outcome = supervisor.run(**attempt)
    assert outcome.result["value"] == 2
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
    supervisor.end_run()
    assert _payload(attempt).exists()
    assert (_payload(attempt).parent / "failure.json").exists()


@pytest.mark.parametrize("bootstrap_limit, store_limit", [(None, 1000), (1000, None), (1000, 2000)])
def test_store_and_worker_must_use_same_payload_policy(tmp_path, bootstrap_limit, store_limit):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    attempt["bootstrap"] = replace(attempt["bootstrap"], max_payload_bytes=bootstrap_limit)
    attempt["store"] = FileTaskArtifactStore(attempt["store"].root_dir, max_payload_bytes=store_limit)
    with pytest.raises(ValueError, match="payload policy"):
        supervisor.run(**attempt)
    attempt["launcher"].launch_task.assert_not_called()
    assert not Path(attempt["store"].attempt_dir(attempt["bootstrap"].identity)).exists()


def test_trusted_artifact_directory_alias_matches_store(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    Path(attempt["store"].root_dir).mkdir()
    alias = tmp_path / "artifact-alias"
    alias.symlink_to(attempt["store"].root_dir, target_is_directory=True)
    attempt["bootstrap"] = replace(attempt["bootstrap"], artifact_root=str(alias))
    assert supervisor.run(**attempt).result["value"] == 2


def test_abort_during_settlement_does_not_read_result(tmp_path):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)

    def abort_when_settled(**kwargs):
        attempt["abort_signal"].trigger(True)
        return SETTLED

    handle.wait_for_settlement.side_effect = abort_when_settled
    attempt["store"].read_result = Mock()
    assert supervisor.run(**attempt).aborted
    attempt["store"].read_result.assert_not_called()
    assert _payload(attempt).exists()


def test_mismatched_launch_request_does_not_launch_and_retains_staged_input(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    request_factory = attempt["request_factory"]
    attempt["request_factory"] = lambda identity, path: replace(request_factory(identity, path), attempt_id="stale")
    with pytest.raises(ValueError, match="does not match staged"):
        supervisor.run(**attempt)
    attempt["launcher"].launch_task.assert_not_called()
    supervisor.end_run()
    assert (_payload(attempt).parent / "input.fobs").exists()


def test_end_run_while_worker_lives_keeps_only_unresolved_payloads(tmp_path):
    supervisor = TaskSupervisor()
    first, _ = _attempt(tmp_path)
    outcome = supervisor.run(**first)
    supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
    second, handle = _attempt(tmp_path, attempt_id="attempt-2")
    cancelling, proceed = threading.Event(), threading.Event()
    handle.poll.return_value = RUNNING

    def blocked_cancel():
        cancelling.set()
        assert proceed.wait(5)
        return replace(SETTLED, cancel_requested=True)

    handle.cancel.side_effect = blocked_cancel
    second["options"] = TaskSupervisorOptions(worker_timeout=0.01, poll_interval=0.001)
    with ThreadPoolExecutor() as pool:
        running = pool.submit(supervisor.run, **second)
        ending = None
        try:
            assert cancelling.wait(5)
            ending = pool.submit(supervisor.end_run)
            supervisor.cleanup_job_payloads()
            assert not _payload(first).exists()
            assert _payload(second).exists()
        finally:
            proceed.set()
        with pytest.raises(TimeoutError):
            running.result(timeout=5)
        if ending:
            ending.result(timeout=5)
    assert not _payload(first).exists()
    assert _payload(second).exists()


def test_staging_failure_retains_partial_artifacts_and_fences_identity(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    attempt["store"].write_input = Mock(side_effect=OSError("stage failed"))
    with pytest.raises(OSError, match="stage failed"):
        supervisor.run(**attempt)
    attempt["launcher"].launch_task.assert_not_called()
    with pytest.raises(RuntimeError, match="already owned"):
        supervisor.run(**attempt)
    supervisor.end_run()
    assert Path(attempt["store"].attempt_dir(attempt["bootstrap"].identity)).is_dir()


def test_pretriggered_abort_and_zero_timeout_do_not_stage(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    attempt["abort_signal"].trigger(True)
    assert supervisor.run(**attempt).aborted
    attempt["abort_signal"].reset()
    with pytest.raises(TimeoutError):
        supervisor.run(**attempt, options=TaskSupervisorOptions(worker_timeout=0))
    assert not Path(attempt["store"].attempt_dir(attempt["bootstrap"].identity)).exists()
    attempt["launcher"].launch_task.assert_not_called()


def test_attempt_cannot_relaunch_through_different_launcher_or_task_name(tmp_path):
    supervisor = TaskSupervisor()
    attempt, _ = _attempt(tmp_path)
    supervisor.run(**attempt)
    second, _ = _attempt(tmp_path)
    second["bootstrap"] = replace(
        second["bootstrap"], identity=replace(second["bootstrap"].identity, task_name="other")
    )
    with pytest.raises(RuntimeError, match="already owned"):
        supervisor.run(**second)
    second["launcher"].launch_task.assert_not_called()


@pytest.mark.parametrize("name", ["worker_timeout", "settlement_timeout", "poll_interval"])
@pytest.mark.parametrize("value", [True, False, "1", -1, float("inf"), float("nan")])
def test_options_reject_invalid_timeouts(name, value):
    with pytest.raises(ValueError, match=name):
        TaskSupervisorOptions(**{name: value})


def test_released_attempts_drop_owned_handles_but_cannot_be_relaunched(tmp_path):
    supervisor = TaskSupervisor()
    released = []
    for index in range(12):
        attempt, _handle = _attempt(tmp_path, attempt_id=f"attempt-{index}")
        outcome = supervisor.run(**attempt)
        released.append(weakref.ref(supervisor._attempts[outcome.identity]))
        assert supervisor.record_receipt(outcome.identity, TaskResultReceipt.RECEIVED)
        assert not _payload(attempt).exists()
        assert all(reference() is None for reference in released)
        with pytest.raises(RuntimeError, match="already owned"):
            supervisor.run(**attempt)
    assert len(supervisor._launch_identities) == 12


def test_tensor_resend_retains_files_and_closed_receipt_preserves_active_mapping(tmp_path):
    import torch
    from safetensors.torch import load

    from nvflare.app_opt.pt.decomposers import TensorDecomposer
    from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable
    from nvflare.fuel.utils import fobs

    fobs.register(TensorDecomposer)
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    identity, store = attempt["bootstrap"].identity, attempt["store"]
    handle.poll.return_value = SETTLED

    def launch(request):
        handle.request = request
        store.commit_result(identity, Shareable({"params": {"w": torch.arange(4.0)}}))
        return handle

    attempt["launcher"].launch_task.side_effect = launch
    outcome = supervisor.run(**attempt)
    sender = TensorDownloadable(outcome.result["params"], 2 * 1024 * 1024)
    directory = Path(store.attempt_dir(identity))
    try:
        assert supervisor.record_receipt(identity, TaskResultReceipt.RETRY)
        assert (directory / "result.T0.safetensors").exists()
        saved = supervisor.read_saved_result(identity)
        assert torch.equal(saved["params"]["w"], torch.arange(4.0))
        assert supervisor.record_receipt(identity, TaskResultReceipt.TASK_CLOSED)
        assert not (directory / "result.T0.safetensors").exists()
        assert torch.equal(load(sender.produce_item(0))["w"], torch.arange(4.0))
    finally:
        sender.release()


def test_result_read_follows_confirmed_settlement(tmp_path):
    supervisor = TaskSupervisor()
    attempt, handle = _attempt(tmp_path)
    store = attempt["store"]
    events = []
    read_result = store.read_result
    handle.wait_for_settlement.side_effect = lambda **_: events.append("settled") or SETTLED
    store.read_result = lambda identity: (events.append("result"), read_result(identity))[1]

    outcome = supervisor.run(**attempt)

    assert outcome.result["value"] == 2
    assert outcome.completion.identity == outcome.identity
    assert not outcome.aborted
    assert events == ["settled", "result"]
    assert [r["event"] for r in _records(attempt)] == ["launched", "settled", "validated"]
    assert supervisor._active is None


@pytest.mark.parametrize("receipt", [TaskResultReceipt.RECEIVED, TaskResultReceipt.TASK_CLOSED])
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("read_failure", [False, True])
def test_saved_result_readers_hold_payloads_without_holding_compute(tmp_path, receipt, readers, read_failure):
    supervisor = TaskSupervisor()
    first, _ = _attempt(tmp_path)
    outcome = supervisor.run(**first)
    reading, proceed = threading.Barrier(readers + 1), threading.Event()
    read_result = first["store"].read_result

    def blocked_read(identity):
        reading.wait(timeout=5)
        assert proceed.wait(5)
        if read_failure:
            raise OSError("saved result read failed")
        return read_result(identity)

    first["store"].read_result = Mock(side_effect=blocked_read)
    with ThreadPoolExecutor(max_workers=readers) as pool:
        replays = [pool.submit(supervisor.read_saved_result, outcome.identity) for _ in range(readers)]
        try:
            reading.wait(timeout=5)
            assert supervisor.record_receipt(outcome.identity, receipt)
            supervisor.cleanup_job_payloads()
            assert _payload(first).exists()
            second, _ = _attempt(tmp_path, attempt_id="attempt-2")
            next_outcome = supervisor.run(**second)
            supervisor.record_receipt(next_outcome.identity, TaskResultReceipt.RECEIVED)
            assert not _payload(second).exists()
            assert _payload(first).exists()
        finally:
            proceed.set()
        for replay in replays:
            if read_failure:
                with pytest.raises(OSError, match="saved result read failed"):
                    replay.result(timeout=5)
            else:
                assert replay.result(timeout=5)["value"] == 2
    assert _payload(first).exists() is read_failure
