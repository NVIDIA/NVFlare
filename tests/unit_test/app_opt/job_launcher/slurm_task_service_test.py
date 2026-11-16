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

"""CP ownership must survive CJ exit, lost replies and in-flight submissions."""

import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.task_launcher_spec import TaskLauncherError, TaskLaunchError, TaskResourceRequest
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.slurm.config import CommandResult, LookupStatus, QueryResult, SubmissionResult
from nvflare.app_opt.job_launcher.slurm.launcher import _TaskCJHandle
from nvflare.app_opt.job_launcher.slurm.task_service import (
    ENV_TASK_OWNER,
    ENV_TASK_TOKEN,
    SlurmTaskClient,
    SlurmTaskService,
)
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.private.fed.task_worker.protocol import read_bootstrap, write_bootstrap
from tests.slurm_test_utils import _accounting, _backend, _query, _supervisor_case


def _eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "CP owner did not make progress"
        time.sleep(0.002)


@pytest.fixture(params=(0, 1), ids=("cpu", "gpu"))
def owned(tmp_path, monkeypatch, request):
    backend, adapter, staged = _backend(tmp_path, monkeypatch, request.param, cp_backend=True)
    backend.manager._monotonic = time.monotonic
    identity, store, arguments = _supervisor_case((backend, adapter, staged), write_result=False)
    Path(store.root_dir).mkdir()
    path = str(Path(staged.cwd, "bootstrap.json"))
    write_bootstrap(
        path,
        replace(
            arguments["bootstrap"],
            cell_session={
                "connect_url": "tcp://cj.example:8002",
                "worker_fqcn": "site-1.job-1.task_worker_fixture",
                "connection_security": "clear",
                "credentials": {},
            },
        ),
    )
    staged = replace(
        staged,
        argv=(
            sys.executable,
            "-m",
            "nvflare.private.fed.app.client.task_worker_process",
            "--bootstrap",
            path,
            "--parent_pid",
            "1234",
        ),
        resources=TaskResourceRequest(),
    )
    backend.bind_job(
        "job-1",
        "site-1",
        Workspace(backend.manager.config.workspace_path, site_name="site-1"),
        {"launcher_spec": {"site-1": {"slurm": {"gpus_per_node": request.param}}}},
    )
    adapter.active_by_id.return_value = _query("RUNNING")
    adapter.cancel.return_value = QueryResult(LookupStatus.FOUND)
    service = SlurmTaskService(SimpleNamespace(get_fqcn=lambda: "site-1", register_request_cb=Mock()))
    owner = service.add_job(backend, "job-1", "site-1", "test-token")
    monkeypatch.setenv(ENV_TASK_OWNER, "site-1")
    monkeypatch.setenv(ENV_TASK_TOKEN, "test-token")

    def send_request(**kwargs):
        message = kwargs["request"]
        message.set_header(MessageHeaderKey.ORIGIN, "site-1.job-1")
        return service._request(message)

    cell = SimpleNamespace(send_request=send_request)
    client = SlurmTaskClient(cell, "job-1", "site-1", 1, 0.001, 0.02)
    case = SimpleNamespace(
        backend=backend,
        adapter=adapter,
        request=staged,
        service=service,
        owner=owner,
        client=client,
        cell=cell,
    )
    try:
        yield case
    finally:
        adapter.active_by_id.return_value = _query()
        adapter.task_accounting.return_value = _accounting("CANCELLED")
        owner.release()
        owner.thread.join(5)
        assert not owner.thread.is_alive()


def test_cp_backend_rejects_unowned_cwd_and_resource_claims(owned, tmp_path):
    with pytest.raises(TaskLauncherError, match="cwd must be inside the site workspace"):
        owned.backend.launch_task(replace(owned.request, cwd=str(tmp_path)))
    with pytest.raises(TaskLauncherError, match="resources differ"):
        owned.backend.launch_task(replace(owned.request, resources=TaskResourceRequest(cpu_cores=1.5)))
    owned.adapter.submit.assert_not_called()


def _launch(case):
    handle = case.client.launch(case.request)
    _eventually(lambda: case.adapter.submit.call_count == 1 or handle.poll().settled)
    assert case.adapter.submit.call_count == 1, handle.poll()
    _eventually(lambda: handle.poll().phase.value == "running")
    return handle


def _settle(case):
    case.adapter.active_by_id.return_value = _query()
    case.adapter.task_accounting.return_value = _accounting("CANCELLED")
    case.owner.wake.set()


def test_cj_completion_is_bounded_while_cp_retains_unconfirmed_cleanup(owned, caplog):
    remote = _launch(owned)
    native = Mock()
    native.poll.return_value = JobReturnCode.SUCCESS
    cj = _TaskCJHandle(native, owned.owner, cleanup_timeout=0.01)
    cj.terminate()
    _eventually(lambda: owned.adapter.cancel.called)
    native.terminate.assert_called_once_with()
    assert cj.poll() == JobReturnCode.SUCCESS
    assert not remote.poll().settled
    started = time.monotonic()
    cj.wait()
    assert time.monotonic() - started < 1
    assert owned.owner.thread.is_alive()
    assert "job-1" in owned.service.jobs
    assert "cleanup unconfirmed" in caplog.text and "job-1" in caplog.text
    _settle(owned)


@pytest.mark.parametrize("loss", [None, "request", "reply", "retry"])
def test_verified_result_allows_next_assignment_while_cleanup_remains_owned(owned, loss, monkeypatch):
    owned.adapter.submit.side_effect = [
        SubmissionResult(CommandResult(0, str(job_id), ""), str(job_id)) for job_id in (42, 43)
    ]

    def accounting(job_id, *args, **kwargs):
        result = _accounting("CANCELLED")
        return replace(
            result, records=tuple(replace(r, job_id=r.job_id.replace("42", job_id, 1)) for r in result.records)
        )

    owned.adapter.task_accounting.side_effect = accounting
    first = _launch(owned)
    send = owned.cell.send_request
    completions = []
    launches = []
    unavailable = bool(loss)

    def transport(**kwargs):
        action = json.loads(kwargs["request"].payload)["action"]
        if action == "launch":
            launches.append(action)
        if action == "complete":
            completions.append(action)
            if unavailable:
                if loss == "reply":
                    send(**kwargs)
                raise TimeoutError("completion request or reply lost")
        return send(**kwargs)

    monkeypatch.setattr(owned.cell, "send_request", transport)
    if loss:
        with pytest.raises(TimeoutError):
            first.result_received()
    else:
        first.result_received()
    assert not first.poll().settled
    assert owned.owner.attempts[("task-1", "attempt-1")].completed == (loss in (None, "reply"))
    bootstrap = read_bootstrap(owned.request.argv[4])
    next_path = str(Path(owned.request.cwd, "bootstrap-2.json"))
    write_bootstrap(
        next_path, replace(bootstrap, identity=replace(bootstrap.identity, task_id="task-2", attempt_id="attempt-2"))
    )
    argv = (*owned.request.argv[:4], next_path, *owned.request.argv[5:])
    next_request = replace(owned.request, task_id="task-2", attempt_id="attempt-2", argv=argv)
    if loss == "retry":
        with pytest.raises(TaskLauncherError):
            owned.client.launch(next_request)
        assert not launches, "unconfirmed completion must be retried before another launch request"
        assert ("task-2", "attempt-2") not in owned.owner.attempts
        assert owned.adapter.submit.call_count == 1
    unavailable = False
    second = owned.client.launch(next_request)
    _eventually(lambda: owned.adapter.submit.call_count == 2)
    assert not second.poll().settled
    assert len(owned.owner.attempts) == 2
    replay = owned.client.launch(owned.request)
    assert replay.execution_id == first.execution_id
    assert owned.adapter.submit.call_count == 2
    assert owned.owner.attempts[("task-1", "attempt-1")].handle is not None
    assert owned.owner.attempts[("task-1", "attempt-1")].completed
    _eventually(lambda: owned.adapter.cancel.called)
    assert len(completions) == {None: 1, "request": 2, "reply": 2, "retry": 3}[loss]


def test_result_acknowledgement_requires_an_existing_owned_attempt(owned):
    with pytest.raises(TaskLauncherError):
        owned.client.call("complete", task_id="unknown", attempt_id="unknown")
    assert not owned.owner.attempts
    owned.adapter.submit.assert_not_called()


def test_old_operator_release_record_cannot_settle_a_live_allocation(owned):
    remote = _launch(owned)
    native = owned.owner.attempts[("task-1", "attempt-1")].handle
    Path(native.job_dir, "operator-release.json").write_text(
        json.dumps(
            dict(identity=list(owned.request.identity), execution_id=remote.execution_id, uid=os.getuid(), reason="old")
        )
    )
    assert not remote.poll().settled
    assert owned.owner.attempts[("task-1", "attempt-1")].handle is native


def test_cp_detects_cj_loss_without_cj_cleanup_rpc(owned):
    _launch(owned)
    native = Mock()
    native.poll.return_value = 1  # Native CJ handle has proven the terminal allocation.
    owned.owner.attach_cj(native)
    _eventually(lambda: owned.owner.released and owned.adapter.cancel.called)
    assert owned.owner.thread.is_alive()
    assert owned.owner.closed
    _settle(owned)
    owned.owner.thread.join(5)
    assert not owned.owner.thread.is_alive()


def test_cj_exit_during_sbatch_retains_late_allocation_for_cancellation(owned):
    entered, finish = threading.Event(), threading.Event()

    def submit(*args, **kwargs):
        entered.set()
        finish.wait()
        return SubmissionResult(CommandResult(0, "42", ""), "42")

    owned.adapter.submit.side_effect = submit
    try:
        owned.client.launch(owned.request)
        assert entered.wait(5)
        owned.owner.release()
        assert owned.owner.thread.is_alive()
        assert "job-1" in owned.service.jobs
    finally:
        finish.set()
    _eventually(lambda: owned.adapter.cancel.called)
    assert owned.owner.thread.is_alive()
    _settle(owned)
    owned.owner.thread.join(5)
    assert not owned.owner.thread.is_alive()


def test_lost_cell_launch_reply_does_not_lose_or_duplicate_allocation(owned):
    send = owned.cell.send_request

    def lose_reply(**kwargs):
        send(**kwargs)
        raise TimeoutError("reply lost after CP accepted the request")

    owned.cell.send_request = lose_reply
    with pytest.raises(TaskLaunchError) as failure:
        owned.client.launch(owned.request)
    owned.cell.send_request = send
    _eventually(lambda: failure.value.handle.poll().phase.value == "running")
    replay = owned.client.launch(owned.request)
    assert replay.execution_id == failure.value.handle.execution_id
    assert owned.adapter.submit.call_count == 1
    owned.owner.release()
    _eventually(lambda: owned.adapter.cancel.called)
    assert owned.owner.thread.is_alive()


def test_cancel_before_delayed_launch_fences_the_attempt(owned):
    owned.client.call("cancel", task_id="task-1", attempt_id="attempt-1")
    with pytest.raises(TaskLaunchError):
        owned.client.launch(owned.request)
    assert not owned.adapter.submit.called


def test_controller_outage_does_not_release_cleanup_or_job_resources(owned):
    _launch(owned)
    owned.adapter.active_by_id.return_value = QueryResult(LookupStatus.UNAVAILABLE)
    owned.owner.release()
    _eventually(lambda: owned.owner.attempts[("task-1", "attempt-1")].handle.cancel_requested)
    assert not owned.adapter.cancel.called, "cancellation requires a healthy ownership observation"
    assert owned.owner.thread.is_alive()
    assert "job-1" in owned.service.jobs
    owned.adapter.active_by_id.return_value = _query("RUNNING")
    _eventually(lambda: owned.adapter.cancel.called)
    _settle(owned)
    owned.owner.thread.join(5)
    assert not owned.owner.thread.is_alive()


@pytest.mark.parametrize(
    "field,value",
    [
        ("token", "wrong"),
        ("job_id", "other-job"),
        ("origin", "site-1.other-job"),
        ("resources", {"gpu_count": 8}),
        ("bootstrap", "/etc/passwd"),
        ("cwd", "/etc"),
        ("environment", {"SLURM_JOB_ID": "42"}),
        ("executable", "/bin/bash"),
    ],
)
def test_control_channel_rejects_wrong_identity_and_caller_authority(owned, field, value):
    data = dict(
        job_id="job-1",
        token="test-token",
        action="launch",
        task_id="task-1",
        attempt_id="attempt-1",
        bootstrap=owned.request.argv[4],
        cwd=owned.request.cwd,
    )
    origin = "site-1.job-1"
    if field == "origin":
        origin = value
    else:
        data[field] = value
    reply = owned.service._request(new_cell_message({MessageHeaderKey.ORIGIN: origin}, json.dumps(data)))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) in (ReturnCode.INVALID_REQUEST, ReturnCode.UNAUTHENTICATED)
    owned.adapter.submit.assert_not_called()


def test_end_run_seals_admission_and_cp_shutdown_waits_for_release(owned):
    _launch(owned)
    owned.client.close()
    _eventually(lambda: owned.adapter.cancel.called)
    assert owned.owner.closed and not owned.owner.released
    shutdown = threading.Thread(target=owned.service.shutdown)
    shutdown.start()
    try:
        _eventually(lambda: owned.owner.released)
        assert shutdown.is_alive()
        _settle(owned)
        shutdown.join(5)
        assert not shutdown.is_alive()
    finally:
        _settle(owned)
        shutdown.join(5)


def test_lost_sbatch_reply_is_reconciled_only_for_cleanup(owned):
    owned.adapter.submit.side_effect = TimeoutError("sbatch response lost")
    owned.adapter.task_by_marker.return_value = _query("RUNNING")
    remote = owned.client.launch(owned.request)
    _eventually(lambda: owned.adapter.cancel.called)
    assert owned.adapter.submit.call_count == 1
    assert not remote.poll().settled
    _eventually(lambda: remote.poll().phase.value == "running")
    assert ":42:" in remote.execution_id
    owned.owner.release()
    assert owned.owner.thread.is_alive()
    _settle(owned)
    owned.owner.thread.join(5)
    assert not owned.owner.thread.is_alive()


def test_missing_marker_after_submission_timeout_never_proves_release(owned):
    owned.adapter.submit.side_effect = TimeoutError("sbatch response lost")
    owned.adapter.task_by_marker.return_value = _query()
    remote = owned.client.launch(owned.request)
    _eventually(lambda: owned.adapter.task_by_marker.called)
    owned.owner.release()
    assert not remote.poll().settled
    assert owned.owner.thread.is_alive()
    assert "job-1" in owned.service.jobs
    # A later exact match permits cancellation and accounting-based release.
    owned.adapter.task_by_marker.return_value = _query("RUNNING")
    _eventually(lambda: owned.adapter.cancel.called)
    _settle(owned)


def test_settled_attempt_keeps_replay_fence_without_retaining_native_handle(owned):
    remote = _launch(owned)
    _settle(owned)
    _eventually(lambda: remote.poll().settled)
    attempt = owned.owner.attempts[("task-1", "attempt-1")]
    assert attempt.handle is None
    assert not owned.backend.manager._handles
    replay = owned.client.launch(owned.request)
    assert replay.poll().settled
    assert replay.execution_id == remote.execution_id
    assert owned.adapter.submit.call_count == 1


def test_nonzero_sbatch_with_id_requests_cancel_without_cj_intervention(owned):
    owned.adapter.submit.return_value = SubmissionResult(
        CommandResult(1, "42", "accepted but post-submit failed"), "42"
    )
    owned.adapter.submit.side_effect = None
    remote = _launch(owned)
    _eventually(lambda: owned.adapter.cancel.called)
    assert not owned.owner.closed and not owned.owner.released
    assert not remote.poll().settled
    assert "accepted but post-submit failed" in remote.poll().failure_reason
    _settle(owned)


def test_closed_job_rejects_a_new_attempt_at_the_cp_service(owned):
    remote = _launch(owned)
    _settle(owned)
    _eventually(lambda: remote.poll().settled)
    owned.client.close()
    with pytest.raises(TaskLauncherError, match="rejected"):
        owned.client.call(
            "launch", task_id="task-2", attempt_id="attempt-2", bootstrap=owned.request.argv[4], cwd=owned.request.cwd
        )
    assert ("task-2", "attempt-2") not in owned.owner.attempts
    assert owned.adapter.submit.call_count == 1


def test_proxy_settlement_wait_refreshes_cp_evidence_before_long_poll_interval(owned):
    owned.backend.manager.config = replace(owned.backend.manager.config, poll_interval=10.0)
    owned.client.poll_interval = 10.0
    remote = _launch(owned)
    native = owned.owner.attempts[("task-1", "attempt-1")].handle
    assert native._next_query_at > time.monotonic()
    owned.adapter.active_by_id.return_value = _query()
    owned.adapter.task_accounting.return_value = _accounting()
    assert remote.wait_for_settlement(timeout=2.5).succeeded
    owned.adapter.cancel.assert_not_called()


def test_proxy_settlement_timeout_does_not_sleep_past_deadline(owned):
    owned.client.poll_interval = 10.0
    remote = _launch(owned)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="settlement unconfirmed"):
        remote.wait_for_settlement(timeout=0.05)
    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize("full_history", [False, True])
def test_closed_admission_settles_cancellation_of_a_rejected_launch(owned, full_history):
    if full_history:
        owned.backend.max_attempts = 1
        owned.client.call("cancel", task_id="earlier-task", attempt_id="earlier-attempt")
    owned.client.call("close")
    with pytest.raises(TaskLaunchError) as failure:
        owned.client.launch(owned.request)

    # A failed regression must report unconfirmed cleanup immediately, rather
    # than spending the normal cleanup deadline waiting for a nonexistent job.
    owned.client.cleanup_timeout = 0
    status = failure.value.handle.cancel()
    assert status.settled and status.cancel_requested
    assert status.phase.value == "terminal"
    assert failure.value.handle.execution_id == "slurm:pending"
    assert len(owned.owner.attempts) == int(full_history)
    assert ("task-1", "attempt-1") not in owned.owner.attempts
    assert not owned.adapter.submit.called
    assert not owned.adapter.cancel.called


def test_closed_admission_still_observes_an_existing_live_allocation(owned):
    remote = _launch(owned)
    owned.client.call("close")
    status = owned.client.call("cancel", task_id=remote.request.task_id, attempt_id=remote.request.attempt_id)
    assert not status["settled"]
    assert not remote.poll().settled
    _eventually(lambda: owned.adapter.cancel.called)
    assert owned.owner.thread.is_alive()
    _settle(owned)
    assert remote.wait_for_settlement(timeout=1).settled


@pytest.mark.parametrize("during_poll", [False, True])
def test_cancel_intent_survives_successful_settlement_cache(owned, during_poll):
    owned.client.cleanup_timeout = 1
    remote = _launch(owned)
    entered, release = threading.Event(), threading.Event()

    def final_accounting(*args, **kwargs):
        entered.set()
        release.wait()
        return _accounting()

    if during_poll:
        owned.adapter.task_accounting.side_effect = final_accounting
    owned.adapter.active_by_id.return_value = _query()
    owned.owner.wake.set()
    try:
        if during_poll:
            assert entered.wait(5), "owner did not enter final native poll"
            owned.client.call("cancel", task_id=remote.request.task_id, attempt_id=remote.request.attempt_id)
            release.set()
        else:
            _eventually(lambda: remote.poll().succeeded)
        status = remote.cancel()
        assert status.settled and status.cancel_requested and not status.succeeded
        assert remote.poll().cancel_requested
    finally:
        release.set()


def test_single_unsettled_attempt_is_enforced_by_the_cp_service(owned):
    _launch(owned)
    with pytest.raises(TaskLauncherError, match="rejected"):
        owned.client.call(
            "launch", task_id="task-2", attempt_id="attempt-2", bootstrap=owned.request.argv[4], cwd=owned.request.cwd
        )
    assert not owned.owner.closed
    assert ("task-2", "attempt-2") not in owned.owner.attempts
    assert owned.adapter.submit.call_count == 1


def test_shutdown_deadline_retains_and_reports_unsettled_jobs(owned, caplog):
    _launch(owned)
    started = time.monotonic()
    assert owned.service.shutdown(timeout=0.02) == ("job-1",)
    assert time.monotonic() - started < 1
    assert owned.owner.thread.is_alive()
    assert "job-1" in owned.service.jobs
    assert "job-1" in caplog.text and "task-1" in caplog.text
    _settle(owned)


def test_definitive_rejection_allows_cj_and_system_shutdown(owned):
    reason = "Invalid partition name specified"
    owned.adapter.submit.side_effect = None
    owned.adapter.submit.return_value = SubmissionResult(
        CommandResult(
            1,
            "",
            "sbatch: error: job_submit.lua: applying site policy\n"
            "sbatch: error: Batch job submission failed: " + reason,
        )
    )
    remote = owned.client.launch(owned.request)
    _eventually(lambda: remote.poll().settled)
    assert reason in remote.poll().failure_reason
    native = Mock()
    native.poll.return_value = 1
    cj = _TaskCJHandle(native, owned.owner)
    cj.wait()
    assert cj.poll() == 1
    assert not owned.owner.thread.is_alive()
    assert owned.service.shutdown(timeout=0.05) == ()


def test_cj_recovery_waits_for_workers_to_settle(owned):
    _launch(owned)
    native = Mock()
    native.can_restart.return_value = True
    cj = _TaskCJHandle(native, owned.owner, cleanup_timeout=0.05)
    assert cj.can_restart() is False
    assert owned.owner.thread.is_alive()
    _settle(owned)
    cj.cleanup_timeout = 5
    assert cj.can_restart() is True
    assert not owned.owner.thread.is_alive()
    assert "job-1" not in owned.service.jobs


def test_cj_recovery_requires_confirmed_cj_exit(owned):
    native = Mock()
    native.can_restart.return_value = False
    cj = _TaskCJHandle(native, owned.owner, cleanup_timeout=0.05)
    assert cj.can_restart() is False
    assert not owned.owner.released
    owned.owner.release()
