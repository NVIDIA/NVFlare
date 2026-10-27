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
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.task_launcher_spec import TaskLaunchError, TaskResourceRequest
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.slurm.config import CommandResult, LookupStatus, QueryResult, SubmissionResult
from nvflare.app_opt.job_launcher.slurm.task_job_launcher import _TaskCJHandle
from nvflare.app_opt.job_launcher.slurm.task_service import (
    ENV_TASK_OWNER,
    ENV_TASK_TOKEN,
    SlurmTaskClient,
    SlurmTaskService,
)
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.private.fed.task_worker.protocol import write_bootstrap
from tests.unit_test.app_opt.job_launcher.slurm_task_launcher_test import (
    _accounting,
    _backend,
    _query,
    _supervisor_case,
)


def _eventually(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "CP owner did not make progress"
        time.sleep(0.002)


@pytest.fixture(params=(0, 1), ids=("cpu", "gpu"))
def owned(tmp_path, monkeypatch, request):
    backend, adapter, staged = _backend(tmp_path, monkeypatch, request.param)
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
        adapter.task_accounting.return_value = _accounting("CANCELLED", step=None)
        owner.release()
        owner.thread.join(5)
        assert not owner.thread.is_alive()


def _launch(case):
    handle = case.client.launch(case.request)
    _eventually(lambda: case.adapter.submit.call_count == 1 or handle.poll().settled)
    assert case.adapter.submit.call_count == 1, handle.poll()
    _eventually(lambda: handle.poll().phase.value == "running")
    return handle


def _settle(case):
    case.adapter.active_by_id.return_value = _query()
    case.adapter.task_accounting.return_value = _accounting("CANCELLED", step=None)
    case.owner.wake.set()


def test_cj_abort_cannot_release_job_until_task_allocation_settles(owned):
    remote = _launch(owned)
    native = Mock()
    native.poll.return_value = 1
    cj = _TaskCJHandle(native, owned.owner)
    cj.terminate()
    _eventually(lambda: owned.adapter.cancel.called)
    native.terminate.assert_called_once_with()
    assert cj.poll() == JobReturnCode.UNKNOWN
    assert not remote.poll().settled
    waiter = threading.Thread(target=cj.wait)
    waiter.start()
    try:
        assert waiter.is_alive()
        assert "job-1" in owned.service.jobs
        _settle(owned)
        waiter.join(5)
        assert not waiter.is_alive()
        assert cj.poll() == 1
        assert "job-1" not in owned.service.jobs
    finally:
        _settle(owned)
        waiter.join(5)


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
        assert finish.wait(5)
        return SubmissionResult(CommandResult(0, "42", ""), "42")

    owned.adapter.submit.side_effect = submit
    owned.client.launch(owned.request)
    assert entered.wait(5)
    try:
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
