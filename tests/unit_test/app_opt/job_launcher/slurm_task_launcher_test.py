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
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import TaskResultReceipt
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskLauncherError,
    TaskLaunchError,
    TaskLaunchRequest,
    TaskResourceRequest,
    TaskSettlementError,
    UnsupportedTaskResourceError,
)
from nvflare.app_opt.job_launcher.slurm.config import (
    SLURM_CHILD_PROCESS_ENV,
    CommandResult,
    LookupStatus,
    QueryResult,
    SlurmLauncherError,
    SlurmProtocolError,
    SlurmRecord,
    SubmissionResult,
)
from nvflare.app_opt.job_launcher.slurm.task_launcher import SlurmTaskLauncher, SlurmTaskManager, _batch_script
from nvflare.app_opt.job_launcher.slurm.task_scheduler import SlurmTaskCliAdapter, TaskSlurmRecord
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore, IncompleteTaskArtifactError
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor, TaskSupervisorOptions
from tests.unit_test.private.fed.conftest import isolated_fobs  # noqa: F401


def _query(state=None):
    return (
        QueryResult(LookupStatus.FOUND, (SlurmRecord("42", state),)) if state else QueryResult(LookupStatus.NOT_FOUND)
    )


def _accounting(state="COMPLETED", step="COMPLETED", **kwargs):
    records = [TaskSlurmRecord("42", state, **kwargs)]
    if step:
        records.append(TaskSlurmRecord("42.batch", step))
    return QueryResult(LookupStatus.FOUND, tuple(records))


def _backend(tmp_path, monkeypatch, gpu_count):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    (root / "startup").mkdir()
    (root / "local").mkdir()
    (root / "job-1").mkdir()
    launcher = SlurmTaskLauncher(
        workspace_path=str(root),
        shared_filesystem=True,
        cgroup_enforced=True,
        sbatch_directives={"account": "site-account", "partition": "task-compute", "time": "00:05:00"},
        cancel_timeout=0.1,
        poll_interval=0.001,
    )
    adapter = Mock()
    adapter.accounting_probe.return_value = CommandResult(0, "", "")
    adapter.submit.return_value = SubmissionResult(CommandResult(0, "42\n", ""), "42")
    adapter.active_by_id.return_value = _query()
    adapter.task_accounting.return_value = _accounting()
    # Inject only the transport. Exercise real manager initialization/files.
    launcher.manager.adapter = adapter
    launcher.manager.initialize()
    # Advance a deterministic clock on each access, keeping most tests instant.
    ticks = iter(range(1, 1000000))
    launcher.manager._monotonic = lambda: next(ticks) * 0.002
    monkeypatch.setenv(SLURM_CHILD_PROCESS_ENV, "1")
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    launch_request = TaskLaunchRequest(
        "job-1",
        "site-1",
        "task-1",
        "attempt-1",
        (sys.executable, "-V"),
        cwd=str(root / "job-1"),
        resources=TaskResourceRequest(gpu_count=gpu_count),
    )
    return launcher, adapter, launch_request


@pytest.fixture(params=(0, 1), ids=("cpu", "gpu"))
def backend(tmp_path, monkeypatch, request):
    return _backend(tmp_path, monkeypatch, request.param)


def test_submission_and_resources_are_separate_from_cj(backend, monkeypatch):
    launcher, adapter, request = backend
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
    request = replace(request, resources=TaskResourceRequest(cpu_cores=3, memory_mb=4096, gpu_count=2))
    handle = launcher.launch_task(request)
    argv = adapter.submit.call_args.args[0]
    for value in (
        "--nodes=1",
        "--ntasks=1",
        "--no-requeue",
        "--export=NIL",
        "--gres=gpu:2",
        "--cpus-per-task=3",
        "--mem=4096M",
        "--account=site-account",
        "--partition=task-compute",
    ):
        assert value in argv
    assert "--time=00:05:00" in argv
    assert handle.execution_id.startswith("slurm:local:42:")
    assert json.loads(Path(handle.job_dir, "identity.json").read_text())["attempt_id"] == "attempt-1"
    assert Path(handle.job_dir, "batch.sh").stat().st_mode & 0o777 == 0o700
    assert not Path(handle.job_dir, "secret.env").exists()
    assert handle.poll().succeeded
    assert Path(handle.job_dir, "batch.sh").exists()
    assert Path(handle.job_dir, "scheduler.jsonl").exists()


def test_cpu_and_gpu_allocation_and_unique_attempts(backend):
    launcher, adapter, request = backend
    first = launcher.launch_task(request)
    expected_gres = f"--gres=gpu:{request.resources.gpu_count}" if request.resources.gpu_count else "--gres=none"
    assert expected_gres in adapter.submit.call_args.args[0]
    second = launcher.launch_task(replace(request, attempt_id="attempt-2"))
    assert first.job_name != second.job_name
    assert first.execution_id != second.execution_id
    assert first.job_dir != second.job_dir
    with pytest.raises(SlurmLauncherError, match="stale Slurm artifacts"):
        launcher.launch_task(request)
    assert adapter.submit.call_count == 2


@pytest.mark.parametrize(
    "state,phase",
    [
        ("PENDING", "queued"),
        ("RUNNING", "running"),
        ("COMPLETING", "running"),
        ("COMPLETED", "terminal"),
        ("UNKNOWN_STATE", "running"),
        ("SUSPENDED", "running"),
    ],
)
def test_visible_allocation_cannot_settle_even_after_leader_exit(backend, state, phase):
    launcher, adapter, request = backend
    adapter.active_by_id.return_value = _query(state)
    status = launcher.launch_task(request).poll()
    assert status.phase.value == phase
    assert not status.settled
    adapter.task_accounting.assert_not_called()


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETING", "REQUEUED", "UNKNOWN_STATE"])
def test_unfinished_step_keeps_allocation_unsettled(backend, state):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = _accounting(step=state)
    status = launcher.launch_task(request).poll()
    assert status.phase == TaskExecutionPhase.TERMINAL
    assert not status.settled


def test_extern_and_nonleader_steps_are_checked(backend):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = QueryResult(
        LookupStatus.FOUND,
        (
            TaskSlurmRecord("42", "COMPLETED"),
            TaskSlurmRecord("42.batch", "COMPLETED"),
            TaskSlurmRecord("42.extern", "COMPLETED"),
            TaskSlurmRecord("42.0", "RUNNING"),
        ),
    )
    assert not launcher.launch_task(request).poll().settled


@pytest.mark.parametrize(
    "accounting", [QueryResult(LookupStatus.NOT_FOUND), QueryResult(LookupStatus.UNAVAILABLE), _accounting(step=None)]
)
def test_missing_accounting_never_becomes_release_evidence(backend, accounting):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = accounting
    handle = launcher.launch_task(request)
    for _ in range(12):
        assert not handle.poll().settled
    assert Path(handle.job_dir, "batch.sh").exists()


@pytest.mark.parametrize(
    "kwargs", [{"exit_status": 3}, {"derived_exit_status": 2}, {"exit_signal": 9}, {"derived_exit_signal": 15}]
)
def test_allocation_and_step_failures_are_attributable(backend, kwargs):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = _accounting(**kwargs)
    status = launcher.launch_task(request).poll()
    assert status.settled and not status.succeeded
    assert "42" in status.failure_reason
    assert status.termination_signal == (kwargs.get("exit_signal") or kwargs.get("derived_exit_signal"))


def test_controller_recheck_blocks_completion_race(backend):
    launcher, adapter, request = backend
    adapter.active_by_id.side_effect = [_query(), _query("COMPLETING")]
    assert not launcher.launch_task(request).poll().settled


@pytest.mark.parametrize("initial", ["PENDING", "RUNNING"])
def test_cancel_requires_scheduler_release(backend, initial):
    launcher, adapter, request = backend
    adapter.active_by_id.return_value = _query(initial)
    handle = launcher.launch_task(request)

    def cancelled(*args, **kwargs):
        assert not handle._last_status.settled
        adapter.active_by_id.return_value = _query()
        adapter.task_accounting.return_value = _accounting(state="CANCELLED", step=None)
        return QueryResult(LookupStatus.FOUND)

    adapter.cancel.side_effect = cancelled
    status = handle.cancel()
    assert status.settled and status.cancel_requested and not status.succeeded
    adapter.cancel.assert_called_once_with("42", timeout=launcher.manager.config.cancel_timeout)


@pytest.mark.parametrize("queue", [_query("RUNNING"), QueryResult(LookupStatus.UNAVAILABLE)])
def test_cancel_ack_or_unavailable_queue_cannot_settle(backend, queue):
    launcher, adapter, request = backend
    adapter.active_by_id.return_value = queue
    handle = launcher.launch_task(request)
    with pytest.raises(TaskSettlementError) as error:
        handle.cancel()
    assert not error.value.status.settled
    assert Path(handle.job_dir).is_dir()


def test_cancel_after_completion_preserves_cancellation_intent(backend):
    launcher, adapter, request = backend
    handle = launcher.launch_task(request)
    assert handle.poll().succeeded
    status = handle.cancel()
    assert status.settled and status.cancel_requested and not status.succeeded
    adapter.cancel.assert_not_called()


def test_pending_timeout_requests_cancel_without_asserting_release(backend):
    launcher, adapter, request = backend
    adapter.active_by_id.return_value = _query("PENDING")
    handle = launcher.launch_task(request)
    launcher.manager._monotonic = lambda: 1000
    assert handle.poll().cancel_requested
    assert not handle.poll().settled
    adapter.cancel.assert_called_once()


def test_queries_are_throttled_even_when_supervisor_polls_quickly(backend):
    launcher, adapter, request = backend
    adapter.active_by_id.return_value = _query("RUNNING")
    launcher.manager._monotonic = lambda: 1
    handle = launcher.launch_task(request)
    for _ in range(100):
        handle.poll()
    adapter.active_by_id.assert_called_once()


@pytest.mark.parametrize(
    "submission",
    [
        SubmissionResult(CommandResult(None, "", "", timed_out=True)),
        SubmissionResult(CommandResult(1, "42", "error"), "42"),
        SubmissionResult(CommandResult(0, "42;other", ""), "42", "other"),
    ],
)
def test_uncertain_submission_retains_cleanup_handle_and_identity_fence(backend, submission):
    launcher, adapter, request = backend
    adapter.submit.return_value = submission
    with pytest.raises(TaskLaunchError) as error:
        launcher.launch_task(request)
    handle = error.value.handle
    assert handle.request == request
    assert Path(handle.job_dir, "batch.sh").exists()
    with pytest.raises(SlurmLauncherError, match="stale Slurm artifacts"):
        launcher.launch_task(request)
    if not submission.job_id or submission.cluster:
        assert not handle.poll().settled
        with pytest.raises(TaskSettlementError):
            handle.cancel()
        adapter.cancel.assert_not_called()


def test_submit_exception_and_shutdown(backend):
    launcher, adapter, request = backend
    adapter.submit.side_effect = OSError("connection lost")
    with pytest.raises(TaskLaunchError) as error:
        launcher.launch_task(request)
    assert not error.value.handle.poll().settled
    launcher.manager.shutdown()
    with pytest.raises(TaskLauncherError, match="shutting down"):
        launcher.launch_task(replace(request, attempt_id="another"))


def test_bad_query_is_unconfirmed_and_retained(backend):
    launcher, adapter, request = backend
    adapter.active_by_id.side_effect = SlurmProtocolError("ownership mismatch")
    handle = launcher.launch_task(request)
    assert not handle.poll().settled
    assert "SlurmProtocolError" in handle.poll().failure_reason
    with pytest.raises(TimeoutError):
        handle.wait_for_settlement(0)


@pytest.mark.parametrize(
    "name",
    [
        JobProcessEnv.AUTH_TOKEN,
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SBATCH_GRES",
        "NVFLARE_CLIENT_API_CONFIG",
        "BASH_FUNC_bad%%",
    ],
)
def test_reject_request_environment_authority(backend, name):
    launcher, adapter, request = backend
    with pytest.raises((TaskLauncherError, SlurmLauncherError)):
        launcher.launch_task(replace(request, environment={name: "value"}))
    adapter.submit.assert_not_called()


def test_reject_mixed_mode_paths_and_fractional_cpu(backend, monkeypatch, tmp_path):
    launcher, adapter, request = backend
    monkeypatch.delenv(SLURM_CHILD_PROCESS_ENV)
    with pytest.raises(TaskLauncherError, match="Slurm Client Job"):
        launcher.launch_task(request)
    monkeypatch.setenv(SLURM_CHILD_PROCESS_ENV, "1")
    with pytest.raises(TaskLauncherError, match="inside the site workspace"):
        launcher.launch_task(replace(request, cwd=str(tmp_path)))
    with pytest.raises(TaskLauncherError, match="absolute executable"):
        launcher.launch_task(replace(request, argv=("python", "-V")))
    with pytest.raises(UnsupportedTaskResourceError):
        launcher.launch_task(replace(request, resources=TaskResourceRequest(cpu_cores=1.5)))
    with pytest.raises(TaskLauncherError, match="Process parent-PID"):
        launcher.launch_task(
            replace(request, argv=(sys.executable, "-m", "nvflare.private.fed.app.client.task_worker_process"))
        )
    adapter.submit.assert_not_called()


@pytest.mark.parametrize("kwargs", [{}, {"shared_filesystem": True}, {"cgroup_enforced": True}])
def test_require_explicit_operator_profile(tmp_path, kwargs):
    with pytest.raises(ValueError, match="coherent shared paths"):
        SlurmTaskLauncher(workspace_path=str(tmp_path), **kwargs)


def test_check_filesystem_fails_before_submit(backend, monkeypatch):
    launcher, adapter, request = backend
    from nvflare.app_opt.job_launcher.slurm import task_launcher

    manager = SlurmTaskManager(launcher.manager.config, launcher.logger, adapter=adapter)
    monkeypatch.setattr(task_launcher.os, "link", Mock(side_effect=OSError("no hardlinks")))
    with pytest.raises(OSError, match="no hardlinks"):
        manager.initialize()
    assert not manager._initialized
    with pytest.raises(OSError, match="no hardlinks"):
        manager.initialize()
    adapter.submit.assert_not_called()


def test_batch_quoting_and_environment_are_explicit(backend, monkeypatch):
    launcher, adapter, request = backend
    output = Path(request.cwd, "command.json")
    malicious = "literal $(touch injected); 'quoted'\nnext"
    code = "import json,os,sys; open(sys.argv[1],'w').write(json.dumps([sys.argv[2],dict(os.environ)]))"
    request = replace(
        request,
        argv=(sys.executable, "-c", code, str(output), malicious),
        environment={"PATH": "/usr/bin:/bin", "SETTING": malicious},
    )
    script = Path(request.cwd, "batch.sh")
    script.write_text(_batch_script(request, "export SETTING='site runtime'"))
    subprocess.run(
        ["/bin/bash", str(script)],
        check=True,
        timeout=20,
        env={"SLURM_JOB_ID": "42", "CJ_PRIVATE": "not forwarded", "CUDA_VISIBLE_DEVICES": "7", "SLURM_JWT": "secret"},
    )
    argument, environment = json.loads(output.read_text())
    assert argument == malicious
    assert environment["SETTING"] == "site runtime"
    assert environment["SLURM_JOB_ID"] == "42"
    assert environment["CUDA_VISIBLE_DEVICES"] == ("7" if request.resources.gpu_count else "")
    assert "CJ_PRIVATE" not in environment
    assert "SLURM_JWT" not in environment
    assert not Path(request.cwd, "injected").exists()


def _cli(response):
    runner = Mock(return_value=CommandResult(0, response, ""))
    adapter = SlurmTaskCliAdapter({"sacct": "/usr/bin/sacct"}, os.getuid(), logging.getLogger("test"), runner=runner)
    return adapter, runner


@pytest.mark.parametrize("visibility,returncode", [("2", 0), (None, 102)])
def test_gpu_worker_requires_allocation_visibility(backend, visibility, returncode):
    _, _, request = backend
    output = Path(request.cwd, "gpu-environment.json")
    code = "import os,sys; open(sys.argv[1],'w').write(os.environ['CUDA_VISIBLE_DEVICES'])"
    request = replace(
        request, argv=(sys.executable, "-c", code, str(output)), resources=TaskResourceRequest(gpu_count=1)
    )
    script = Path(request.cwd, "gpu-batch.sh")
    script.write_text(_batch_script(request, ""))
    environment = {"SLURM_JOB_ID": "42"}
    if visibility is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visibility
    completed = subprocess.run(["/bin/bash", str(script)], env=environment, capture_output=True, timeout=20)
    assert completed.returncode == returncode
    if returncode == 0:
        assert output.read_text() == visibility
    else:
        assert not output.exists()


def test_accounting_checks_identity_and_every_step():
    rows = (
        f"42|name|{os.getuid()}|marker|COMPLETED|0:0|0:0\n42.batch|batch|||COMPLETED|0:0|\n42.0|step|||RUNNING|0:0|\n"
    )
    adapter, runner = _cli(rows)
    records = adapter.task_accounting("42", "name", "marker", 1).records
    assert len(records) == 3 and records[2].state == "RUNNING"
    argv = runner.call_args.args[0]
    assert "-X" not in argv and "--duplicates" in argv and "--local" in argv


@pytest.mark.parametrize(
    "row",
    [
        "42|wrong|{uid}|marker|COMPLETED|0:0|0:0",
        "42|name|{uid}|wrong|COMPLETED|0:0|0:0",
        "42|name|9999999|marker|COMPLETED|0:0|0:0",
        "42|name|{uid}|marker|COMPLETED|bad|0:0",
        "42|name|{uid}|marker|COMPLETED|-1:0|0:0",
        "42.batch|batch|||COMPLETED|0:0|",
        "43|name|{uid}|marker|COMPLETED|0:0|0:0",
        "42|name|{uid}|marker||0:0|0:0",
        "42|name|{uid}|marker|COMPLETED|0:0|",
        "42+1|name|{uid}|marker|COMPLETED|0:0|0:0",
        "incomplete",
    ],
)
def test_invalid_accounting_cannot_authorize_settlement(row):
    adapter, _ = _cli(row.format(uid=os.getuid()) + "\n")
    with pytest.raises(SlurmProtocolError):
        adapter.task_accounting("42", "name", "marker", 1)


def test_duplicate_accounting_rejects_requeue_or_job_id_recycling():
    row = f"42|name|{os.getuid()}|marker|COMPLETED|0:0|0:0\n"
    adapter, _ = _cli(row * 2)
    with pytest.raises(SlurmProtocolError, match="duplicate"):
        adapter.task_accounting("42", "name", "marker", 1)


def _supervisor_case(backend, *, write_result=True):
    launcher, adapter, request = backend
    identity = TaskAttemptIdentity(*request.identity[:3], "train", request.attempt_id)
    store = FileTaskArtifactStore(str(Path(request.cwd, "artifacts")))
    bootstrap = WorkerBootstrap(
        identity,
        store.root_dir,
        launcher.manager.config.workspace_path,
        {"path": "nvflare.app_common.np.np_trainer.NPTrainer", "args": {"delta": 2}},
    )
    run_dir = Path(request.cwd)
    (run_dir / "app_site-1" / "config").mkdir(parents=True)
    (run_dir / "meta.json").write_text("{}")
    script = "from nvflare.private.fed.task_worker.worker import run_worker; import sys; run_worker(sys.argv[1])"

    def submit(argv, timeout):
        if write_result:
            environment = {"SLURM_JOB_ID": "42"}
            if request.resources.gpu_count:
                # Simulate allocation visibility only. NPTrainer is a simple
                # CPU application; real GPU compute needs deployed qualification.
                environment["CUDA_VISIBLE_DEVICES"] = "0"
            completed = subprocess.run(["/bin/bash", argv[-1]], timeout=30, capture_output=True, env=environment)
            assert completed.returncode == 0, completed.stderr.decode()
        return SubmissionResult(CommandResult(0, "42", ""), "42")

    adapter.submit.side_effect = submit

    def factory(attempt, path):
        return replace(
            request,
            argv=(sys.executable, "-c", script, path),
            environment={"PYTHONPATH": os.getcwd(), "PATH": os.environ["PATH"]},
        )

    return (
        identity,
        store,
        dict(
            bootstrap=bootstrap,
            store=store,
            input_payload=DXO(DataKind.WEIGHTS, {"numpy_key": np.ones(3)}).to_shareable(),
            runtime_root=str(run_dir / "runtime"),
            launcher=launcher,
            abort_signal=Signal(),
            request_factory=factory,
            options=TaskSupervisorOptions(poll_interval=0.001, settlement_timeout=0.1),
        ),
    )


@pytest.mark.usefixtures("isolated_fobs")
def test_real_worker_result_survives_mock_scheduler_settlement_and_receipt(backend):
    launcher, adapter, _ = backend
    identity, store, arguments = _supervisor_case(backend)
    supervisor = TaskSupervisor()
    outcome = supervisor.run(**arguments)
    assert outcome.execution_status.succeeded
    np.testing.assert_array_equal(from_shareable(outcome.result).data["numpy_key"], np.full(3, 3.0))
    assert Path(store.attempt_dir(identity), "result.fobs").exists()
    assert not supervisor.owns_execution
    # Lost receipt resends the committed result and does not resubmit sbatch.
    supervisor.record_receipt(identity, TaskResultReceipt.RETRY)
    supervisor.read_saved_result(identity)
    assert adapter.submit.call_count == 1
    supervisor.record_receipt(identity, TaskResultReceipt.RECEIVED)
    assert not Path(store.attempt_dir(identity), "result.fobs").exists()
    assert Path(store.attempt_dir(identity), "completion.json").exists()
    assert all(Path(handle.job_dir, "scheduler.jsonl").exists() for handle in launcher.manager._handles.values())


@pytest.mark.usefixtures("isolated_fobs")
def test_incomplete_result_is_not_published_after_allocation_release(backend):
    identity, store, arguments = _supervisor_case(backend, write_result=False)
    supervisor = TaskSupervisor()
    with pytest.raises(IncompleteTaskArtifactError):
        supervisor.run(**arguments)
    assert Path(store.attempt_dir(identity), "input.fobs").exists()
    assert not supervisor.owns_execution


@pytest.mark.usefixtures("isolated_fobs")
def test_end_run_during_active_result_reader_retains_payloads(backend):
    identity, store, arguments = _supervisor_case(backend)
    supervisor = TaskSupervisor()
    reading, proceed = threading.Event(), threading.Event()
    original = store.read_result

    def blocked_read(identity):
        reading.set()
        assert proceed.wait(10)
        return original(identity)

    store.read_result = blocked_read
    with ThreadPoolExecutor() as pool:
        future = pool.submit(supervisor.run, **arguments)
        try:
            assert reading.wait(20)
            supervisor.end_run()
            assert Path(store.attempt_dir(identity), "result.fobs").exists()
        finally:
            proceed.set()
        assert future.result(timeout=10).aborted
    assert Path(store.attempt_dir(identity), "result.fobs").exists()
