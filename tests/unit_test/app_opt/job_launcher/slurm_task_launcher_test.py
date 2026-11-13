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
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.task_launcher_spec import (
    TaskExecutionInterrupted,
    TaskExecutionPhase,
    TaskLauncherError,
    TaskLaunchError,
    TaskResourceRequest,
    TaskSettlementError,
)
from nvflare.app_opt.job_launcher.slurm.config import (
    CommandResult,
    LookupStatus,
    QueryResult,
    SlurmLauncherError,
    SlurmProtocolError,
    SubmissionResult,
)
from nvflare.app_opt.job_launcher.slurm.task_inventory import inventory
from nvflare.app_opt.job_launcher.slurm.task_inventory import main as inventory_main
from nvflare.app_opt.job_launcher.slurm.task_launcher import SlurmTaskLauncher, SlurmTaskManager
from nvflare.app_opt.job_launcher.slurm.task_scheduler import SlurmTaskCliAdapter, TaskSlurmRecord
from nvflare.private.fed.task_worker.artifacts import IncompleteTaskArtifactError
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor
from tests.slurm_test_utils import _accounting, _backend, _query, _supervisor_case


@pytest.fixture(params=(0, 1), ids=("cpu", "gpu"))
def backend(tmp_path, monkeypatch, request):
    return _backend(tmp_path, monkeypatch, request.param)


def test_task_allocation_always_requests_cpu_and_memory(backend):
    launcher, adapter, request = backend
    launcher.launch_task(request)
    argv = adapter.submit.call_args.args[0]
    assert "--cpus-per-task=1" in argv
    assert "--mem=1024M" in argv


@pytest.mark.parametrize("defaults", [{"cpu_cores": None}, {"memory_mb": None}])
def test_site_task_defaults_require_cpu_and_memory(tmp_path, defaults):
    with pytest.raises(ValueError, match="defaults require CPU and memory"):
        SlurmTaskLauncher(workspace_path=str(tmp_path), shared_filesystem=True, cgroup_enforced=True, **defaults)


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


@pytest.mark.parametrize("accounting", [LookupStatus.NOT_FOUND, LookupStatus.UNAVAILABLE, "invalid"])
def test_missing_accounting_after_observed_allocation_enters_bounded_settlement(backend, accounting):
    launcher, adapter, request = backend
    adapter.active_by_id.return_value = _query("RUNNING")
    handle = launcher.launch_task(request)
    assert handle.poll().phase == TaskExecutionPhase.RUNNING
    adapter.active_by_id.return_value = _query()
    if accounting == "invalid":
        adapter.task_accounting.side_effect = SlurmProtocolError("ownership mismatch")
    else:
        adapter.task_accounting.return_value = QueryResult(accounting)
    status = handle.poll()
    assert status.phase == TaskExecutionPhase.TERMINAL and not status.settled
    assert handle.nvflare_job_id in launcher.manager._handles
    if accounting == "invalid":
        assert "ownership mismatch" in status.failure_reason
    else:
        assert "accounting is missing or unavailable" in status.failure_reason


def test_unseen_allocation_missing_from_queue_stays_queued(backend):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = QueryResult(LookupStatus.NOT_FOUND)
    status = launcher.launch_task(request).poll()
    assert status.phase == TaskExecutionPhase.QUEUED and not status.settled


@pytest.mark.parametrize("state", ["NODE_FAIL", "CANCELLED"])
@pytest.mark.parametrize(
    "initial,nodes", [("RUNNING", None), (None, None), (None, 1), ("PENDING", None), ("RUNNING", 0)]
)
def test_started_failed_allocation_waits_for_terminal_batch_evidence(backend, state, initial, nodes):
    launcher, adapter, request = backend
    adapter.active_by_id.return_value = _query(initial)
    adapter.task_accounting.return_value = QueryResult(LookupStatus.NOT_FOUND)
    handle = launcher.launch_task(request)
    assert not handle.poll().settled
    adapter.active_by_id.return_value = _query()
    adapter.task_accounting.return_value = _accounting(state=state, step=None, allocated_nodes=nodes)
    assert not handle.poll().settled
    assert handle.nvflare_job_id in launcher.manager._handles
    adapter.task_accounting.return_value = _accounting(state=state, step="RUNNING")
    assert not handle.poll().settled
    adapter.task_accounting.return_value = _accounting(state=state, step="COMPLETED")
    status = handle.poll()
    assert status.settled and not status.succeeded


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


@pytest.mark.parametrize("state", ["PREEMPTED", "REVOKED", "LAUNCH_FAILED", "RECONFIG_FAIL"])
@pytest.mark.parametrize("step", [None, "COMPLETED"])
def test_infrastructure_terminal_states_settle_as_failures(backend, state, step):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = _accounting(state=state, step=step, allocated_nodes=0 if step is None else 1)
    status = launcher.launch_task(request).poll()
    assert status.settled and not status.succeeded
    assert state in status.failure_reason


def test_settled_handle_releases_memory_but_retains_control_evidence(backend):
    launcher, adapter, request = backend
    adapter.active_by_id.return_value = _query("RUNNING")
    handle = launcher.launch_task(request)
    assert handle.nvflare_job_id in launcher.manager._handles
    assert not handle.poll().settled
    adapter.active_by_id.return_value = _query()
    assert handle.poll().succeeded
    assert handle.nvflare_job_id not in launcher.manager._handles
    assert Path(handle.job_dir, "identity.json").is_file()
    assert Path(handle.job_dir, "scheduler.jsonl").is_file()
    with pytest.raises(SlurmLauncherError, match="stale Slurm artifacts"):
        launcher.launch_task(request)


def test_settlement_wait_refreshes_lagging_accounting_inside_default_budget(backend, monkeypatch):
    launcher, adapter, request = backend
    now = [0.0]
    launcher.manager._monotonic = lambda: now[0]
    launcher.manager.config = replace(launcher.manager.config, poll_interval=10.0)

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(
        "nvflare.app_opt.job_launcher.slurm.task_launcher.time", SimpleNamespace(time=lambda: 0.0, sleep=sleep)
    )
    adapter.task_accounting.side_effect = lambda *args, **kwargs: _accounting(
        step="RUNNING" if now[0] < 2 else "COMPLETED"
    )
    handle = launcher.launch_task(request)
    assert not handle.poll().settled
    assert handle.wait_for_settlement(10.0).succeeded
    assert now[0] == 3.0, "settlement must refresh accounting with backoff before the ten-second deadline"


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
        adapter.task_accounting.return_value = _accounting(state="CANCELLED")
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
    assert handle.request == replace(
        request, resources=TaskResourceRequest(cpu_cores=1, memory_mb=1024, gpu_count=request.resources.gpu_count)
    )
    assert Path(handle.job_dir, "batch.sh").exists()
    with pytest.raises(SlurmLauncherError, match="stale Slurm artifacts"):
        launcher.launch_task(request)
    if not submission.job_id or submission.cluster:
        assert not handle.poll().settled
        with pytest.raises(TaskSettlementError):
            handle.cancel()
        adapter.cancel.assert_not_called()


def test_task_inventory_survives_unconfirmed_and_confirmed_cleanup(backend, capsys):
    launcher, adapter, request = backend
    adapter.active_by_id.return_value = _query("RUNNING")
    handle = launcher.launch_task(request)
    assert not handle.poll().settled
    rows = inventory(launcher.manager.config.workspace_path)
    assert len(rows) == 1
    assert rows[0]["state"] == "unconfirmed"
    assert (rows[0]["job_id"], rows[0]["task_id"], rows[0]["attempt_id"]) == (
        request.job_id,
        request.task_id,
        request.attempt_id,
    )
    assert rows[0]["execution_id"] == handle.execution_id
    assert rows[0]["scheduler_job_name"] == handle.job_name
    assert rows[0]["scheduler_comment"] == launcher.manager._marker(handle.nvflare_job_id)
    inventory_main(["--workspace", launcher.manager.config.workspace_path])
    assert json.loads(capsys.readouterr().out)["state"] == "unconfirmed"
    adapter.active_by_id.return_value = _query()
    adapter.task_accounting.return_value = _accounting()
    assert handle.poll_fresh().settled
    assert inventory(launcher.manager.config.workspace_path)[0]["state"] == "confirmed_settled"
    with Path(handle.job_dir, "scheduler.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"event": "status", "execution_id": handle.execution_id, "phase": "running", "settled": True})
        )
        stream.write("\n")
    rows = inventory(launcher.manager.config.workspace_path)
    assert rows[0]["state"] == "unconfirmed"
    assert "inventory_error" in rows[0]
    identity_path = Path(handle.job_dir, "identity.json")
    identity = json.loads(identity_path.read_text())
    identity["job_id"] = "different-job"
    identity_path.write_text(json.dumps(identity))
    rows = inventory(launcher.manager.config.workspace_path)
    assert rows[0]["state"] == "unconfirmed"
    assert "task identity does not match directory key" in rows[0]["inventory_error"]


def test_task_inventory_keeps_ambiguous_and_corrupt_records_unconfirmed(backend):
    launcher, adapter, request = backend
    adapter.submit.return_value = SubmissionResult(CommandResult(None, "", "", timed_out=True))
    with pytest.raises(TaskLaunchError) as error:
        launcher.launch_task(request)
    handle = error.value.handle
    rows = inventory(launcher.manager.config.workspace_path)
    assert rows[0]["state"] == "unconfirmed"
    assert rows[0]["execution_id"] == handle.execution_id
    assert rows[0]["scheduler_comment"] == launcher.manager._marker(handle.nvflare_job_id)
    Path(handle.job_dir, "scheduler.jsonl").write_text('{"event": "status", "settled": true}\n')
    rows = inventory(launcher.manager.config.workspace_path)
    assert rows[0]["state"] == "unconfirmed"
    assert "inventory_error" in rows[0]


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
def test_reject_request_environment_authority(tmp_path, monkeypatch, name):
    launcher, adapter, request = _backend(tmp_path, monkeypatch, 0, cp_backend=True)
    with pytest.raises((TaskLauncherError, SlurmLauncherError)):
        launcher.launch_task(replace(request, environment={name: "value"}))
    adapter.submit.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        (sys.executable, "-V"),
        ("python", "-V"),
        (sys.executable, "-m", "nvflare.private.fed.app.client.task_worker_process"),
        (
            sys.executable,
            "-m",
            "nvflare.private.fed.app.client.task_worker_process",
            "--bootstrap",
            "/workspace/bootstrap.json",
            "--execution_scope",
            "native",
        ),
    ],
)
def test_cp_backend_rejects_arbitrary_or_prematerialized_commands(tmp_path, monkeypatch, argv):
    launcher, adapter, request = _backend(tmp_path, monkeypatch, 0, cp_backend=True)
    with pytest.raises(TaskLauncherError, match="common staged worker request"):
        launcher.launch_task(replace(request, argv=argv))
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


def _cli(response):
    runner = Mock(return_value=CommandResult(0, response, ""))
    adapter = SlurmTaskCliAdapter({"sacct": "/usr/bin/sacct"}, os.getuid(), logging.getLogger("test"), runner=runner)
    return adapter, runner


def test_accounting_checks_identity_and_every_step():
    rows = (
        f"42|name|{os.getuid()}|marker|COMPLETED|0:0|0:0|1\n"
        "42.batch|batch|||COMPLETED|0:0||1\n42.0|step|||RUNNING|0:0||\n"
    )
    adapter, runner = _cli(rows)
    records = adapter.task_accounting("42", "name", "marker", 1).records
    assert len(records) == 3 and records[2].state == "RUNNING"
    assert records[0].allocated_nodes == 1 and records[2].allocated_nodes is None
    argv = runner.call_args.args[0]
    assert "-X" not in argv and "--duplicates" in argv and "--local" in argv


def test_accounting_accepts_unstored_comment_but_checks_uid_and_attempt_name():
    adapter, _ = _cli(f"42|name|{os.getuid()}||COMPLETED|0:0|0:0|1\n")
    assert adapter.task_accounting("42", "name", "marker", 1).records[0].job_id == "42"


@pytest.mark.parametrize("name,uid", [("wrong", os.getuid()), ("name", 9999999)])
def test_unstored_comment_does_not_weaken_accounting_identity(name, uid):
    adapter, _ = _cli(f"42|{name}|{uid}||COMPLETED|0:0|0:0|1\n")
    with pytest.raises(SlurmProtocolError, match="ownership mismatch"):
        adapter.task_accounting("42", "name", "marker", 1)


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
    adapter, _ = _cli(row.format(uid=os.getuid()) + "|1\n")
    with pytest.raises(SlurmProtocolError):
        adapter.task_accounting("42", "name", "marker", 1)


def test_duplicate_accounting_rejects_requeue_or_job_id_recycling():
    row = f"42|name|{os.getuid()}|marker|COMPLETED|0:0|0:0|1\n"
    adapter, _ = _cli(row * 2)
    with pytest.raises(SlurmProtocolError, match="duplicate"):
        adapter.task_accounting("42", "name", "marker", 1)


@pytest.mark.parametrize("nodes", ["", "-1", "unknown"])
def test_missing_or_invalid_allocated_node_count_cannot_prove_no_execution(nodes):
    adapter, _ = _cli(f"42|name|{os.getuid()}|marker|CANCELLED|0:0|0:0|{nodes}\n")
    with pytest.raises(SlurmProtocolError):
        adapter.task_accounting("42", "name", "marker", 1)


def test_accounting_explicitly_identifies_unallocated_cancellation():
    adapter, _ = _cli(f"42|name|{os.getuid()}|marker|CANCELLED|0:0|0:0|0\n")
    assert adapter.task_accounting("42", "name", "marker", 1).records[0].allocated_nodes == 0


@pytest.mark.usefixtures("isolated_fobs")
def test_incomplete_result_is_not_published_after_allocation_release(backend):
    identity, store, arguments = _supervisor_case(backend, write_result=False)
    supervisor = TaskSupervisor()
    with pytest.raises(IncompleteTaskArtifactError):
        supervisor.run(**arguments)
    assert Path(store.attempt_dir(identity), "input.fobs").exists()
    assert not supervisor.owns_execution


def test_long_cancel_backs_off_queries_and_does_not_repeat_acknowledged_scancel(backend, monkeypatch):
    launcher, adapter, request = backend
    now = [0.0]
    launcher.manager._monotonic = lambda: now[0]
    launcher.manager.config = replace(launcher.manager.config, poll_interval=10.0, cancel_timeout=120.0)

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(
        "nvflare.app_opt.job_launcher.slurm.task_launcher.time", SimpleNamespace(time=lambda: 0.0, sleep=sleep)
    )
    adapter.active_by_id.return_value = _query("RUNNING")
    adapter.cancel.return_value = QueryResult(LookupStatus.FOUND)
    handle = launcher.launch_task(request)
    with pytest.raises(TaskSettlementError):
        handle.cancel()
    assert now[0] == 120.0
    assert adapter.active_by_id.call_count <= 18
    adapter.cancel.assert_called_once()
    assert not handle.poll().settled


@pytest.mark.parametrize(
    "rows,expected",
    [
        ("42|RUNNING|{uid}|marker|name", LookupStatus.FOUND),
        ("42|RUNNING|{uid}|other-marker|name", LookupStatus.NOT_FOUND),
        ("42|RUNNING|999999|marker|name", LookupStatus.NOT_FOUND),
        ("42|RUNNING|{uid}|marker|other-name", LookupStatus.NOT_FOUND),
        ("", LookupStatus.NOT_FOUND),
    ],
)
def test_lost_submission_marker_lookup_requires_exact_ownership(rows, expected):
    adapter = SlurmTaskCliAdapter({"squeue": "/usr/bin/squeue"}, os.getuid(), Mock())
    adapter._run = Mock(return_value=CommandResult(0, rows.format(uid=os.getuid()), ""))
    result = adapter.task_by_marker("name", "marker", 5)
    assert result.status == expected
    argv = adapter._run.call_args.args[0]
    assert "--local" in argv and "--name=name" in argv and f"--user={os.getuid()}" in argv


@pytest.mark.parametrize(
    "rows",
    [
        "42|RUNNING|{uid}|marker|name\n43|RUNNING|{uid}|marker|name",
        "42.batch|RUNNING|{uid}|marker|name",
        "42| |{uid}|marker|name",
        "malformed",
    ],
)
def test_ambiguous_or_malformed_marker_lookup_cannot_select_an_allocation(rows):
    adapter = SlurmTaskCliAdapter({"squeue": "/usr/bin/squeue"}, os.getuid(), Mock())
    adapter._run = Mock(return_value=CommandResult(0, rows.format(uid=os.getuid()), ""))
    with pytest.raises(SlurmProtocolError):
        adapter.task_by_marker("name", "marker", 5)


@pytest.mark.parametrize(
    "reason",
    [
        "Invalid partition name specified",
        "Invalid account or account/partition combination specified",
        "Invalid qos specification",
        "Requested node configuration is not available",
        "Invalid generic resource (gres) specification",
        "Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)",
        "Invalid Trackable RESource (TRES) specification",
        "Requested time limit is invalid (missing or exceeds some limit)",
        "Invalid time specified",
        "Invalid time-min specification (exceeds job's time or other limits)",
        "More processors requested than permitted",
        "Memory required by task is not available",
        "Invalid feature specification",
        "Invalid preferred feature specification",
    ],
)
@pytest.mark.parametrize("plugin_message", ["", "sbatch: error: job_submit.lua: checking site policy\n"])
def test_definitive_sbatch_rejection_settles_with_scheduler_diagnostic(backend, reason, plugin_message):
    launcher, adapter, request = backend
    adapter.submit.return_value = SubmissionResult(
        CommandResult(1, "", f"{plugin_message}sbatch: error: Batch job submission failed: {reason}\n")
    )
    with pytest.raises(TaskLaunchError) as failure:
        launcher.launch_task(request)
    handle = failure.value.handle
    status = handle.poll()
    assert status.settled
    assert not status.succeeded
    assert reason in status.failure_reason and reason in str(failure.value)
    assert plugin_message.strip() in status.failure_reason
    assert not launcher.manager._handles
    assert Path(handle.job_dir, "identity.json").exists()
    adapter.task_by_marker.assert_not_called()
    adapter.cancel.assert_not_called()


@pytest.mark.parametrize(
    "diagnostic",
    [
        "sbatch: error: Unable to open file /shared/workspace/batch script.sh",
        "sbatch: error: invalid partition specified: unavailable",
    ],
)
def test_local_presubmission_rejection_settles(backend, diagnostic):
    launcher, adapter, request = backend
    adapter.submit.return_value = SubmissionResult(CommandResult(1, "", diagnostic))
    with pytest.raises(TaskLaunchError) as failure:
        launcher.launch_task(request)
    assert failure.value.handle.poll().settled
    assert all(line in str(failure.value) for line in diagnostic.splitlines())
    adapter.task_by_marker.assert_not_called()
    adapter.cancel.assert_not_called()


def test_known_rejections_allow_site_explanations_before_between_and_after(backend):
    launcher, adapter, request = backend
    diagnostic = (
        "sbatch: error: job_submit.lua: checking CPU socket geometry\n"
        "sbatch: error: Batch job submission failed: Invalid partition name specified\n"
        "sbatch: site policy uses the requested connection profile\n"
        "sbatch: error: Batch job submission failed: Invalid feature specification\n"
        "sbatch: error: see site documentation for supported features"
    )
    adapter.submit.return_value = SubmissionResult(CommandResult(1, "", diagnostic))
    with pytest.raises(TaskLaunchError) as failure:
        launcher.launch_task(request)
    assert failure.value.handle.poll().settled
    assert all(line in str(failure.value) for line in diagnostic.splitlines())
    adapter.task_by_marker.assert_not_called()


@pytest.mark.parametrize("job_id,cluster", [("42", None), ("42", "other"), (None, "other")])
def test_known_rejection_cannot_discard_returned_allocation_identity(backend, job_id, cluster):
    launcher, adapter, request = backend
    adapter.submit.return_value = SubmissionResult(
        CommandResult(1, "", "sbatch: error: Batch job submission failed: Invalid partition name specified"),
        job_id,
        cluster,
    )
    adapter.for_cluster.return_value = adapter
    adapter.cancel_suffix.return_value = QueryResult(LookupStatus.FOUND)
    adapter.active_by_id.return_value = _query("RUNNING")
    adapter.task_by_marker.return_value = _query()
    with pytest.raises(TaskLaunchError) as failure:
        launcher.launch_task(request)
    assert launcher.manager._handles
    assert not failure.value.handle.poll().settled


@pytest.mark.usefixtures("isolated_fobs")
def test_plugin_message_before_rejection_does_not_fence_supervisor(backend):
    launcher, adapter, request = backend
    identity, store, arguments = _supervisor_case(backend, write_result=False)
    diagnostic = (
        "sbatch: error: job_submit.lua: using site defaults\n"
        "sbatch: error: Batch job submission failed: Memory required by task is not available"
    )
    adapter.submit.side_effect = None
    adapter.submit.return_value = SubmissionResult(CommandResult(1, "", diagnostic))
    adapter.task_by_marker.return_value = _query()
    supervisor = TaskSupervisor()
    with pytest.raises(TaskLaunchError, match="Memory required by task"):
        supervisor.run(**arguments)
    assert not supervisor.owns_execution
    assert not supervisor.settlement_unconfirmed
    assert not launcher.manager._handles
    assert Path(store.attempt_dir(identity), "input.fobs").exists()
    adapter.task_by_marker.assert_not_called()
    adapter.cancel.assert_not_called()


def test_cluster_suffix_cancels_and_waits_for_target_cluster_settlement(backend):
    launcher, adapter, request = backend
    adapter.submit.return_value = SubmissionResult(CommandResult(0, "42;other", ""), "42", "other")
    adapter.for_cluster.return_value = adapter
    adapter.cancel_suffix.return_value = QueryResult(LookupStatus.FOUND)
    adapter.active_by_id.return_value = _query("RUNNING")
    with pytest.raises(TaskLaunchError) as failure:
        launcher.launch_task(request)
    adapter.cancel_suffix.assert_called_once_with("42", "other", timeout=launcher.manager.config.cancel_timeout)
    assert not failure.value.handle.poll().settled
    adapter.active_by_id.return_value = _query()
    adapter.task_accounting.return_value = _accounting("CANCELLED")
    assert failure.value.handle.poll().settled
    adapter.for_cluster.assert_called_with("other")


@pytest.mark.parametrize(
    "command",
    [
        CommandResult(1, "", "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"),
        CommandResult(
            None, "", "sbatch: error: Batch job submission failed: Invalid partition name specified", timed_out=True
        ),
        CommandResult(
            1, "", "sbatch: error: Batch job submission failed: Invalid partition name specified\nSocket timed out"
        ),
        CommandResult(1, "", "unknown plugin error"),
        CommandResult(
            1, "unparseable output", "sbatch: error: Batch job submission failed: Invalid partition name specified"
        ),
        CommandResult(1, "", "sbatch: error: job_submit.lua: custom policy rejection"),
        CommandResult(1, "", "sbatch: error: Unable to open file"),
        CommandResult(
            1,
            "",
            "sbatch: error: Batch job submission failed: Invalid partition name specified\n"
            "sbatch: error: Batch job submission failed: Unknown controller error",
        ),
        CommandResult(
            1,
            "",
            "sbatch: error: Batch job submission failed: Unknown controller error\n"
            "sbatch: error: Batch job submission failed: Invalid partition name specified",
        ),
        CommandResult(
            1,
            "",
            "sbatch: error: Batch job submission failed: Invalid partition name specified\n"
            "sbatch: error: Socket timed out on send/recv operation",
        ),
        CommandResult(
            1,
            "",
            "sbatch: error: Socket timed out on send/recv operation\n"
            "sbatch: error: Batch job submission failed: Invalid partition name specified",
        ),
        CommandResult(
            1,
            "",
            "sbatch: error: Batch job submission failed: Invalid partition name specified\nSubmitted batch job 42",
        ),
        CommandResult(1, "", "sbatch: error: Batch job submission failed: Invalid partition name specified\n42;other"),
        CommandResult(
            1,
            "",
            "sbatch: error: Batch job submission failed: Invalid partition name specified\n"
            "sbatch: error: Batch job submission failed",
        ),
        CommandResult(
            1,
            "",
            "sbatch: error: Batch job submission failed: Invalid partition name specified\n"
            "sbatch: error: batch job submission failed: Unknown controller error",
        ),
    ],
)
def test_nondefinitive_submission_failure_remains_owned(backend, command):
    launcher, adapter, request = backend
    adapter.submit.return_value = SubmissionResult(command)
    adapter.task_by_marker.return_value = _query()
    with pytest.raises(TaskLaunchError) as failure:
        launcher.launch_task(request)
    assert not failure.value.handle.poll().settled
    assert launcher.manager._handles
    assert str(command.returncode) in str(failure.value) or command.timed_out


def test_foreign_cluster_queries_and_cancellation_never_use_local_identity():
    adapter = SlurmTaskCliAdapter({k: "/usr/bin/" + k for k in ("squeue", "sacct", "scancel")}, os.getuid(), Mock())
    adapter._runner = Mock(return_value=CommandResult(0, "", ""))
    remote = adapter.for_cluster("other")
    remote.active_by_id("42", "name", "marker", 1)
    remote.task_accounting("42", "name", "marker", 1)
    remote.cancel("42", 1)
    assert adapter._runner.call_count == 3
    for call in adapter._runner.call_args_list:
        assert "--clusters=other" in call.args[0]
        assert "--local" not in call.args[0]
    assert not hasattr(adapter, "_cleanup_cluster")


@pytest.mark.parametrize("state", ["TIMEOUT", "PREEMPTED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"])
def test_settled_loss_of_running_allocation_is_retryable(backend, state):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = _accounting(state=state)
    status = launcher.launch_task(request).poll()
    assert status.settled and status.retryable


def test_successful_allocation_is_not_retryable(backend):
    launcher, _, request = backend
    status = launcher.launch_task(request).poll()
    assert status.succeeded and not status.retryable


def test_unallocated_failure_is_not_retryable(backend):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = _accounting(state="FAILED", step=None, allocated_nodes=0)
    status = launcher.launch_task(request).poll()
    assert status.settled and not status.retryable


def test_cancelled_allocation_is_not_retryable(backend):
    launcher, adapter, request = backend
    handle = launcher.launch_task(request)
    handle.request_cancel()
    adapter.task_accounting.return_value = _accounting(state="CANCELLED")
    status = handle.poll()
    assert status.settled and status.cancel_requested and not status.retryable


@pytest.mark.usefixtures("isolated_fobs")
@pytest.mark.parametrize("state", ["TIMEOUT", "PREEMPTED"])
def test_supervisor_interrupts_cj_when_allocation_is_lost_without_result(backend, state):
    _, adapter, _ = backend
    adapter.task_accounting.return_value = _accounting(state=state)
    identity, store, arguments = _supervisor_case(backend, write_result=False)
    supervisor = TaskSupervisor()
    with pytest.raises(TaskExecutionInterrupted, match=state):
        supervisor.run(**arguments)
    assert not supervisor.owns_execution


@pytest.mark.parametrize("state", ["CANCELLED", "REVOKED", "DEADLINE"])
def test_external_cancellation_and_other_states_are_not_retryable(backend, state):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = _accounting(state=state)
    status = launcher.launch_task(request).poll()
    assert status.settled and not status.cancel_requested and not status.retryable


def test_failed_allocation_is_retryable_only_when_killed_by_signal(backend):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = _accounting(state="FAILED", exit_status=1)
    assert not launcher.launch_task(request).poll().retryable


def test_signal_killed_allocation_is_retryable(backend):
    launcher, adapter, request = backend
    adapter.task_accounting.return_value = _accounting(state="FAILED", exit_signal=9)
    assert launcher.launch_task(request).poll().retryable


def test_cp_cancel_followed_by_timeout_is_not_retryable(backend):
    launcher, adapter, request = backend
    handle = launcher.launch_task(request)
    handle.request_cancel()
    adapter.task_accounting.return_value = _accounting(state="TIMEOUT")
    status = handle.poll()
    assert status.settled and status.cancel_requested and not status.retryable


@pytest.mark.usefixtures("isolated_fobs")
def test_application_failure_takes_precedence_over_retryable_loss(backend):
    _, adapter, _ = backend
    adapter.task_accounting.return_value = _accounting(state="TIMEOUT")
    identity, store, arguments = _supervisor_case(backend, write_result=False)
    submit = adapter.submit.side_effect

    def fail_then_submit(argv, timeout):
        store.record_failure(
            identity, worker_pid=1, worker_ppid=1, started_at=0.0, failed_at=1.0, error_type="ValueError", message="bad"
        )
        return submit(argv, timeout)

    adapter.submit.side_effect = fail_then_submit
    with pytest.raises(RuntimeError, match="failed: ValueError"):
        TaskSupervisor().run(**arguments)
