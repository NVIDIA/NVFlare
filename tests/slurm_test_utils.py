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

"""Shared Slurm test boundaries and staged-worker fixtures."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import TaskLaunchRequest, TaskResourceRequest
from nvflare.app_opt.job_launcher.slurm.config import (
    SLURM_CHILD_PROCESS_ENV,
    CommandResult,
    LookupStatus,
    QueryResult,
    SlurmRecord,
    SubmissionResult,
)
from nvflare.app_opt.job_launcher.slurm.task_launcher import _SlurmTaskBackend
from nvflare.app_opt.job_launcher.slurm.task_scheduler import TaskSlurmRecord
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity, WorkerBootstrap
from nvflare.private.fed.task_worker.supervisor import TaskSupervisorOptions


def _query(state=None):
    return (
        QueryResult(LookupStatus.FOUND, (SlurmRecord("42", state),)) if state else QueryResult(LookupStatus.NOT_FOUND)
    )


def _accounting(state="COMPLETED", step="COMPLETED", **kwargs):
    records = [TaskSlurmRecord("42", state, **kwargs)]
    if step:
        records.append(TaskSlurmRecord("42.batch", step))
    return QueryResult(LookupStatus.FOUND, tuple(records))


def _backend(tmp_path, monkeypatch, gpu_count, *, job_id="job-1", **launcher_args):
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    (root / "startup").mkdir()
    (root / "local").mkdir()
    (root / job_id).mkdir()
    launcher = _SlurmTaskBackend(
        workspace_path=str(root),
        shared_filesystem=True,
        cgroup_enforced=True,
        sbatch_directives={"account": "site-account", "partition": "task-compute", "time": "00:05:00"},
        cancel_timeout=0.1,
        poll_interval=0.001,
        **launcher_args,
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
        job_id,
        "site-1",
        "task-1",
        "attempt-1",
        (sys.executable, "-V"),
        cwd=str(root / job_id),
        resources=TaskResourceRequest(gpu_count=gpu_count),
    )
    return launcher, adapter, launch_request


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
