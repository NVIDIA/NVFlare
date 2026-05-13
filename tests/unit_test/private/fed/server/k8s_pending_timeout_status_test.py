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
"""End-to-end verification: when a K8s job pod is stuck Pending past the
launcher's pending_timeout (e.g. cluster resources unavailable), list_jobs
must show FINISHED:EXECUTION_EXCEPTION — both for the server's SJ pod and
for a client's CJ pod.

The pieces are exercised individually in their own test modules; this file
chains them so a single regression test covers the full pipeline:

  Server scenario:
    K8sJobHandle (Pending → terminate)
      → poll() == JobReturnCode.EXCEPTION
      → ServerEngine.wait_for_complete records exception_run_processes
      → JobRunner._update_job_status → set_status(FINISHED_EXECUTION_EXCEPTION)

  Client scenario:
    K8sJobHandle (Pending → terminate)
      → poll() == JobReturnCode.EXCEPTION
      → ClientExecutor.REPORTABLE_JOB_FAILURES has reason for code
      → FederatedServer.process_job_failure → JobRunner.fail_run
      → JobRunner._update_job_status → set_status(FINISHED_EXECUTION_EXCEPTION)
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import RunProcessKey
from nvflare.apis.job_def import RunStatus
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.app_opt.job_launcher.k8s_launcher import K8sJobHandle, PodPhase
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.private.fed.client.client_executor import REPORTABLE_JOB_FAILURES
from nvflare.private.fed.server.job_runner import JobRunner
from nvflare.private.fed.server.server_engine import ServerEngine


def _stuck_pending_handle(pending_timeout=2):
    """Build a real K8sJobHandle whose pod stays in PENDING forever.

    Using pending_timeout=2 makes _stuck_in_pending() trip after two polls,
    so the test doesn't have to wait the default 120s. We patch time.sleep
    at the call site to keep the loop tight.
    """
    api = MagicMock()
    resp = MagicMock()
    resp.status.phase = PodPhase.PENDING.value
    api.read_namespaced_pod.return_value = resp

    job_config = {
        "name": "stuck-pod",
        "image": "test:latest",
        "container_name": "ctr",
        "command": "nvflare.private.fed.app.client.worker_process",
        "volume_mount_list": [],
        "volume_list": [],
        "module_args": {},
    }
    return K8sJobHandle(
        job_id="job-1",
        api_instance=api,
        job_config=job_config,
        pending_timeout=pending_timeout,
    )


@patch("nvflare.app_opt.job_launcher.k8s_launcher.time.sleep")
def test_k8s_launcher_returns_exception_when_pod_stuck_pending(_mock_sleep):
    """Sanity: the K8s launcher's contract is that stuck-pending → EXCEPTION."""
    handle = _stuck_pending_handle(pending_timeout=2)

    from nvflare.app_opt.job_launcher.k8s_launcher import JobState

    assert handle.enter_states([JobState.RUNNING]) is False
    assert handle.poll() == JobReturnCode.EXCEPTION


@patch("nvflare.app_opt.job_launcher.k8s_launcher.time.sleep")
def test_server_side_pending_timeout_reports_finished_execution_exception(_mock_sleep):
    """Server's SJ pod stuck pending → list_jobs shows FINISHED:EXECUTION_EXCEPTION.

    Wires the real K8sJobHandle through ServerEngine.wait_for_complete and
    JobRunner._update_job_status, then asserts the JobManager status set call.
    """
    # 1. K8s pod stuck pending → poll() = EXCEPTION (101).
    handle = _stuck_pending_handle(pending_timeout=2)
    from nvflare.app_opt.job_launcher.k8s_launcher import JobState

    handle.enter_states([JobState.RUNNING])
    assert handle.poll() == JobReturnCode.EXCEPTION

    # 2. ServerEngine.wait_for_complete reads the launcher's return code and
    #    records the run as exception_run_processes.
    engine = ServerEngine.__new__(ServerEngine)
    engine.lock = threading.Lock()
    run_process_info = {
        RunProcessKey.PARTICIPANTS: {},
        RunProcessKey.JOB_HANDLE: handle,
    }
    engine.run_processes = {"job-1": run_process_info}
    engine.exception_run_processes = {}
    engine.engine_info = MagicMock()
    engine.logger = MagicMock()
    engine.client_manager = MagicMock()
    engine.client_manager.clients = {}

    # get_return_code prefers the rc_file then falls back to handle.poll();
    # patch it so we exercise the actual launcher contract via poll().
    with patch(
        "nvflare.private.fed.server.server_engine.get_return_code",
        return_value=handle.poll(),
    ):
        engine.wait_for_complete(workspace="/tmp", job_id="job-1", process=handle)

    assert "job-1" in engine.exception_run_processes
    assert engine.exception_run_processes["job-1"][RunProcessKey.PROCESS_RETURN_CODE] == JobReturnCode.EXCEPTION
    assert "job-1" not in engine.run_processes  # popped after wait

    # 3. JobRunner._update_job_status maps EXCEPTION → FINISHED_EXECUTION_EXCEPTION
    #    and persists it via JobManager.set_status — this is what list_jobs
    #    surfaces.
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.abort_client_run = MagicMock()

    job = MagicMock()
    job.job_id = "job-1"
    job_manager = MagicMock()
    fl_ctx = MagicMock()

    status = runner._update_job_status(engine, job, job_manager, fl_ctx)

    assert status == RunStatus.FINISHED_EXECUTION_EXCEPTION
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_EXECUTION_EXCEPTION, fl_ctx)


@patch("nvflare.app_opt.job_launcher.k8s_launcher.time.sleep")
def test_client_side_pending_timeout_reports_finished_execution_exception(_mock_sleep):
    """Client's CJ pod stuck pending → list_jobs shows FINISHED:EXECUTION_EXCEPTION.

    Wires the real K8sJobHandle through the client's REPORTABLE_JOB_FAILURES
    contract, the server's process_job_failure dispatch, JobRunner.fail_run,
    and finally JobRunner._update_job_status. Asserts the JobManager status
    set call.
    """
    # 1. K8s CJ pod stuck pending → poll() = EXCEPTION (101).
    handle = _stuck_pending_handle(pending_timeout=2)
    from nvflare.app_opt.job_launcher.k8s_launcher import JobState

    handle.enter_states([JobState.RUNNING])
    launcher_return_code = handle.poll()
    assert launcher_return_code == JobReturnCode.EXCEPTION

    # 2. Client recognizes this as a reportable failure and would send
    #    REPORT_JOB_FAILURE with that exact code.
    assert launcher_return_code in REPORTABLE_JOB_FAILURES
    failure_reason = REPORTABLE_JOB_FAILURES[launcher_return_code]
    assert failure_reason  # non-empty

    # 3. Server's process_job_failure dispatcher routes EXCEPTION-class codes
    #    to JobRunner.fail_run with ProcessExitCode.EXCEPTION. We assert the
    #    routing rule directly here (covered fully in fed_server_test.py).
    assert launcher_return_code in (
        ProcessExitCode.CONFIG_ERROR,
        ProcessExitCode.EXCEPTION,
    )

    # 4. JobRunner.fail_run records exception_run_processes and calls _stop_run
    #    (which would tell the SJ to abort). We mock _stop_run to focus on the
    #    state recording.
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.log_error = MagicMock()
    runner.abort_client_run = MagicMock()
    runner._stop_run = MagicMock()

    engine = MagicMock()
    engine.run_processes = {"job-1": {RunProcessKey.PARTICIPANTS: {}}}
    engine.exception_run_processes = {}
    engine.client_manager.clients = {}
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    job = MagicMock()
    job.job_id = "job-1"
    runner.running_jobs = {"job-1": job}

    runner.fail_run("job-1", ProcessExitCode.EXCEPTION, fl_ctx)

    assert engine.exception_run_processes["job-1"][RunProcessKey.PROCESS_RETURN_CODE] == ProcessExitCode.EXCEPTION

    # 5. Simulate the SJ's graceful abort completing (UPDATE_RUN_STATUS with
    #    execution_error=False sets PROCESS_FINISHED=True; PROCESS_EXE_ERROR
    #    stays False). The EXCEPTION rc must still take precedence — otherwise
    #    list_jobs would mistakenly show FINISHED:COMPLETED.
    engine.exception_run_processes["job-1"][RunProcessKey.PROCESS_FINISHED] = True
    engine.exception_run_processes["job-1"][RunProcessKey.PROCESS_EXE_ERROR] = False

    job_manager = MagicMock()
    status = runner._update_job_status(engine, job, job_manager, fl_ctx)

    assert status == RunStatus.FINISHED_EXECUTION_EXCEPTION
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_EXECUTION_EXCEPTION, fl_ctx)


@pytest.mark.parametrize(
    "sj_exit_code",
    [
        JobReturnCode.ABORTED,  # SJ aborted via SIGTERM, exits with 9
        JobReturnCode.EXECUTION_ERROR,  # SJ exits with 1 (some error)
    ],
)
@patch("nvflare.app_opt.job_launcher.k8s_launcher.time.sleep")
def test_sj_exit_after_fail_run_does_not_clobber_exception_status(_mock_sleep, sj_exit_code):
    """After fail_run records EXCEPTION on the client-pending-timeout path,
    the SJ's own non-zero exit code from the abort signal must not overwrite
    the authoritative EXCEPTION rc — otherwise list_jobs would degrade to
    FINISHED:ABORTED.
    """
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.log_error = MagicMock()
    runner.abort_client_run = MagicMock()
    runner._stop_run = MagicMock()

    engine = ServerEngine.__new__(ServerEngine)
    engine.lock = threading.Lock()
    run_process_info = {RunProcessKey.PARTICIPANTS: {}}
    engine.run_processes = {"job-1": run_process_info}
    engine.exception_run_processes = {}
    engine.engine_info = MagicMock()
    engine.logger = MagicMock()
    engine.client_manager = MagicMock()
    engine.client_manager.clients = {}

    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    job = MagicMock()
    job.job_id = "job-1"
    runner.running_jobs = {"job-1": job}

    # Step 1: client reports EXCEPTION via REPORT_JOB_FAILURE → fail_run.
    runner.fail_run("job-1", ProcessExitCode.EXCEPTION, fl_ctx)
    assert engine.exception_run_processes["job-1"][RunProcessKey.PROCESS_RETURN_CODE] == ProcessExitCode.EXCEPTION

    # Step 2: the SJ subsequently exits with sj_exit_code (e.g. 9 from SIGTERM).
    # ServerEngine.wait_for_complete must keep the EXCEPTION code, not overwrite.
    process = MagicMock()
    with patch(
        "nvflare.private.fed.server.server_engine.get_return_code",
        return_value=sj_exit_code,
    ):
        engine.wait_for_complete(workspace="/tmp", job_id="job-1", process=process)

    assert engine.exception_run_processes["job-1"][RunProcessKey.PROCESS_RETURN_CODE] == ProcessExitCode.EXCEPTION

    # Step 3: _update_job_status persists FINISHED_EXECUTION_EXCEPTION.
    job_manager = MagicMock()
    status = runner._update_job_status(engine, job, job_manager, fl_ctx)

    assert status == RunStatus.FINISHED_EXECUTION_EXCEPTION
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_EXECUTION_EXCEPTION, fl_ctx)
