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

import importlib
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import (
    AdminCommandNames,
    FLContextKey,
    FLMetaKey,
    JobConstants,
    ReservedKey,
    RunProcessKey,
    WorkspaceConstants,
)
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobProcessArgs, JobReturnCode, publish_job_handle
from nvflare.apis.workspace import Workspace
from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher
from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher
from nvflare.app_opt.job_launcher.slurm.launcher import ServerSlurmJobLauncher
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.core_cell import FQCN
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.private.defs import CellChannel, CellChannelTopic, JobOutcomeMsgKey, RequestHeader
from nvflare.private.fed.client import client_executor as client_executor_module
from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.client_executor import (
    _ABORT_REQUESTED_KEY,
    _CPU_ATTEMPT_DIR,
    _RESTART_KEY,
    _RUN_STATE_KEY,
    REPORTABLE_JOB_FAILURES,
    JobCleanupPending,
    JobExecutor,
    _build_participant_resource_report,
    _PendingJobHandle,
    _record_client_cpu_attempt,
    _RunPhase,
    _RunState,
)
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.client.communicator import Communicator
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.resource_stats.handoff import terminal_handoff_path, write_terminal_handoff
from nvflare.private.fed.resource_stats.measurements import participant_measurements
from nvflare.private.fed.utils.job_cert_utils import find_job_cert, write_job_cert
from tests.resource_stats_report_test_utils import _participant_summary
from tests.timing_utils import ManualClock, ObservedLock, ThreadGroup, isolate_time, join_thread, wait_for

_start_thread = threading.Thread.start


def _skip_completion_monitor(thread):
    if getattr(thread._target, "__name__", "") == "_wait_child_process_finish":
        executor = thread._target.__self__
        state = executor.run_processes[thread._args[1]][_RUN_STATE_KEY]
        if state.phase == _RunPhase.MONITORING:
            state.monitor = None
            return
    _start_thread(thread)


@pytest.fixture(autouse=True)
def cleanup_monitors(monkeypatch):
    # Each test owns the monitors it creates, including an assertion failure while
    # a query is being retried. Runtime gates remain the individual test's responsibility.
    executors = []
    states = []
    original_start_monitor = JobExecutor._start_monitor
    original_init = JobExecutor.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.job_cleanup_poll_interval = 0.01
        executors.append(self)

    def start_monitor(self, job_id, state):
        if all(existing is not state for existing in states):
            states.append(state)
        return original_start_monitor(self, job_id, state)

    monkeypatch.setattr(JobExecutor, "__init__", init)
    monkeypatch.setattr(JobExecutor, "_start_monitor", start_monitor)
    yield
    from tests.timing_utils import join_thread

    for executor in executors:
        with executor.lock:
            owners = list(executor.run_processes.values())
            executor.run_processes.clear()
        for owner in owners:
            state = owner.get(_RUN_STATE_KEY)
            if state and all(existing is not state for existing in states):
                states.append(state)
            handle = owner.get(RunProcessKey.JOB_HANDLE)
            if isinstance(handle, _PendingJobHandle) and handle.cancel_call and handle.cancel_call.thread.ident:
                join_thread(handle.cancel_call.thread)
    for state in states:
        state.phase = _RunPhase.RELEASED
        state.handoff.set()
        state.wake.set()
        if state.monitor and state.monitor.ident and state.monitor is not threading.current_thread():
            join_thread(state.monitor)


@pytest.fixture
def cleanup_deadlines(monkeypatch):
    from nvflare.private.fed.client import client_executor as subject
    from tests.client_executor_test_utils import ControlledCleanupDeadlines

    return ControlledCleanupDeadlines(monkeypatch, subject)


def _monitor_until_pending(executor, *args, **kwargs):
    pending = threading.Event()
    retain = executor._retain_cleanup_owner

    def retained(job_id, expected_owner=None, **kwargs):
        retain(job_id, expected_owner, **kwargs)
        pending.set()

    with patch.object(executor, "_retain_cleanup_owner", side_effect=retained):
        thread = threading.Thread(target=lambda: executor._wait_child_process_finish(*args, **kwargs))
        thread.start()
        assert pending.wait(5)
    return thread


def _run_recovery(tmp_path, outcomes, on_wait=None, limit=3, confirmed=True, is_leaf=True, on_launch=None):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    ctx.set_prop(ReservedKey.IS_LEAF, is_leaf, private=False, sticky=True)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    executor.max_cj_restart_failures = limit
    launcher = MagicMock()
    resources = MagicMock()
    handles = []
    run_dir = tmp_path / "job-1"
    write_job_cert(str(run_dir), b"cert", b"key")
    checkpoint = run_dir / "checkpoint"
    checkpoint.write_text("preserved")

    def launch(*_):
        index = len(handles)
        assert ctx.get_prop(FLContextKey.JOB_RESTART_HANDLE) is (handles[-1] if handles else None)
        ready, code = outcomes[index]
        launch_id = executor.run_processes["job-1"][_RESTART_KEY].launch_id
        option = f"{RequestHeader.CJ_LAUNCH_ID}={launch_id}"
        assert ctx.get_prop(FLContextKey.ARGS).set == ["print_conf=True", option]
        assert ctx.get_prop(FLContextKey.ALLOCATED_RESOURCES) == {"gpu": 1}
        assert ctx._get_ctx_manager().new_context().get_prop(FLContextKey.ALLOCATED_RESOURCES) is None
        assert ctx._get_ctx_manager().new_context().get_prop(FLContextKey.ARGS).set == ["parent-only=True"]
        assert ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)[JobProcessArgs.OPTIONS][1].split() == [
            "print_conf=True",
            option,
        ]
        handle = MagicMock(spec=JobHandleSpec)
        handle.poll.return_value = code
        handle.can_restart.return_value = confirmed
        handles.append(handle)

        def wait():
            assert find_job_cert(str(run_dir)) is not None
            assert checkpoint.read_text() == "preserved"
            assert executor.get_run_processes_keys() == ["job-1"]
            client.send_request_before_shutdown.assert_not_called()
            resources.free_resources.assert_not_called()
            restart = executor.run_processes["job-1"][_RESTART_KEY]
            if ready:
                executor.notify_job_status("job-1", ClientStatus.STARTED, restart.launch_id)
                executor.notify_job_status("job-1", ClientStatus.READY, restart.launch_id)
            if on_wait:
                on_wait(executor, index, handle)

        handle.wait.side_effect = wait
        if on_launch:
            on_launch(executor, index, handle, ctx)
        return handle

    launcher.launch_job.side_effect = launch
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        executor.start_app(
            client, "job-1", meta, SimpleNamespace(workspace=str(tmp_path), set=[]), {"gpu": 1}, "token", resources, ctx
        )
    executor._wait_child_process_finish(client, "job-1", {"gpu": 1}, "token", resources, str(tmp_path), ctx)
    assert ctx.get_prop(FLContextKey.JOB_RESTART_HANDLE) is None
    assert ctx.get_prop(FLContextKey.ARGS).set == ["print_conf=True"]
    assert executor.get_run_processes_keys() == []
    assert find_job_cert(str(run_dir)) is None
    assert checkpoint.read_text() == "preserved"
    resources.free_resources.assert_called_once()
    client.send_request_before_shutdown.assert_called_once()
    assert sum(call.args[0] == EventType.JOB_COMPLETED for call in ctx.get_engine().fire_event.call_args_list) == 1
    return launcher, client, handles


@pytest.mark.parametrize(
    "outcomes, launches, final_code",
    [
        ([(True, 104)] * 12 + [(True, 0)], 13, 0),
        ([(True, 104)] + [(False, 1)] * 3, 4, 104),
        ([(True, 104), (False, 1), (False, 1), (True, 104), (False, 1), (False, 1), (True, 0)], 7, 0),
    ],
    ids=["twelve-successful-replacements", "three-failed-startups", "ready-resets-failure-budget"],
)
def test_recovery_limits_failed_starts_not_successful_replacements(tmp_path, outcomes, launches, final_code):
    launcher, client, _ = _run_recovery(tmp_path, outcomes)
    assert launcher.launch_job.call_count == launches
    assert client.send_request_before_shutdown.call_args.kwargs["request"].payload[JobOutcomeMsgKey.CODE] == final_code
    assert all(call.args[0] == {JobConstants.JOB_ID: "job-1"} for call in launcher.launch_job.call_args_list)


def test_recovery_retires_old_statistics_before_launch_with_previous_handle(tmp_path):
    handoff = None

    def publish_old_handoff(executor, index, handle):
        nonlocal handoff
        if index == 0:
            generation = executor.run_processes["job-1"][_RESTART_KEY].launch_id
            handoff = terminal_handoff_path(tmp_path / "job-1", launch_generation=generation)
            handoff.parent.mkdir(parents=True, exist_ok=True)
            handoff.write_text("earlier generation measurements")

    def check_replacement(executor, index, handle, ctx):
        if index == 1:
            assert not handoff.exists()
            assert ctx.get_prop(FLContextKey.JOB_RESTART_HANDLE) is not None

    launcher, _, _ = _run_recovery(
        tmp_path,
        [(True, 104), (True, 0)],
        on_wait=publish_old_handoff,
        on_launch=check_replacement,
    )
    assert launcher.launch_job.call_count == 2


@pytest.mark.parametrize("code", [0, 9, 102, 103, 127])
def test_nonretryable_outcomes_do_not_restart(tmp_path, code):
    launcher, _, _ = _run_recovery(tmp_path, [(True, code)])
    assert launcher.launch_job.call_count == 1


@pytest.mark.parametrize("limit, confirmed, is_leaf", [(0, True, True), (3, False, True), (3, True, False)])
def test_recovery_requires_leaf_site_enabled_policy_and_confirmed_death(tmp_path, limit, confirmed, is_leaf):
    launcher, client, _ = _run_recovery(tmp_path, [(True, 104)], limit=limit, confirmed=confirmed, is_leaf=is_leaf)
    assert launcher.launch_job.call_count == 1
    assert client.send_request_before_shutdown.call_args.kwargs["request"].payload[JobOutcomeMsgKey.CODE] == 104


def test_delayed_readiness_from_dead_cj_does_not_reset_budget(tmp_path):
    old_ids = []

    def status(executor, index, _):
        restart = executor.run_processes["job-1"][_RESTART_KEY]
        old_ids.append(restart.launch_id)
        if index:
            executor.notify_job_status("job-1", ClientStatus.STARTED, old_ids[index - 1])
            executor.notify_job_status("job-1", ClientStatus.READY, old_ids[index - 1])
            executor.notify_job_status("job-1", ClientStatus.STOPPED, old_ids[index - 1])
            assert restart.attempts == index
            assert executor.get_status("job-1") == ClientStatus.STARTING

    launcher, _, _ = _run_recovery(tmp_path, [(True, 104)] + [(False, 1)] * 3, on_wait=status)
    assert launcher.launch_job.call_count == 4
    assert len(set(old_ids)) == 4


@pytest.mark.parametrize("task_abort", [False, True])
def test_abort_during_replacement_stops_recovery(tmp_path, task_abort):
    def abort(executor, index, handle):
        if index == 1:
            (executor.abort_task if task_abort else executor.abort_app)("job-1")
            handle.terminate.assert_called_once()

    launcher, _, _ = _run_recovery(tmp_path, [(True, 104), (False, 1)], on_wait=abort)
    assert launcher.launch_job.call_count == 2


def test_task_abort_while_cj_is_dying_prevents_replay(tmp_path):
    def abort(executor, *_):
        with patch.object(executor, "_terminate_job") as terminate:
            executor.abort_task("job-1")
        terminate.assert_called_once()

    launcher, _, _ = _run_recovery(tmp_path, [(True, 104)], on_wait=abort)
    assert launcher.launch_job.call_count == 1


@pytest.mark.parametrize("task_abort", [False, True])
def test_abort_while_old_process_group_drains_prevents_recovery(tmp_path, task_abort):
    def abort(executor, _, handle):
        handle.can_restart.return_value = False  # CJ exited, but its children have not.
        with patch.object(executor, "_terminate_job") as terminate:
            (executor.abort_task if task_abort else executor.abort_app)("job-1")
        terminate.assert_called_once()
        handle.can_restart.return_value = True  # The old group finishes before wait returns.

    launcher, _, _ = _run_recovery(tmp_path, [(True, 104)], on_wait=abort)
    assert launcher.launch_job.call_count == 1


@pytest.mark.parametrize("ready", [False, True], ids=["initial-startup", "live-cj"])
def test_task_abort_does_not_permanently_disable_recovery(tmp_path, ready):
    def abort(executor, index, handle):
        if index == 0:
            handle.can_restart.return_value = False  # Still alive when abort_task arrives.
            code = handle.poll.return_value
            handle.poll.return_value = JobReturnCode.UNKNOWN
            executor.abort_task("job-1")
            handle.terminate.assert_not_called()
            assert not executor.run_processes["job-1"][_ABORT_REQUESTED_KEY]
            handle.can_restart.return_value = True  # Confirmed dead later.
            handle.poll.return_value = code

    launcher, _, _ = _run_recovery(tmp_path, [(ready, 104), (True, 0)], on_wait=abort)
    assert launcher.launch_job.call_count == 2


@pytest.mark.parametrize("limit, legacy", [(3, False), (3, True), (0, True)])
def test_command_start_does_not_reset_budget_and_preserves_graceful_abort(tmp_path, limit, legacy):
    def started(executor, index, handle):
        restart = executor.run_processes["job-1"][_RESTART_KEY]
        executor.notify_job_status("job-1", ClientStatus.STARTED, None if legacy else restart.launch_id)
        assert executor.get_status("job-1") == ClientStatus.STARTED
        assert restart.attempts == index
        if index == 0:
            handle.can_restart.return_value = False  # The task abort arrives while CJ is alive.
            handle.poll.return_value = JobReturnCode.UNKNOWN
            executor.abort_task("job-1")
            assert executor.client.cell.fire_and_forget.call_args.kwargs["topic"] == AdminCommandNames.ABORT_TASK
            assert not executor.run_processes["job-1"][_ABORT_REQUESTED_KEY]
            handle.terminate.assert_not_called()
            handle.can_restart.return_value = True
            handle.poll.return_value = JobReturnCode.EXECUTION_ERROR
        if index == limit:
            with patch.object(executor, "_terminate_job") as terminate:
                executor.abort_app("job-1")
            terminate.assert_called_once()
            handle.terminate.assert_not_called()
            assert executor.client.cell.fire_and_forget.call_args.kwargs["topic"] == AdminCommandNames.ABORT

    launcher, _, _ = _run_recovery(tmp_path, [(False, 1)] * (limit + 1), on_wait=started, limit=limit)
    assert launcher.launch_job.call_count == limit + 1


@pytest.mark.parametrize("legacy", [False, True])
def test_stopped_runner_teardown_failure_does_not_restart(tmp_path, legacy):
    def stopped(executor, *_):
        restart = executor.run_processes["job-1"][_RESTART_KEY]
        executor.notify_job_status("job-1", ClientStatus.STOPPED, None if legacy else restart.launch_id)
        executor.notify_job_status("job-1", ClientStatus.READY, restart.launch_id)  # Delayed delivery.
        assert executor.get_status("job-1") == ClientStatus.STOPPED

    launcher, _, _ = _run_recovery(tmp_path, [(True, 104)], on_wait=stopped)
    assert launcher.launch_job.call_count == 1


def test_legacy_started_crash_is_reported_as_execution_failure(tmp_path):
    def started(executor, *_):
        executor.notify_job_status("job-1", ClientStatus.STARTED)

    launcher, client, _ = _run_recovery(tmp_path, [(False, 1)], on_wait=started, limit=0)
    assert launcher.launch_job.call_count == 1
    assert (
        client.send_request_before_shutdown.call_args.kwargs["request"].payload[JobOutcomeMsgKey.CODE]
        == ProcessExitCode.EXCEPTION
    )


def test_ambiguous_replacement_submission_is_not_retried(tmp_path):
    def fail_launch(executor, *_):
        executor.run_processes["job-1"][_RESTART_KEY].launcher.launch_job.side_effect = RuntimeError("submission lost")

    launcher, client, _ = _run_recovery(tmp_path, [(True, 104)], on_wait=fail_launch)
    assert launcher.launch_job.call_count == 2
    assert client.send_request_before_shutdown.call_args.kwargs["request"].payload[JobOutcomeMsgKey.CODE] == 104


def test_replacement_failure_after_attachment_reaps_worker_before_releasing_resources(tmp_path):
    def fail_after_launch(executor, index, handle):
        if index:
            handle.terminate.assert_called_once()
            return
        launcher = executor.run_processes["job-1"][_RESTART_KEY].launcher
        launch = launcher.launch_job.side_effect

        def failing_launch(meta, ctx):
            def fail_event(event, _):
                if event == EventType.AFTER_JOB_LAUNCH:
                    raise RuntimeError("post-launch failure")

            ctx.get_engine().fire_event.side_effect = fail_event
            return launch(meta, ctx)

        launcher.launch_job.side_effect = failing_launch

    launcher, client, handles = _run_recovery(tmp_path, [(True, 104), (False, 1)], on_wait=fail_after_launch)
    assert launcher.launch_job.call_count == 2
    handles[-1].wait.assert_called_once()
    assert client.send_request_before_shutdown.call_args.kwargs["request"].payload[JobOutcomeMsgKey.CODE] == 104


@pytest.mark.parametrize("heartbeat_cleanup", [False, True])
@pytest.mark.parametrize("abort_before_publication", [False, True])
def test_replacement_handle_publication_keeps_abort_live(tmp_path, heartbeat_cleanup, abort_before_publication):
    def intercept_launch(executor, index, _):
        if index != 0:
            return
        launcher = executor.run_processes["job-1"][_RESTART_KEY].launcher
        launch = launcher.launch_job.side_effect

        def publish_and_abort(meta, ctx):
            if abort_before_publication:
                executor.abort_app("job-1", heartbeat_cleanup=heartbeat_cleanup)
            handle = launch(meta, ctx)
            publish_job_handle(handle, ctx)
            if not abort_before_publication:
                executor.abort_app("job-1", heartbeat_cleanup=heartbeat_cleanup)
            handle.terminate.assert_called_once()
            return handle

        launcher.launch_job.side_effect = publish_and_abort

    launcher, _, handles = _run_recovery(tmp_path, [(True, 104), (False, 9)], on_wait=intercept_launch)
    assert launcher.launch_job.call_count == 2
    handles[-1].terminate.assert_called_once()


def test_replacement_submission_failure_after_publication_reaps_accepted_worker(tmp_path):
    def intercept_launch(executor, index, handle):
        if index:
            handle.terminate.assert_called_once()
            return
        launcher = executor.run_processes["job-1"][_RESTART_KEY].launcher
        launch = launcher.launch_job.side_effect

        def publish_then_fail(meta, ctx):
            handle = launch(meta, ctx)
            publish_job_handle(handle, ctx)
            raise RuntimeError("launcher failed after accepting the replacement")

        launcher.launch_job.side_effect = publish_then_fail

    launcher, _, handles = _run_recovery(tmp_path, [(True, 104), (False, 1)], on_wait=intercept_launch)
    assert launcher.launch_job.call_count == 2
    handles[-1].terminate.assert_called_once()
    handles[-1].wait.assert_called_once()


def test_abort_while_replacement_launcher_has_not_returned(tmp_path):
    def intercept_launch(executor, index, _):
        if index != 0:
            return
        launcher = executor.run_processes["job-1"][_RESTART_KEY].launcher
        launch = launcher.launch_job.side_effect

        def abort_and_launch(*args):
            executor.abort_app("job-1")
            return launch(*args)

        launcher.launch_job.side_effect = abort_and_launch

    launcher, _, handles = _run_recovery(tmp_path, [(True, 104), (False, 9)], on_wait=intercept_launch)
    assert launcher.launch_job.call_count == 2
    handles[-1].terminate.assert_called_once()


EXPECTED_REPORTABLE_JOB_FAILURES = {
    ProcessExitCode.EXCEPTION: "exception",
    ProcessExitCode.UNSAFE_COMPONENT: "unsafe component",
    ProcessExitCode.CONFIG_ERROR: "config error",
    ProcessExitCode.INFRASTRUCTURE_ERROR: "infrastructure error",
    JobReturnCode.ABORTED: "aborted",
}


def test_reportable_job_failures_has_expected_codes():
    assert REPORTABLE_JOB_FAILURES == EXPECTED_REPORTABLE_JOB_FAILURES


def test_build_participant_resource_report_binds_parent_identity_and_removes_handoff(tmp_path):
    run_dir = Workspace._join_under_root(str(tmp_path), WorkspaceConstants.WORKSPACE_PREFIX + "job-1")
    handoff = _participant_summary(job_id="untrusted-job", participant_name="untrusted-site")
    handoff["resource_utilization"] = {
        "cpu_consumed": {"cpu_seconds": 12.5},
        "message_traffic": {"sent_bytes": 1200, "received_bytes": 800},
    }
    write_terminal_handoff(run_dir, handoff, launch_generation="test-launch")

    report_bytes = _build_participant_resource_report(
        job_id="job-1",
        participant_name="site-1",
        workspace=str(tmp_path),
        logger=MagicMock(),
        expected_generation="test-launch",
    )

    report = json.loads(report_bytes)
    assert report["participant_name"] == "site-1"
    assert report["job_id"] == "job-1"
    assert report["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": 12.5}
    assert report["resource_utilization"]["message_traffic"] == {"sent_bytes": 1200, "received_bytes": 800}
    assert not terminal_handoff_path(run_dir, launch_generation="test-launch").exists()


def test_slow_handoff_preserves_its_report_without_skipping_another_job(tmp_path):
    client = MagicMock(client_name="site-1")
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    client.send_request_before_shutdown.return_value.payload = {}
    executor = JobExecutor(client, "startup")
    for job_id in ("job-1", "job-2"):
        write_terminal_handoff(
            tmp_path / job_id,
            participant_measurements(_participant_summary(job_id=job_id)),
            launch_generation="test-launch",
        )
        executor.run_processes[job_id] = {
            RunProcessKey.JOB_HANDLE: MagicMock(),
            _RESTART_KEY: SimpleNamespace(launch_id="test-launch"),
        }
    manager, ctx = MagicMock(), MagicMock()
    entered, release = threading.Event(), threading.Event()
    read_handoff = client_executor_module.read_terminal_handoff

    def slow_read(run_dir, **kwargs):
        if Path(run_dir).name == "job-1":
            if threading.current_thread() not in threads.threads:
                threads.threads.append(threading.current_thread())
            entered.set()
            release.wait()
        return read_handoff(run_dir, **kwargs)

    with (
        ThreadGroup(release) as threads,
        patch.object(client_executor_module, "read_terminal_handoff", slow_read),
        patch.object(client_executor_module, "get_return_code", return_value=JobReturnCode.SUCCESS),
    ):
        caller = threads.start(
            executor._wait_child_process_finish, client, "job-1", {"gpu": [0]}, None, manager, str(tmp_path), ctx
        )
        assert entered.wait(10)
        manager.free_resources.assert_called_once()
        executor._wait_child_process_finish(client, "job-2", None, None, manager, str(tmp_path), ctx)
        second = client.send_request_before_shutdown.call_args.kwargs["request"].payload
        assert second[JobOutcomeMsgKey.JOB_ID] == "job-2"
        assert JobOutcomeMsgKey.RESOURCE_REPORT in second
        release.set()
        join_thread(caller)

    for call in client.send_request_before_shutdown.call_args_list:
        payload = call.kwargs["request"].payload
        report = json.loads(payload[JobOutcomeMsgKey.RESOURCE_REPORT][JobOutcomeMsgKey.PARTICIPANT_SUMMARY])
        assert report["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": 1.5}
        assert payload[JobOutcomeMsgKey.CODE] == JobReturnCode.SUCCESS
        assert not terminal_handoff_path(
            tmp_path / payload[JobOutcomeMsgKey.JOB_ID], launch_generation="test-launch"
        ).exists()
    assert client.send_request_before_shutdown.call_count == 2
    assert executor.run_processes == {}


def test_build_participant_resource_report_survives_missing_workspace(tmp_path):
    report_bytes = _build_participant_resource_report(
        job_id="job-1",
        participant_name="site-1",
        workspace=str(tmp_path / "already-cleaned"),
        logger=MagicMock(),
        expected_generation="test-launch",
    )

    report = json.loads(report_bytes)
    assert report["participant_name"] == "site-1"
    assert report["resource_capacity"] == {"status": "unavailable"}
    assert {"metric": "resource_capacity", "code": "observation_incomplete"} in report["issues"]


def test_abort_app_terminates_starting_job_without_worker_command():
    client = MagicMock()
    job_executor = JobExecutor(client=client, startup="startup")
    job_handle = MagicMock()
    pending = _PendingJobHandle()
    pending.attach(job_handle)
    job_executor.run_processes["job-1"] = {
        RunProcessKey.JOB_HANDLE: pending,
        RunProcessKey.STATUS: ClientStatus.STARTING,
    }

    ClientEngine.abort_app(SimpleNamespace(client_executor=job_executor), "job-1")

    job_handle.terminate.assert_called_once_with()
    client.cell.fire_and_forget.assert_not_called()
    assert job_executor.run_processes["job-1"][_ABORT_REQUESTED_KEY]


@pytest.mark.parametrize("heartbeat_cleanup", [False, True], ids=["admin_abort", "server_cleanup"])
def test_abort_app_terminates_registered_stopped_job_without_worker_command(heartbeat_cleanup):
    client = MagicMock()
    job_executor = JobExecutor(client=client, startup="startup")
    job_handle = MagicMock()
    job_executor.run_processes["job-1"] = {
        RunProcessKey.JOB_HANDLE: job_handle,
        RunProcessKey.STATUS: ClientStatus.STOPPED,
    }

    with patch.object(job_executor, "_terminate_job") as terminate_job:
        ClientEngine.abort_app(
            SimpleNamespace(client_executor=job_executor), "job-1", heartbeat_cleanup=heartbeat_cleanup
        )

    terminate_job.assert_called_once_with(job_handle, "job-1", heartbeat_cleanup)
    job_handle.terminate.assert_not_called()
    client.cell.fire_and_forget.assert_not_called()


def test_abort_app_does_not_terminate_unregistered_stopped_job():
    job_executor = MagicMock()
    job_executor.get_status.return_value = ClientStatus.STOPPED
    job_executor.get_run_processes_keys.return_value = []

    result = ClientEngine.abort_app(SimpleNamespace(client_executor=job_executor), "job-1")

    assert result == "Client app already stopped."
    job_executor.abort_app.assert_not_called()


def test_terminate_job_gives_stopped_worker_bounded_cleanup_grace():
    job_executor = JobExecutor(client=MagicMock(), startup="startup")
    job_executor.logger = MagicMock()
    job_handle = MagicMock(spec=JobHandleSpec)
    pending = _PendingJobHandle()
    pending.attach(job_handle)
    job_executor.run_processes["job-1"] = {
        RunProcessKey.JOB_HANDLE: pending,
        RunProcessKey.STATUS: ClientStatus.STOPPED,
    }

    with (
        patch("nvflare.private.fed.client.client_executor.time.time", side_effect=[10.0, 10.0, 20.1]),
        patch("nvflare.private.fed.client.client_executor.time.sleep") as sleep,
    ):
        job_executor._terminate_job(pending, "job-1", heartbeat_cleanup=True)

    sleep.assert_called_once_with(0.05)
    job_handle.terminate.assert_called_once_with()


def test_terminate_job_allows_stopped_worker_to_finish_during_cleanup_grace():
    job_executor = JobExecutor(client=MagicMock(), startup="startup")
    job_handle = MagicMock()
    job_executor.run_processes["job-1"] = {
        RunProcessKey.JOB_HANDLE: job_handle,
        RunProcessKey.STATUS: ClientStatus.STOPPED,
    }

    def finish_worker(_):
        job_executor.run_processes.pop("job-1")

    with patch("nvflare.private.fed.client.client_executor.time.sleep", side_effect=finish_worker) as sleep:
        job_executor._terminate_job(job_handle, "job-1", heartbeat_cleanup=True)

    sleep.assert_called_once_with(0.05)
    job_handle.terminate.assert_not_called()


def test_terminate_job_does_not_signal_handle_after_worker_was_reaped():
    job_executor = JobExecutor(client=MagicMock(), startup="startup")
    job_handle = MagicMock()

    job_executor._terminate_job(job_handle, "job-1")

    job_handle.terminate.assert_not_called()


def test_heartbeat_cleanup_propagates_non_user_abort_intent():
    job_executor = MagicMock()
    job_executor.get_status.return_value = ClientStatus.STARTED
    engine = SimpleNamespace(client_executor=job_executor)
    engine.abort_app = ClientEngine.abort_app.__get__(engine)

    Communicator(client_config={"client_name": "site-1"})._clean_up_runs(engine, ["job-1"])

    job_executor.abort_app.assert_called_once_with("job-1", heartbeat_cleanup=True)


def _write_deployed_meta(tmp_path, job_id, deployed_meta):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    workspace = Workspace(str(tmp_path), site_name="site-1")
    (tmp_path / job_id).mkdir()
    with open(workspace.get_job_meta_path(job_id), "w") as f:
        json.dump(deployed_meta, f)
    return workspace


def _make_start_app_inputs(tmp_path, job_id="job-1"):
    job_meta = {JobConstants.JOB_ID: job_id}
    workspace = _write_deployed_meta(tmp_path, job_id, job_meta)
    client = MagicMock()
    client.client_name = "site-1"
    client.cell.get_internal_listener_url.return_value = "tcp://parent:8002"
    client.cell.get_internal_listener_params.return_value = {}
    manager = FLContextManager(
        engine=MagicMock(),
        identity_name="site-1",
        job_id=job_id,
        private_stickers={
            FLContextKey.ARGS: SimpleNamespace(set=["parent-only=True"]),
            FLContextKey.WORKSPACE_OBJECT: workspace,
            FLContextKey.SERVER_CONFIG: [{"service": {"scheme": "grpc", "target": "parent:8002"}}],
        },
        public_stickers={},
    )
    fl_ctx = manager.new_context()
    fl_ctx.get_prop(FLContextKey.ARGS)  # Exercise a cached sticky parent value.
    return job_meta, workspace, client, fl_ctx


def test_start_app_pending_handle_operations_before_launcher_returns(tmp_path):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
    launcher = MagicMock()

    def launch_job(*_args):
        pending_handle = executor.run_processes[job_id][RunProcessKey.JOB_HANDLE]
        assert pending_handle.poll() is None
        with pytest.raises(RuntimeError, match="before it is attached"):
            pending_handle.wait()
        return MagicMock()

    launcher.launch_job.side_effect = launch_job

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        executor.start_app(
            client,
            job_id,
            job_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )


def test_start_app_does_not_create_an_unused_client_parent_counter(tmp_path):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
    launcher = MagicMock()

    def get_job_launcher(*_args, **_kwargs):
        assert not hasattr(executor, "f3_counters")
        return launcher

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", side_effect=get_job_launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        executor.start_app(
            client,
            job_id,
            job_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )


def test_start_app_removes_pending_process_when_launch_fails(tmp_path):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
    launcher = MagicMock()
    launcher.launch_job.side_effect = RuntimeError("launch failed")

    def retire_while_registered(cell, completed_job_id):
        assert cell is client.cell
        assert completed_job_id == job_id
        assert job_id in executor.run_processes

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch(
            "nvflare.private.fed.client.client_executor.retire_job_traffic",
            side_effect=retire_while_registered,
        ) as retire,
        pytest.raises(RuntimeError, match="launch failed"),
    ):
        executor.start_app(
            client,
            job_id,
            job_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )

    retire.assert_called_once_with(client.cell, job_id)
    assert job_id not in executor.run_processes


@pytest.mark.parametrize("schedule_count", [1, 3])
def test_client_cpu_attempt_marker_survives_parent_restart_and_redeployment(tmp_path, schedule_count):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    job_meta[JobMetaKey.SCHEDULE_COUNT.value] = schedule_count
    launcher = MagicMock()

    def launch_with_fresh_parent():
        executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
        with (
            patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
            patch.object(threading.Thread, "start", _skip_completion_monitor),
        ):
            executor.start_app(
                client,
                job_id,
                job_meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                None,
                None,
                None,
                fl_ctx,
            )
        return executor.run_processes[job_id][_RUN_STATE_KEY].prior_attempt_incomplete

    assert launch_with_fresh_parent() is False
    assert (tmp_path / _CPU_ATTEMPT_DIR / job_id).is_file()

    # The deployer removes the run directory; the scheduler's count need not
    # change when the parent is recreated, but the CPU marker must survive.
    shutil.rmtree(workspace.get_run_dir(job_id))
    (tmp_path / job_id).mkdir()
    with open(workspace.get_job_meta_path(job_id), "w") as f:
        json.dump(job_meta, f)

    assert launch_with_fresh_parent() is True


def test_client_cpu_attempt_marker_failure_does_not_claim_complete_or_block_launch(tmp_path, caplog):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    launcher = MagicMock()
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())

    with (
        patch("nvflare.private.fed.client.client_executor.os.open", side_effect=OSError("disk unavailable")),
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        executor.start_app(
            client,
            job_id,
            job_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )

    launcher.launch_job.assert_called_once()
    assert executor.run_processes[job_id][_RUN_STATE_KEY].prior_attempt_incomplete is True
    assert "could not persist resource-measurement attempt history" in caplog.text


def test_client_cpu_attempt_marker_existing_file_is_prior_attempt(tmp_path):
    workspace = _write_deployed_meta(tmp_path, "job-1", {JobConstants.JOB_ID: "job-1"})
    logger = MagicMock()

    assert _record_client_cpu_attempt(workspace, "job-1", logger) is False
    assert _record_client_cpu_attempt(workspace, "job-1", logger) is True
    logger.warning.assert_not_called()


@pytest.mark.parametrize("heartbeat_cleanup", [False, True], ids=["user_abort", "heartbeat_cleanup"])
def test_start_app_preserves_abort_intent_while_launcher_is_running(tmp_path, heartbeat_cleanup):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
    job_handle = MagicMock()
    job_handle.poll.return_value = JobReturnCode.ABORTED
    launcher = MagicMock()

    def launch_job(*_args):
        assert executor.get_status(job_id) == ClientStatus.STARTING
        ClientEngine.abort_app(SimpleNamespace(client_executor=executor), job_id, heartbeat_cleanup=heartbeat_cleanup)
        return job_handle

    launcher.launch_job.side_effect = launch_job

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        executor.start_app(
            client,
            job_id,
            job_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )

    pending_handle = executor.run_processes[job_id][RunProcessKey.JOB_HANDLE]
    if heartbeat_cleanup:
        job_handle._terminate_for_heartbeat_cleanup.assert_called_once_with()
        job_handle.terminate.assert_not_called()
    else:
        job_handle.terminate.assert_called_once_with()
        job_handle._terminate_for_heartbeat_cleanup.assert_not_called()
    client.cell.fire_and_forget.assert_not_called()
    assert pending_handle.poll() == JobReturnCode.ABORTED
    pending_handle.wait()
    job_handle.wait.assert_called_once_with()


@pytest.mark.parametrize("heartbeat_cleanup", [False, True], ids=["admin_abort", "server_cleanup"])
@pytest.mark.parametrize("abort_before_publish", [False, True], ids=["during_startup_wait", "before_publish"])
def test_start_app_can_abort_published_runtime_before_launcher_returns(
    tmp_path, heartbeat_cleanup, abort_before_publish
):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
    job_handle = MagicMock(spec=JobHandleSpec)
    launcher = MagicMock()

    def abort():
        ClientEngine.abort_app(SimpleNamespace(client_executor=executor), job_id, heartbeat_cleanup=heartbeat_cleanup)

    def launch_job(_meta, context):
        if abort_before_publish:
            abort()
            job_handle.terminate.assert_not_called()
        publish_job_handle(job_handle, context)
        if not abort_before_publish:
            abort()
        # The runtime is reclaimed while launch_job is still waiting for startup.
        job_handle.terminate.assert_called_once_with()
        return job_handle

    launcher.launch_job.side_effect = launch_job
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        executor.start_app(
            client, job_id, job_meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, fl_ctx
        )

    job_handle.terminate.assert_called_once_with()
    client.cell.fire_and_forget.assert_not_called()
    assert fl_ctx.get_prop(FLContextKey.JOB_HANDLE_CALLBACK) is None


def test_saved_abort_cannot_be_overtaken_during_handle_publication(tmp_path):
    job_id = "job-1"
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    executor.lock = ObservedLock(executor.lock)
    engine = SimpleNamespace(client_executor=executor)
    accepted = MagicMock(spec=JobHandleSpec)
    stops = []
    accepted.terminate.side_effect = lambda: stops.append("admin")
    accepted._terminate_for_heartbeat_cleanup = Mock(side_effect=lambda: stops.append("heartbeat"))
    attached = threading.Event()
    finish_publication = threading.Event()
    admin_returned = threading.Event()
    attach = _PendingJobHandle.attach

    def pause_after_attach(pending, handle):
        saved = attach(pending, handle)
        if saved is not None:
            attached.set()
            finish_publication.wait()
        return saved

    def launch(_meta, context):
        ClientEngine.abort_app(engine, job_id, heartbeat_cleanup=True)
        publish_job_handle(accepted, context)
        return accepted

    def start():
        executor.start_app(
            client, job_id, meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, ctx
        )

    def abort():
        ClientEngine.abort_app(engine, job_id)
        admin_returned.set()

    launcher = MagicMock()
    launcher.launch_job.side_effect = launch
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(_PendingJobHandle, "attach", pause_after_attach),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
        ThreadGroup(finish_publication) as threads,
    ):
        threads.start(start)
        assert attached.wait(5)
        threads.start(abort)
        wait_for(lambda: executor.lock.blocked.is_set() or admin_returned.is_set())
        assert executor.lock.blocked.is_set(), "admin abort overtook the saved stop during publication"
        assert not stops
    assert stops[0] == "heartbeat"


@pytest.mark.parametrize("publish_early", [False, True])
def test_start_app_removes_pending_handle_when_launch_fails(tmp_path, publish_early):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
    launcher = MagicMock()

    handle = MagicMock(spec=JobHandleSpec)

    def launch_job(*_args):
        assert any(
            option.startswith(f"{RequestHeader.CJ_LAUNCH_ID}=") for option in fl_ctx.get_prop(FLContextKey.ARGS).set
        )
        if publish_early:
            publish_job_handle(handle, fl_ctx)
        executor.abort_app(job_id)
        raise RuntimeError("launch failed")

    launcher.launch_job.side_effect = launch_job

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        pytest.raises(RuntimeError, match="launch failed"),
    ):
        executor.start_app(
            client,
            job_id,
            job_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )

    assert job_id not in executor.run_processes
    assert fl_ctx.get_prop(FLContextKey.ARGS).set == ["print_conf=True"]
    assert fl_ctx.get_prop(FLContextKey.JOB_HANDLE_CALLBACK) is None
    if publish_early:
        handle.terminate.assert_called()
        handle.wait.assert_called_once_with()


def test_start_app_removes_pending_handle_when_launcher_returns_no_handle(tmp_path):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
    launcher = MagicMock()
    launcher.launch_job.return_value = None

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        pytest.raises(RuntimeError, match="returned no job handle"),
    ):
        executor.start_app(
            client,
            job_id,
            job_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )

    assert job_id not in executor.run_processes


def test_pending_handle_heartbeat_cleanup_falls_back_to_terminate():
    job_handle = MagicMock(spec=JobHandleSpec)
    pending_handle = _PendingJobHandle()
    pending_handle.attach(job_handle)

    pending_handle.terminate(heartbeat_cleanup=True)

    job_handle.terminate.assert_called_once_with()


def test_start_app_does_not_replace_existing_launch_registration(tmp_path):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
    existing_handle = MagicMock()
    executor.run_processes[job_id] = {
        RunProcessKey.JOB_HANDLE: existing_handle,
        RunProcessKey.STATUS: ClientStatus.STARTING,
    }
    launcher = MagicMock()

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        pytest.raises(RuntimeError, match="still registered"),
    ):
        executor.start_app(
            client,
            job_id,
            job_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )

    assert executor.run_processes[job_id][RunProcessKey.JOB_HANDLE] is existing_handle
    launcher.launch_job.assert_not_called()


def test_start_app_allows_scheduler_metadata(tmp_path):
    job_id = "job-1"
    deployed_meta = {
        JobConstants.JOB_ID: job_id,
        JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "trusted/image:1"}}},
    }
    workspace = _write_deployed_meta(tmp_path, job_id, deployed_meta)
    start_meta = {
        **deployed_meta,
        JobMetaKey.STATUS.value: "RUNNING",
        JobMetaKey.JOB_CLIENTS.value: [{"name": "site-1", "token": "token-1"}],
    }

    client = MagicMock()
    client.client_name = "site-1"
    client.cell.get_internal_listener_url.return_value = "tcp://parent:8002"
    client.cell.get_internal_listener_params.return_value = {}
    fl_ctx = MagicMock()
    properties = {
        FLContextKey.WORKSPACE_OBJECT: workspace,
        FLContextKey.SERVER_CONFIG: [{"service": {"scheme": "grpc", "target": "parent:8002"}}],
    }
    fl_ctx.get_prop.side_effect = lambda key, *args, **kwargs: properties.get(key)
    fl_ctx.set_prop.side_effect = lambda key, value, **kwargs: properties.__setitem__(key, value)
    launcher = MagicMock()
    launcher.launch_job.return_value = MagicMock()

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
            client,
            job_id,
            start_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )

    launch_meta = launcher.launch_job.call_args.args[0]
    assert launch_meta[JobMetaKey.JOB_CLIENTS.value] == [{"name": "site-1", "token": "token-1"}]
    assert launch_meta[JobMetaKey.JOB_LAUNCHER_SPEC.value]["default"]["docker"]["image"] == "trusted/image:1"


@pytest.mark.parametrize(
    "deployed_byoc, start_byoc, expected_byoc",
    [
        (False, True, False),
        (True, False, True),
    ],
)
def test_start_app_uses_deployed_byoc_when_rewriting_job_meta(tmp_path, deployed_byoc, start_byoc, expected_byoc):
    job_id = "job-1"
    deployed_meta = {JobConstants.JOB_ID: job_id}
    if deployed_byoc:
        deployed_meta[AppValidationKey.BYOC] = True
    workspace = _write_deployed_meta(tmp_path, job_id, deployed_meta)
    start_meta = {JobConstants.JOB_ID: job_id, AppValidationKey.BYOC: start_byoc}

    client = MagicMock()
    client.client_name = "site-1"
    client.cell.get_internal_listener_url.return_value = "tcp://parent:8002"
    client.cell.get_internal_listener_params.return_value = {}
    fl_ctx = MagicMock()
    properties = {
        FLContextKey.WORKSPACE_OBJECT: workspace,
        FLContextKey.SERVER_CONFIG: [{"service": {"scheme": "grpc", "target": "parent:8002"}}],
    }
    fl_ctx.get_prop.side_effect = lambda key, *args, **kwargs: properties.get(key)
    fl_ctx.set_prop.side_effect = lambda key, value, **kwargs: properties.__setitem__(key, value)
    launcher = MagicMock()
    launcher.launch_job.return_value = MagicMock()

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
            client,
            job_id,
            start_meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            None,
            None,
            None,
            fl_ctx,
        )

    with open(workspace.get_job_meta_path(job_id)) as f:
        refreshed_meta = json.load(f)
    launch_meta = launcher.launch_job.call_args.args[0]

    assert refreshed_meta.get(AppValidationKey.BYOC, False) is expected_byoc
    assert launch_meta.get(AppValidationKey.BYOC, False) is expected_byoc


@pytest.mark.parametrize(
    "deployed_meta, start_meta, expected_key",
    [
        (
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "trusted/image:1"}}},
            },
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "attacker/image:latest"}}},
            },
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
        ),
        (
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"k8s": {"image": "trusted/image:1"}}},
            },
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"k8s": {"image": "attacker/image:latest"}}},
            },
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
        ),
        (
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.RESOURCE_SPEC.value: {"site-1": {"docker": {"image": "trusted/image:1"}}},
            },
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.RESOURCE_SPEC.value: {"site-1": {"docker": {"image": "attacker/image:latest"}}},
            },
            JobMetaKey.RESOURCE_SPEC.value,
        ),
        (
            {JobConstants.JOB_ID: "job-1", JobMetaKey.STUDY.value: "study-a"},
            {JobConstants.JOB_ID: "job-1", JobMetaKey.STUDY.value: "study-b"},
            JobMetaKey.STUDY.value,
        ),
        (
            {JobConstants.JOB_ID: "job-1", JobMetaKey.SCOPE.value: "scope-a"},
            {JobConstants.JOB_ID: "job-1", JobMetaKey.SCOPE.value: "scope-b"},
            JobMetaKey.SCOPE.value,
        ),
        (
            {JobConstants.JOB_ID: "job-1"},
            {JobConstants.JOB_ID: "job-2"},
            JobMetaKey.JOB_ID.value,
        ),
        (
            {JobConstants.JOB_ID: "job-1"},
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "attacker/image:latest"}}},
            },
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
        ),
        (
            {
                JobConstants.JOB_ID: "job-1",
                JobMetaKey.JOB_LAUNCHER_SPEC.value: {"default": {"docker": {"image": "trusted/image:1"}}},
            },
            {JobConstants.JOB_ID: "job-1"},
            JobMetaKey.JOB_LAUNCHER_SPEC.value,
        ),
    ],
)
def test_start_app_rejects_launch_metadata_drift(tmp_path, deployed_meta, start_meta, expected_key):
    job_id = "job-1"
    workspace = _write_deployed_meta(tmp_path, job_id, deployed_meta)
    client = MagicMock()
    client.client_name = "site-1"

    with patch("nvflare.private.fed.client.client_executor.get_job_launcher") as get_job_launcher:
        with pytest.raises(RuntimeError, match=expected_key):
            JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
                client=client,
                job_id=job_id,
                job_meta=start_meta,
                args=SimpleNamespace(workspace=str(tmp_path), set=[]),
                allocated_resource=None,
                token=None,
                resource_manager=None,
                fl_ctx=MagicMock(),
            )

    get_job_launcher.assert_not_called()
    with open(workspace.get_job_meta_path(job_id)) as f:
        assert json.load(f) == deployed_meta


@pytest.mark.parametrize(
    "return_code, reason",
    [
        *EXPECTED_REPORTABLE_JOB_FAILURES.items(),
        (JobReturnCode.SUCCESS, None),
        (JobReturnCode.EXECUTION_ERROR, None),
        (JobReturnCode.UNKNOWN, None),
    ],
)
def test_wait_child_process_reports_terminal_outcome_to_server(return_code, reason):
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    job_executor = JobExecutor(client=client, startup="startup")

    job_handle = MagicMock()
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}

    engine = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    with patch("nvflare.private.fed.client.client_executor.get_return_code", return_value=return_code):
        job_executor._wait_child_process_finish(
            client=client,
            job_id="job-1",
            allocated_resource=None,
            token=None,
            resource_manager=MagicMock(),
            workspace="/tmp/workspace",
            fl_ctx=fl_ctx,
        )

    job_handle.wait.assert_called_once()
    client.send_request_before_shutdown.assert_called_once()

    call_kwargs = client.send_request_before_shutdown.call_args.kwargs
    assert call_kwargs["target"] == FQCN.ROOT_SERVER
    assert call_kwargs["channel"] == CellChannel.SERVER_MAIN
    assert call_kwargs["topic"] == CellChannelTopic.REPORT_JOB_OUTCOME
    assert call_kwargs["topic"] == "report_job_failure"  # Existing servers still receive the report.
    assert call_kwargs["optional"] is True

    payload = call_kwargs["request"].payload
    assert payload[JobOutcomeMsgKey.JOB_ID] == "job-1"
    assert payload[JobOutcomeMsgKey.CODE] == return_code
    assert payload[JobOutcomeMsgKey.REASON] == reason
    resource_report = payload[JobOutcomeMsgKey.RESOURCE_REPORT]
    assert set(resource_report) == {JobOutcomeMsgKey.PARTICIPANT_SUMMARY}
    participant_summary = json.loads(resource_report[JobOutcomeMsgKey.PARTICIPANT_SUMMARY])
    assert participant_summary["participant_name"] == "site-1"
    assert participant_summary["job_id"] == "job-1"

    assert "job-1" not in job_executor.run_processes
    fl_ctx.set_prop.assert_any_call(FLContextKey.CURRENT_JOB_ID, "job-1", private=True, sticky=False)
    fl_ctx.set_prop.assert_any_call(FLContextKey.CLIENT_NAME, "site-1", private=True, sticky=False)
    engine.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, fl_ctx)


@pytest.mark.parametrize("report_status", ["accepted", "duplicate", "invalid", "too_late", None])
def test_wait_child_process_retires_credentials_and_acknowledged_attempt_marker(tmp_path, report_status):
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    client.send_request_before_shutdown.return_value.payload = {JobOutcomeMsgKey.RESOURCE_REPORT_STATUS: report_status}
    job_executor = JobExecutor(client=client, startup="startup")
    job_executor._cpu_seen_job_launches.add("job-1")
    marker = tmp_path / _CPU_ATTEMPT_DIR / "job-1"
    marker.parent.mkdir()
    marker.touch()

    def check_marker_before_ack(**_):
        assert marker.exists()
        assert "job-1" in job_executor._cpu_seen_job_launches
        return client.send_request_before_shutdown.return_value

    client.send_request_before_shutdown.side_effect = check_marker_before_ack
    job_handle = MagicMock()
    job_handle.poll.return_value = JobReturnCode.SUCCESS
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}
    run_dir = tmp_path / "job-1"
    run_dir.mkdir()
    (run_dir / "app_site-1").mkdir()
    write_job_cert(str(run_dir), b"cert", b"key")

    job_executor._wait_child_process_finish(
        client=client,
        job_id="job-1",
        allocated_resource=None,
        token=None,
        resource_manager=MagicMock(),
        workspace=str(tmp_path),
        fl_ctx=MagicMock(),
    )

    assert find_job_cert(str(run_dir)) is None
    assert (run_dir / "app_site-1").is_dir()

    acknowledged = report_status in {"accepted", "duplicate"}
    assert marker.exists() is not acknowledged
    assert ("job-1" in job_executor._cpu_seen_job_launches) is not acknowledged


def test_wait_child_process_retains_job_credential_when_wait_raises(tmp_path):
    client = MagicMock()
    client.client_name = "site-1"
    job_executor = JobExecutor(client=client, startup="startup")
    job_handle = MagicMock()
    job_handle.wait.side_effect = lambda: (_ for _ in ()).throw(RuntimeError("launcher lost the job"))
    job_handle.poll_interval = 0.01
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}
    run_dir = tmp_path / "job-1"
    run_dir.mkdir()
    write_job_cert(str(run_dir), b"cert", b"key")

    resources = MagicMock()
    _monitor_until_pending(
        job_executor,
        client=client,
        job_id="job-1",
        allocated_resource={"gpu": 1},
        token="token",
        resource_manager=resources,
        workspace=str(tmp_path),
        fl_ctx=MagicMock(),
    )

    assert find_job_cert(str(run_dir)) is not None
    assert job_executor.get_run_processes_keys() == ["job-1"]
    resources.free_resources.assert_not_called()
    client.send_request_before_shutdown.assert_not_called()


def test_wait_child_process_without_registration_cannot_remove_credentials(tmp_path):
    client = MagicMock()
    client.client_name = "site-1"
    job_executor = JobExecutor(client=client, startup="startup")
    job_executor.run_processes = {}
    run_dir = tmp_path / "job-1"
    run_dir.mkdir()
    write_job_cert(str(run_dir), b"cert", b"key")

    job_executor._wait_child_process_finish(
        client=client,
        job_id="job-1",
        allocated_resource=None,
        token=None,
        resource_manager=MagicMock(),
        workspace=str(tmp_path),
        fl_ctx=MagicMock(),
    )

    assert find_job_cert(str(run_dir)) is not None


def test_wait_child_process_preserves_launcher_infrastructure_error_over_rc_file(tmp_path):
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    job_executor = JobExecutor(client=client, startup="startup")
    job_handle = MagicMock()
    job_handle.poll.return_value = ProcessExitCode.INFRASTRUCTURE_ERROR
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}
    run_dir = tmp_path / "job-1"
    run_dir.mkdir()
    rc_file = run_dir / FLMetaKey.PROCESS_RC_FILE
    rc_file.write_text("0\n", encoding="utf-8")
    fl_ctx = MagicMock()

    job_executor._wait_child_process_finish(
        client=client,
        job_id="job-1",
        allocated_resource=None,
        token=None,
        resource_manager=MagicMock(),
        workspace=str(tmp_path),
        fl_ctx=fl_ctx,
    )

    payload = client.send_request_before_shutdown.call_args.kwargs["request"].payload
    assert payload[JobOutcomeMsgKey.CODE] == ProcessExitCode.INFRASTRUCTURE_ERROR
    assert not rc_file.exists()


@pytest.mark.parametrize(
    ("return_code", "process_status", "abort_requested", "expected_code"),
    [
        (JobReturnCode.SUCCESS, ClientStatus.STARTING, False, JobReturnCode.SUCCESS),
        (JobReturnCode.UNKNOWN, ClientStatus.STARTING, False, JobReturnCode.UNKNOWN),
        (JobReturnCode.UNKNOWN, ClientStatus.STARTED, False, JobReturnCode.UNKNOWN),
        (JobReturnCode.EXECUTION_ERROR, ClientStatus.STARTING, False, ProcessExitCode.INFRASTRUCTURE_ERROR),
        (JobReturnCode.EXECUTION_ERROR, ClientStatus.STARTED, False, ProcessExitCode.EXCEPTION),
        (JobReturnCode.EXECUTION_ERROR, ClientStatus.STOPPED, False, JobReturnCode.EXECUTION_ERROR),
        (JobReturnCode.EXECUTION_ERROR, ClientStatus.STARTING, True, JobReturnCode.EXECUTION_ERROR),
        (JobReturnCode.EXECUTION_ERROR, ClientStatus.STARTED, True, JobReturnCode.EXECUTION_ERROR),
        (JobReturnCode.ABORTED, ClientStatus.STARTED, False, JobReturnCode.ABORTED),
    ],
)
def test_wait_child_process_reports_terminal_return_code(return_code, process_status, abort_requested, expected_code):
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    job_executor = JobExecutor(client=client, startup="startup")

    job_handle = MagicMock()
    job_executor.run_processes = {
        "job-1": {
            RunProcessKey.JOB_HANDLE: job_handle,
            RunProcessKey.STATUS: process_status,
            _ABORT_REQUESTED_KEY: abort_requested,
        }
    }

    engine = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    with patch("nvflare.private.fed.client.client_executor.get_return_code", return_value=return_code):
        job_executor._wait_child_process_finish(
            client=client,
            job_id="job-1",
            allocated_resource=None,
            token=None,
            resource_manager=MagicMock(),
            workspace="/tmp/workspace",
            fl_ctx=fl_ctx,
        )

    client.send_request_before_shutdown.assert_called_once()
    payload = client.send_request_before_shutdown.call_args.kwargs["request"].payload
    assert payload[JobOutcomeMsgKey.CODE] == expected_code
    assert payload[JobOutcomeMsgKey.REASON] == REPORTABLE_JOB_FAILURES.get(expected_code)
    assert "job-1" not in job_executor.run_processes
    engine.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, fl_ctx)


def test_wait_child_process_assembles_only_the_child_handoff():
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    job_executor = JobExecutor(client=client, startup="startup")

    job_handle = MagicMock()
    job_executor.run_processes = {
        "job-1": {
            RunProcessKey.JOB_HANDLE: job_handle,
            RunProcessKey.STATUS: ClientStatus.STARTED,
            _ABORT_REQUESTED_KEY: False,
        }
    }
    engine = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    captured = {}

    def assemble_report(**kwargs):
        captured.update(kwargs)
        return {}

    with (
        patch("nvflare.private.fed.client.client_executor.get_return_code", return_value=JobReturnCode.SUCCESS),
        patch(
            "nvflare.private.fed.client.client_executor.assemble_participant_summary",
            side_effect=assemble_report,
        ),
    ):
        job_executor._wait_child_process_finish(
            client=client,
            job_id="job-1",
            allocated_resource=None,
            token=None,
            resource_manager=MagicMock(),
            workspace="/tmp/workspace",
            fl_ctx=fl_ctx,
        )

    assert "parent_f3" not in captured
    assert not hasattr(job_executor, "f3_counters")


def test_wait_child_process_cleans_up_when_terminal_outcome_report_fails():
    order = []
    client = MagicMock()
    client.client_name = "site-1"

    def fail_request(**_kwargs):
        order.append("send")
        raise RuntimeError("network unavailable")

    client.send_request_before_shutdown.side_effect = fail_request
    job_executor = JobExecutor(client=client, startup="startup")

    job_handle = MagicMock()
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}
    resource_manager = MagicMock()
    resource_manager.free_resources.side_effect = lambda **_kwargs: order.append("free")

    engine = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    def retire_while_fenced(cell, job_id):
        assert cell is client.cell
        assert job_id in job_executor._deleting_runs
        order.append("retire")

    with (
        patch("nvflare.private.fed.client.client_executor.get_return_code", return_value=JobReturnCode.SUCCESS),
        patch(
            "nvflare.private.fed.client.client_executor.retire_job_traffic",
            side_effect=retire_while_fenced,
        ),
    ):
        job_executor._wait_child_process_finish(
            client=client,
            job_id="job-1",
            allocated_resource={"gpu": [0]},
            token="token-1",
            resource_manager=resource_manager,
            workspace="/tmp/workspace",
            fl_ctx=fl_ctx,
        )

    assert order == ["free", "send", "retire"]
    resource_manager.free_resources.assert_called_once()
    client.send_request_before_shutdown.assert_called_once()
    assert "job-1" not in job_executor.run_processes
    engine.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, fl_ctx)


@pytest.mark.parametrize("return_code", [JobReturnCode.SUCCESS, JobReturnCode.ABORTED, ProcessExitCode.EXCEPTION])
@pytest.mark.parametrize("release_fails", [False, True])
def test_wait_child_process_releases_before_handoff_and_terminal_request(return_code, release_fails, caplog):
    order = []
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.side_effect = lambda **_kwargs: order.append("send")
    job_executor = JobExecutor(client=client, startup="startup")
    job_handle = MagicMock()
    job_handle.wait.side_effect = lambda: order.append("wait")
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}
    resource_manager = MagicMock()

    def release_resources(**_kwargs):
        order.append("free")
        if release_fails:
            raise RuntimeError("release failed")

    resource_manager.free_resources.side_effect = release_resources
    fl_ctx = MagicMock()

    def retire_while_fenced(cell, job_id):
        assert cell is client.cell
        assert job_id in job_executor._deleting_runs
        order.append("retire")

    with (
        patch("nvflare.private.fed.client.client_executor.get_return_code", return_value=return_code),
        patch(
            "nvflare.private.fed.client.client_executor._build_participant_resource_report",
            side_effect=lambda **_kwargs: order.append("handoff") or b"{}",
        ),
        patch(
            "nvflare.private.fed.client.client_executor.retire_job_traffic",
            side_effect=retire_while_fenced,
        ),
    ):
        job_executor._wait_child_process_finish(
            client=client,
            job_id="job-1",
            allocated_resource={"gpu": [0]},
            token="token-1",
            resource_manager=resource_manager,
            workspace="/tmp/workspace",
            fl_ctx=fl_ctx,
        )

    assert order == ["wait", "free", "handoff", "send", "retire"]
    resource_manager.free_resources.assert_called_once_with(
        resources={"gpu": [0]}, token="token-1", fl_ctx=client.engine.new_context.return_value
    )
    payload = client.send_request_before_shutdown.call_args.kwargs["request"].payload
    assert payload[JobOutcomeMsgKey.CODE] == return_code
    assert payload[JobOutcomeMsgKey.RESOURCE_REPORT] == {JobOutcomeMsgKey.PARTICIPANT_SUMMARY: b"{}"}
    assert ("admission release failed" in caplog.text) is release_fails
    assert "job-1" not in job_executor.run_processes
    fl_ctx.get_engine.return_value.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, fl_ctx)


def test_wait_child_process_requires_a_job_handle_before_releasing_resources():
    client = MagicMock()
    client.client_name = "site-1"
    job_executor = JobExecutor(client=client, startup="startup")
    job_executor.run_processes = {"job-1": {}}
    resource_manager = MagicMock()
    fl_ctx = MagicMock()

    job_executor._wait_child_process_finish(
        client=client,
        job_id="job-1",
        allocated_resource={"gpu": [0]},
        token="token-1",
        resource_manager=resource_manager,
        workspace="/tmp/workspace",
        fl_ctx=fl_ctx,
    )

    resource_manager.free_resources.assert_not_called()
    client.send_request_before_shutdown.assert_not_called()
    assert "job-1" in job_executor.run_processes
    fl_ctx.get_engine.return_value.fire_event.assert_not_called()


def test_wait_child_process_still_reports_outcome_when_resource_report_fails():
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    job_executor = JobExecutor(client=client, startup="startup")
    job_handle = MagicMock()
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}
    fl_ctx = MagicMock()

    with (
        patch("nvflare.private.fed.client.client_executor.get_return_code", return_value=JobReturnCode.SUCCESS),
        patch(
            "nvflare.private.fed.client.client_executor._build_participant_resource_report",
            side_effect=RuntimeError("bad handoff"),
        ),
    ):
        job_executor._wait_child_process_finish(
            client=client,
            job_id="job-1",
            allocated_resource=None,
            token=None,
            resource_manager=MagicMock(),
            workspace="/tmp/workspace",
            fl_ctx=fl_ctx,
        )

    client.send_request_before_shutdown.assert_called_once()
    payload = client.send_request_before_shutdown.call_args.kwargs["request"].payload
    assert payload[JobOutcomeMsgKey.CODE] == JobReturnCode.SUCCESS
    assert JobOutcomeMsgKey.RESOURCE_REPORT not in payload
    assert "job-1" not in job_executor.run_processes


def test_wait_child_process_skips_terminal_outcome_after_client_communication_stops():
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.return_value = None
    job_executor = JobExecutor(client=client, startup="startup")
    job_handle = MagicMock()
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}

    engine = MagicMock()
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    with patch("nvflare.private.fed.client.client_executor.get_return_code", return_value=JobReturnCode.SUCCESS):
        job_executor._wait_child_process_finish(
            client=client,
            job_id="job-1",
            allocated_resource=None,
            token=None,
            resource_manager=MagicMock(),
            workspace="/tmp/workspace",
            fl_ctx=fl_ctx,
        )

    client.send_request_before_shutdown.assert_called_once()
    assert "job-1" not in job_executor.run_processes
    engine.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, fl_ctx)


@pytest.fixture(autouse=True)
def _isolate_module_clocks(monkeypatch):
    isolate_time(
        monkeypatch,
        "nvflare.private.fed.client.client_executor",
    )


def test_start_app_rejects_cj_budget_drift(tmp_path):
    job_meta, workspace, client, context = _make_start_app_inputs(tmp_path)
    deployed = dict(job_meta, cj_resource_spec={"site-1": {"num_of_cpus": 0.5}})
    with open(workspace.get_job_meta_path("job-1"), "w") as stream:
        json.dump(deployed, stream)
    with pytest.raises(RuntimeError, match="cj_resource_spec"):
        JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
            client, "job-1", job_meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, context
        )


def test_cj_admission_rejects_locally_resolved_resident_config(tmp_path):
    job_meta, workspace, client, context = _make_start_app_inputs(tmp_path)
    job_meta["cj_resource_spec"] = {"site-1": {}}
    with open(workspace.get_job_meta_path("job-1"), "w") as stream:
        json.dump(job_meta, stream)
    config = Path(workspace.get_app_config_dir("job-1"))
    config.mkdir(parents=True)
    (config / "config_fed_client.json.default").write_text('{"execution_lifetime":"task"}')
    (config / "shadow").mkdir()
    (config / "shadow" / "config_fed_client.json").write_text('{"execution_lifetime":"job"}')
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher") as choose:
        with pytest.raises(ValueError, match="resolved client execution_lifetime"):
            JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
                client, "job-1", job_meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, context
            )
    choose.assert_not_called()


@pytest.mark.parametrize("budget", [{}, {"num_of_cpus": 0.5}, {"memory": "1Gi"}, {"num_of_gpus": 1}])
@pytest.mark.parametrize("backend", ["process", "docker", "custom", "slurm_server"])
def test_shared_admission_rejects_unsupported_task_cj_before_launch(tmp_path, budget, backend):

    job_meta, workspace, client, context = _make_start_app_inputs(tmp_path)
    job_meta["cj_resource_spec"] = {"site-1": budget}
    Path(workspace.get_job_meta_path("job-1")).write_text(json.dumps(job_meta))
    config_dir = Path(workspace.get_app_config_dir("job-1"))
    config_dir.mkdir(parents=True)
    (config_dir / "config_fed_client.json").write_text('{"execution_lifetime":"task"}')
    classes = {
        "process": ClientProcessJobLauncher,
        "docker": ClientDockerJobLauncher,
        "slurm_server": ServerSlurmJobLauncher,
    }
    launcher = object.__new__(classes[backend]) if backend in classes else Mock()
    launcher.launch_job = Mock(side_effect=AssertionError("unsupported task CJ must not start"))
    # Canonical site-policy validation has its own tests. Exercise the real
    # selected-launcher check at the shared parent admission boundary here.
    from nvflare.app_opt.job_launcher.slurm.task_launcher import SlurmTaskLauncher
    from nvflare.app_opt.task_launcher.k8s_launcher import K8sTaskLauncher

    task = object.__new__(SlurmTaskLauncher if backend == "slurm_server" else K8sTaskLauncher)
    policy = TaskExecutionPolicy(task)
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        with patch.object(TaskExecutionPolicy, "from_workspace", return_value=policy):
            with pytest.raises(ValueError, match="unsupported client-job/task launcher pairing"):
                JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
                    client,
                    "job-1",
                    job_meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    None,
                    None,
                    None,
                    context,
                )
    launcher.launch_job.assert_not_called()


def test_process_resident_compute_admission_keeps_legacy_behavior():

    context = Mock()
    context.get_identity_name.return_value = "site-1"
    with patch("nvflare.app_common.job_launcher.client_process_launcher.generate_client_command", return_value="CJ"):
        assert (
            ClientProcessJobLauncher().get_command({"resource_spec": {"@default": {"num_of_cpus": 2}}}, context) == "CJ"
        )


@pytest.mark.parametrize("gpus", [0, 1])
@pytest.mark.parametrize(
    "mode,module_name,class_name",
    [
        ("k8s", "nvflare.app_opt.job_launcher.k8s_launcher", "ClientK8sJobLauncher"),
        ("slurm", "nvflare.app_opt.job_launcher.slurm.launcher", "ClientSlurmJobLauncher"),
    ],
)
def test_shared_admission_reaches_matching_native_cj_launcher(tmp_path, mode, module_name, class_name, gpus):

    launcher = object.__new__(getattr(importlib.import_module(module_name), class_name))
    launcher.launch_job = Mock(side_effect=RuntimeError("native launch boundary reached"))
    job_meta, workspace, client, context = _make_start_app_inputs(tmp_path)
    job_meta["resource_spec"] = {"site-1": {"num_of_gpus": gpus}}
    job_meta["cj_resource_spec"] = {"site-1": {}}
    Path(workspace.get_job_meta_path("job-1")).write_text(json.dumps(job_meta))
    config_dir = Path(workspace.get_app_config_dir("job-1"))
    config_dir.mkdir(parents=True)
    (config_dir / "config_fed_client.json").write_text('{"execution_lifetime":"task"}')
    from nvflare.app_opt.job_launcher.slurm.task_launcher import SlurmTaskLauncher
    from nvflare.app_opt.task_launcher.k8s_launcher import K8sTaskLauncher

    task = object.__new__(SlurmTaskLauncher if mode == "slurm" else K8sTaskLauncher)
    policy = TaskExecutionPolicy(task)
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        with patch.object(TaskExecutionPolicy, "from_workspace", return_value=policy) as load_policy:
            with pytest.raises(RuntimeError, match="native launch boundary reached"):
                JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
                    client,
                    "job-1",
                    job_meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    None,
                    None,
                    None,
                    context,
                )
    load_policy.assert_called_once_with(workspace)
    launcher.launch_job.assert_called_once()


def test_shared_admission_keeps_job_site_independent_of_another_sites_task_policy(tmp_path):

    job_meta, workspace, client, context = _make_start_app_inputs(tmp_path)
    job_meta["cj_resource_spec"] = {"site-2": {}}
    Path(workspace.get_job_meta_path("job-1")).write_text(json.dumps(job_meta))
    launcher = Mock()
    launcher.launch_job.side_effect = RuntimeError("resident launch boundary reached")
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        with patch.object(TaskExecutionPolicy, "from_workspace") as native_policy:
            with pytest.raises(RuntimeError, match="resident launch boundary reached"):
                JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
                    client,
                    "job-1",
                    job_meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    None,
                    None,
                    None,
                    context,
                )
    native_policy.assert_not_called()
    launcher.launch_job.assert_called_once()


def test_cp_admission_uses_shared_resource_lifetime_classifier(tmp_path):
    job_meta, workspace, client, context = _make_start_app_inputs(tmp_path)
    job_meta["cj_resource_spec"] = {"site-1": {}}
    with open(workspace.get_job_meta_path("job-1"), "w") as stream:
        json.dump(job_meta, stream)
    config = Path(workspace.get_app_config_dir("job-1"))
    config.mkdir(parents=True)
    (config / "config_fed_client.json").write_text('{"execution_lifetime":"task"}')
    with patch(
        "nvflare.private.fed.client.client_executor.get_client_execution_lifetime", return_value="job"
    ) as classify:
        with pytest.raises(ValueError, match="resolved client execution_lifetime"):
            JobExecutor(client=client, startup=workspace.get_startup_kit_dir()).start_app(
                client, "job-1", job_meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, context
            )
    classify.assert_called_once_with({"execution_lifetime": "task"})


def test_cj_admission_uses_shared_resource_lifetime_classifier(tmp_path):

    job_meta, workspace, _, _ = _make_start_app_inputs(tmp_path)
    job_meta["cj_resource_spec"] = {"site-1": {}}
    with open(workspace.get_job_meta_path("job-1"), "w") as stream:
        json.dump(job_meta, stream)
    config = Path(workspace.get_app_config_dir("job-1"))
    config.mkdir(parents=True)
    (config / "config_fed_client.json").write_text('{"format_version":3,"execution_lifetime":"task","executors":[]}')
    args = SimpleNamespace(
        workspace=str(tmp_path),
        job_id="job-1",
        client_name="site-1",
        client_config="config/config_fed_client.json",
        set=[],
        sp_scheme="grpc",
        sp_target="server:8002",
    )
    with (
        patch(
            "nvflare.private.fed.client.client_app_runner.get_client_execution_lifetime", return_value="job"
        ) as classify,
        patch.object(ClientJsonConfigurator, "configure") as build,
    ):
        with pytest.raises(ValueError, match="resolved client execution_lifetime"):
            ClientAppRunner().create_client_runner(workspace.get_app_dir("job-1"), args, "config", MagicMock(), False)
    classify.assert_called_once()
    assert classify.call_args.args[0]["execution_lifetime"] == "task"
    build.assert_not_called()


@pytest.mark.parametrize("abort_before_publish", [False, True])
def test_recovery_launch_keeps_early_handle_abort_support(tmp_path, abort_before_publish):
    def on_launch(executor, index, handle, ctx):
        if index != 1:
            return
        if abort_before_publish:
            executor.abort_app("job-1")
            handle.terminate.assert_not_called()
        publish_job_handle(handle, ctx)
        if not abort_before_publish:
            executor.abort_app("job-1")
        # Cleanup must happen while the replacement launcher still waits for
        # scheduling, rather than only after returning its job handle.
        handle.terminate.assert_called_once()

    launcher, _, handles = _run_recovery(tmp_path, [(True, 104), (False, 0)], on_launch=on_launch)
    assert launcher.launch_job.call_count == 2
    handles[1].terminate.assert_called_once()


@pytest.mark.parametrize("replacement", ["report", "missing", "failed-launch"])
def test_recovery_never_substitutes_stale_report_when_retirement_fails(tmp_path, replacement):
    def publish(executor, index, handle):
        if index == 1 and replacement != "report":
            return
        generation = executor.run_processes["job-1"][_RESTART_KEY].launch_id
        unavailable = {"status": "unavailable"}
        write_terminal_handoff(
            tmp_path / "job-1",
            {
                "duration_seconds": 10,
                "resource_capacity": unavailable,
                "resource_time": unavailable,
                "resource_utilization": {
                    "cpu_consumed": {"cpu_seconds": index + 2},
                    "message_traffic": unavailable,
                },
                "issues": [
                    {"metric": metric, "code": "observation_incomplete"}
                    for metric in (
                        "resource_capacity",
                        "resource_time",
                        "message_traffic",
                    )
                ],
            },
            launch_generation=generation,
        )
        if index == 0 and replacement == "failed-launch":
            executor.run_processes["job-1"][_RESTART_KEY].launcher.launch_job.side_effect = RuntimeError(
                "launch failed"
            )

    with patch("nvflare.private.fed.client.client_executor.remove_terminal_handoff", side_effect=OSError("denied")):
        _, client, _ = _run_recovery(tmp_path, [(True, 104), (True, 0)], on_wait=publish)

    payload = client.send_request_before_shutdown.call_args.kwargs["request"].payload
    report = json.loads(payload[JobOutcomeMsgKey.RESOURCE_REPORT][JobOutcomeMsgKey.PARTICIPANT_SUMMARY])
    cpu = report["resource_utilization"]["cpu_consumed"]
    if replacement == "report":
        assert cpu == {"cpu_seconds": 3, "status": "partial"}
        assert report["duration_seconds"] == 10
        assert {"metric": "cpu_consumed", "code": "observation_incomplete"} in report["issues"]
    else:
        assert cpu == {"status": "unavailable"}
        assert "duration_seconds" not in report
    assert payload[JobOutcomeMsgKey.CODE] == (104 if replacement == "failed-launch" else 0)


@pytest.mark.parametrize("status", [None, "partial", "unavailable"])
def test_client_relaunch_preserves_values_and_marks_only_usable_measurements_partial(tmp_path, status):
    run_dir = tmp_path / (WorkspaceConstants.WORKSPACE_PREFIX + "job-1")
    handoff = _participant_summary()
    measurements = {
        "resource_capacity": handoff["resource_capacity"],
        "resource_time": handoff["resource_time"],
        **handoff["resource_utilization"],
    }
    for metric, value in measurements.items():
        if status == "unavailable":
            value.clear()
        if status:
            value["status"] = status
            handoff["issues"].append({"metric": metric, "code": "attribution_incomplete"})
    write_terminal_handoff(run_dir, handoff, launch_generation="test-launch")
    report = json.loads(
        _build_participant_resource_report(
            job_id="job-1",
            participant_name="site-1",
            workspace=str(tmp_path),
            logger=MagicMock(),
            prior_attempt_incomplete=True,
            expected_generation="test-launch",
        )
    )
    actual = {
        "resource_capacity": report["resource_capacity"],
        "resource_time": report["resource_time"],
        **report["resource_utilization"],
    }
    for metric, value in measurements.items():
        assert actual[metric] == {**value, "status": "unavailable" if status == "unavailable" else "partial"}
        codes = [issue["code"] for issue in report["issues"] if issue["metric"] == metric]
        expected = (["attribution_incomplete"] if status else []) + (
            [] if status == "unavailable" else ["observation_incomplete"]
        )
        assert codes == expected
    assert report["duration_seconds"] == handoff["duration_seconds"]
    assert not terminal_handoff_path(run_dir, launch_generation="test-launch").exists()


@pytest.mark.parametrize("replacement", [False, True], ids=["initial", "replacement"])
@pytest.mark.parametrize("failure", ["terminate", "wait", "query"])
def test_launch_errors_retain_cleanup_owner_and_credentials(tmp_path, replacement, failure):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    launcher = MagicMock()
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.poll.return_value = JobReturnCode.UNKNOWN
    getattr(accepted, failure if failure != "query" else "poll").side_effect = RuntimeError(f"{failure} unavailable")
    if failure == "terminate":
        accepted.wait.side_effect = lambda: (_ for _ in ()).throw(RuntimeError("observation unavailable"))
    run_dir = tmp_path / "job-1"
    write_job_cert(str(run_dir), b"cert", b"key")
    old = MagicMock(spec=JobHandleSpec)
    old.poll.return_value = ProcessExitCode.INFRASTRUCTURE_ERROR
    old.can_restart.return_value = True
    calls = []

    def launch(_meta, context):
        calls.append(True)
        if replacement and len(calls) == 1:
            return old
        publish_job_handle(accepted, context)
        if failure != "query":
            raise RuntimeError("startup reply lost")
        return accepted

    launcher.launch_job.side_effect = launch
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        if replacement or failure == "query":
            executor.start_app(
                client,
                "job-1",
                meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                {"gpu": 1},
                "token",
                resources,
                ctx,
            )
        else:
            with pytest.raises(RuntimeError):
                executor.start_app(
                    client,
                    "job-1",
                    meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    {"gpu": 1},
                    "token",
                    resources,
                    ctx,
                )
    if replacement or failure == "query":
        _monitor_until_pending(executor, client, "job-1", {"gpu": 1}, "token", resources, str(tmp_path), ctx)
    assert executor.get_run_processes_keys() == ["job-1"]
    assert find_job_cert(str(run_dir)) is not None
    resources.free_resources.assert_not_called()
    client.send_request_before_shutdown.assert_not_called()
    assert len(calls) == (2 if replacement else 1)
    if failure != "query":
        accepted.terminate.assert_called_once()
        assert accepted.wait.call_count >= 1  # Cancellation failure cannot skip observation.


@pytest.mark.parametrize("replacement", [False, True], ids=["initial", "replacement"])
@pytest.mark.parametrize("failure", ["raise", "none", "event", "saved-abort"])
def test_failed_launch_observes_published_handle_even_when_cancel_fails(tmp_path, replacement, failure):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    launcher = MagicMock()
    resources = MagicMock()
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.terminate.side_effect = RuntimeError("cancellation unavailable")
    accepted.wait.side_effect = lambda: (_ for _ in ()).throw(RuntimeError("observation unavailable"))
    old = MagicMock(spec=JobHandleSpec)
    old.poll.return_value = ProcessExitCode.INFRASTRUCTURE_ERROR
    old.can_restart.return_value = True
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    calls = []
    callback_errors = []

    def launch(_meta, context):
        calls.append(True)
        if replacement and len(calls) == 1:
            return old
        if failure == "saved-abort":
            executor.abort_app("job-1")
        # Callback errors can trigger a launcher's own blocking cleanup. Saved
        # abort errors must reach JobExecutor only after this launcher returns.
        try:
            publish_job_handle(accepted, context)
        except BaseException as e:
            callback_errors.append(e)
            raise
        if failure == "raise":
            raise RuntimeError("startup reply lost")
        if failure == "none":
            return None
        return accepted

    def event(event_type, context):
        if failure == "event" and event_type == EventType.AFTER_JOB_LAUNCH and len(calls) == (2 if replacement else 1):
            raise RuntimeError("event failed")

    ctx.get_engine().fire_event.side_effect = event
    launcher.launch_job.side_effect = launch
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
    ):
        if replacement:
            executor.start_app(
                client,
                "job-1",
                meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                {"gpu": 1},
                "token",
                resources,
                ctx,
            )
            _monitor_until_pending(executor, client, "job-1", {"gpu": 1}, "token", resources, str(tmp_path), ctx)
        else:
            with pytest.raises(JobCleanupPending):
                executor.start_app(
                    client,
                    "job-1",
                    meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    {"gpu": 1},
                    "token",
                    resources,
                    ctx,
                )
    assert callback_errors == []
    process = executor.run_processes["job-1"]
    assert process[RunProcessKey.JOB_HANDLE]._job_handle is accepted
    assert process[_RUN_STATE_KEY].monitor_args[2:5] == ({"gpu": 1}, "token", resources)
    assert process[_RUN_STATE_KEY].phase == _RunPhase.CLEANUP_PENDING
    executor.abort_app("job-1", heartbeat_cleanup=True)  # Retry the completed cancel while reusing the monitor.
    assert process[RunProcessKey.JOB_HANDLE].cancel_call.done.wait(5)
    assert ctx.get_prop(FLContextKey.JOB_HANDLE_CALLBACK) is None
    assert find_job_cert(str(tmp_path / "job-1")) is not None
    resources.free_resources.assert_not_called()
    client.send_request_before_shutdown.assert_not_called()
    assert accepted.terminate.call_count == (3 if failure == "saved-abort" else 2)
    assert accepted.wait.call_count >= 1
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        pytest.raises(RuntimeError, match="still registered"),
    ):
        executor.start_app(
            client,
            "job-1",
            meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            {"gpu": 1},
            "new-token",
            resources,
            ctx,
        )
    assert len(calls) == (2 if replacement else 1)


@pytest.mark.parametrize("blocked", ["terminate", "wait", "saved-abort"])
def test_launch_cleanup_timeout_retains_owner_and_does_not_block_observation(tmp_path, blocked, cleanup_deadlines):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    entered = threading.Event()
    observed = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors = []

    def block():
        entered.set()
        release.wait()

    accepted.terminate.side_effect = block if blocked != "wait" else None

    def observe():
        observed.set()
        if blocked == "wait":
            block()

    accepted.wait.side_effect = observe
    cleanup_deadlines.cancel_pending = entered if blocked != "wait" else None
    cleanup_deadlines.observation_pending = observed if blocked == "wait" else None
    launcher = MagicMock()

    def launch(_meta, context):
        if blocked == "saved-abort":
            executor.abort_app("job-1")
        publish_job_handle(accepted, context)
        if blocked == "saved-abort":
            return accepted
        raise RuntimeError("startup reply lost")

    launcher.launch_job.side_effect = launch

    def start():
        try:
            executor.start_app(
                client,
                "job-1",
                meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                {"gpu": 1},
                "token",
                MagicMock(),
                ctx,
            )
        except Exception as e:
            errors.append(e)
        finally:
            finished.set()

    # Local fake callbacks are unit boundaries. Release every gate before joining
    # the product's control workers, even if the readiness/completion check fails.
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            with ThreadGroup(release) as threads:
                threads.start(start)
                assert entered.wait(2)
                assert observed.wait(2)  # A stuck cancel cannot suppress observation.
                assert finished.wait(2)  # Caller returns while the operation remains gated.
                assert len(errors) == 1 and isinstance(errors[0], JobCleanupPending)
                assert executor.get_run_processes_keys() == ["job-1"]
                assert not release.is_set()
                pending = executor.run_processes["job-1"][RunProcessKey.JOB_HANDLE]
                monitor = executor.run_processes["job-1"][_RUN_STATE_KEY].monitor
                aborted = threading.Event()

                def abort_again():
                    executor.abort_app("job-1")
                    aborted.set()

                threads.start(abort_again)
                assert aborted.wait(2)
                assert accepted.terminate.call_count == (2 if blocked == "wait" else 1)
                assert accepted.wait.call_count == 1
        finally:
            release.set()
            if "monitor" in locals():
                join_thread(monitor)
    assert accepted.terminate.call_count == (2 if blocked == "wait" else 1)
    assert accepted.wait.call_count == 1
    assert executor.get_run_processes_keys() == []  # Late evidence now completes cleanup.
    assert executor._deleting_runs == set()


def test_failed_cancellation_with_confirmed_observation_preserves_original_launch_error(tmp_path):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.terminate.side_effect = RuntimeError("cancel unavailable")
    launcher = MagicMock()

    def launch(_meta, context):
        publish_job_handle(accepted, context)
        raise ValueError("startup failure")

    launcher.launch_job.side_effect = launch
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        pytest.raises(ValueError, match="startup failure"),
    ):
        executor.start_app(
            client, "job-1", meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, ctx
        )
    accepted.wait.assert_called_once()
    assert executor.get_run_processes_keys() == []


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_launch_cleanup_wait_must_be_finite_and_positive(value):
    def setting(*, name, **kwargs):
        return value if name == "job_launch_cleanup_timeout" else kwargs["default"]

    with (
        patch("nvflare.private.fed.client.client_executor.ConfigService.get_float_var", side_effect=setting),
        pytest.raises(ValueError, match="finite and positive"),
    ):
        JobExecutor(MagicMock(), "startup")


@pytest.mark.parametrize("settled", [False, True])
def test_monitor_start_failure_keeps_owner_until_observation_succeeds(tmp_path, settled):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    if not settled:
        accepted.wait.side_effect = lambda: (_ for _ in ()).throw(RuntimeError("query unavailable"))
    launcher = MagicMock()
    launcher.launch_job.return_value = accepted

    failed = []

    def start(thread):
        if getattr(thread._target, "__name__", "") == "_wait_child_process_finish" and not failed:
            failed.append(True)
            raise RuntimeError("cannot start monitor")
        _start_thread(thread)

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", start),
        pytest.raises(RuntimeError, match="cannot start monitor" if settled else "cleanup remains owned by CP"),
    ):
        executor.start_app(
            client, "job-1", meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, ctx
        )
    assert executor.get_run_processes_keys() == ([] if settled else ["job-1"])
    accepted.terminate.assert_called_once()
    if settled:
        accepted.wait.assert_called_once()
    else:
        assert accepted.wait.call_count >= 1


@pytest.mark.parametrize("failed_control", [1, 2], ids=["cancel-thread", "observe-thread"])
def test_cleanup_control_start_failure_cannot_skip_observation_or_drop_owner(
    tmp_path, failed_control, cleanup_deadlines
):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    launcher = MagicMock()
    control_calls = []
    startup_failed = threading.Event()
    cleanup_deadlines.observation_pending = startup_failed if failed_control == 2 else None

    def launch(_meta, context):
        publish_job_handle(accepted, context)
        raise ValueError("startup failure")

    def start(thread):
        control_calls.append(thread)
        if len(control_calls) == failed_control:
            startup_failed.set()
            raise RuntimeError("cannot start control operation")
        _start_thread(thread)

    launcher.launch_job.side_effect = launch
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", start),
        pytest.raises(JobCleanupPending if failed_control == 2 else ValueError),
    ):
        executor.start_app(
            client, "job-1", meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, ctx
        )
    assert len(control_calls) == 2
    assert executor.get_run_processes_keys() == (["job-1"] if failed_control == 2 else [])
    if failed_control == 1:
        accepted.terminate.assert_not_called()
        accepted.wait.assert_called_once()
    else:
        accepted.terminate.assert_called_once()
        accepted.wait.assert_not_called()


def test_admin_abort_after_monitor_error_still_cancels_and_observes(tmp_path, monkeypatch):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.wait.side_effect = lambda: (_ for _ in ()).throw(RuntimeError("query unavailable"))
    pending = _PendingJobHandle()
    pending.attach(accepted)
    resources = MagicMock()
    owner = {
        RunProcessKey.JOB_HANDLE: pending,
        RunProcessKey.STATUS: ClientStatus.STARTED,
    }
    executor.run_processes["job-1"] = owner
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    _monitor_until_pending(executor, client, "job-1", {"gpu": 1}, "token", resources, str(tmp_path), ctx)
    assert executor.check_status("job-1").startswith("observation pending (age ")
    assert executor.get_status("job-1") == ClientStatus.STARTED
    accepted.terminate.assert_not_called()
    clock = isolate_time(monkeypatch, "nvflare.private.fed.client.client_executor")
    clock.time = Mock(side_effect=[0.0, 11.0])
    executor.abort_app("job-1")
    assert client.cell.fire_and_forget.call_args.kwargs["topic"] == AdminCommandNames.ABORT
    accepted.terminate.assert_called_once()
    wait_for(lambda: accepted.wait.call_count >= 2)
    executor.abort_app("job-1")  # Completed failures may be retried by another explicit request.
    assert accepted.terminate.call_count == 2
    wait_for(lambda: accepted.wait.call_count >= 3)
    assert executor.run_processes["job-1"] is owner
    assert find_job_cert(str(tmp_path / "job-1")) is not None
    resources.free_resources.assert_not_called()


@pytest.mark.parametrize("status", [ClientStatus.STARTING, ClientStatus.STARTED, ClientStatus.STOPPED])
def test_retained_heartbeat_reuses_inflight_cancel_without_new_calls(tmp_path, caplog, status, cleanup_deadlines):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    cancel_entered = threading.Event()
    release_cancel = threading.Event()

    def terminate(adapter, _attempt):
        cancel_entered.set()
        release_cancel.wait()
        adapter.exited.set()

    adapter = ControlledProcessAdapter(terminate)
    cleanup_deadlines.cancel_pending = cancel_entered
    cleanup_deadlines.observation_pending = adapter.wait_entered
    launcher = MagicMock()

    def launch(_meta, context):
        publish_job_handle(ProcessHandle(process_adapter=adapter), context)
        raise RuntimeError("launch failed after publication")

    launcher.launch_job.side_effect = launch
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            with pytest.raises(JobCleanupPending):
                executor.start_app(
                    client,
                    "job-1",
                    meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    None,
                    None,
                    None,
                    ctx,
                )
            owner = executor.run_processes["job-1"]
            state = owner[_RUN_STATE_KEY]
            with executor.lock:
                owner[RunProcessKey.STATUS] = status
            assert executor.get_status("job-1") == status
            assert executor.check_status("job-1").startswith("cleanup pending")
            caplog.clear()
            for _ in range(3):
                ClientEngine.abort_app(SimpleNamespace(client_executor=executor), "job-1", heartbeat_cleanup=True)
            assert adapter.terminate_count == adapter.wait_count == 1
            client.cell.fire_and_forget.assert_not_called()
            assert not caplog.records
        finally:
            release_cancel.set()
            adapter.exited.set()
            if "state" in locals():
                join_thread(state.monitor)


@pytest.mark.parametrize("retained", [False, True], ids=["active", "retained"])
def test_admin_delete_preserves_owned_run_files_and_credentials(tmp_path, retained):
    _, workspace, client, _ = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    owner = {RunProcessKey.JOB_HANDLE: MagicMock(), RunProcessKey.STATUS: ClientStatus.STARTED}
    executor.run_processes["job-1"] = owner
    if retained:
        executor._retain_cleanup_owner("job-1")
    run_dir = tmp_path / "job-1"
    launch_file = run_dir / "launch.sh"
    launch_file.write_text("owned launch files")
    write_job_cert(str(run_dir), b"cert", b"key")
    engine = SimpleNamespace(args=SimpleNamespace(workspace=str(tmp_path)), client_executor=executor)
    result = ClientEngine.delete_run(engine, "job-1")
    assert "cannot delete an owned run" in result
    assert launch_file.read_text() == "owned launch files"
    assert find_job_cert(str(run_dir)) is not None
    assert executor.run_processes["job-1"] is owner
    executor.run_processes.pop("job-1")  # Reconciled by the operator.
    assert "Delete run folder" in ClientEngine.delete_run(engine, "job-1")
    assert not run_dir.exists()


def test_unconfirmed_launch_cleanup_logs_original_launch_error(tmp_path, caplog):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    handle = MagicMock(spec=JobHandleSpec)
    handle.wait.side_effect = RuntimeError("observation unavailable")
    launcher = MagicMock()

    def launch(_meta, context):
        publish_job_handle(handle, context)
        raise ValueError("startup handshake failed")

    launcher.launch_job.side_effect = launch
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        pytest.raises(JobCleanupPending),
    ):
        executor.start_app(
            client, "job-1", meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, ctx
        )
    assert "launch failed: ValueError: startup handshake failed" in caplog.text
    assert "observation unavailable" in caplog.text


def test_slow_run_delete_allows_heartbeat_status_notifications_and_other_launches(tmp_path):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path, "new-job")
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    executor.run_processes["other-job"] = {RunProcessKey.STATUS: ClientStatus.STARTING}
    deleting = tmp_path / "delete-me"
    deleting.mkdir()
    deleting_meta = {JobConstants.JOB_ID: "delete-me"}
    Path(workspace.get_job_meta_path("delete-me")).write_text(json.dumps(deleting_meta))
    entered = threading.Event()
    release = threading.Event()
    progressed = threading.Event()
    from nvflare.private.fed.client import client_executor as subject

    rmtree = subject.shutil.rmtree
    launcher = MagicMock()

    def slow_delete(path):
        entered.set()
        release.wait()
        rmtree(path)

    def other_operations():
        assert executor.get_run_processes_keys() == ["other-job"]  # Heartbeat job list.
        assert executor.get_status("other-job") == ClientStatus.STARTING
        executor.notify_job_status("other-job", ClientStatus.READY)
        assert executor.check_status("other-job") == "started"
        with pytest.raises(RuntimeError, match="deletion in progress"):
            executor.check_start_app("delete-me")
        # Also fence a caller that passed the public check before deletion began.
        with pytest.raises(RuntimeError, match="deletion in progress"):
            executor.start_app(
                client,
                "delete-me",
                deleting_meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                None,
                None,
                None,
                ctx,
            )
        executor.start_app(
            client, "new-job", meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, ctx
        )
        assert set(executor.get_run_processes_keys()) == {"other-job", "new-job"}
        progressed.set()

    with (
        patch.object(subject.shutil, "rmtree", side_effect=slow_delete),
        patch.object(subject, "get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", _skip_completion_monitor),
        ThreadGroup(release) as threads,
    ):
        threads.start(executor.delete_run, "delete-me", str(deleting))
        assert entered.wait(5)
        threads.start(other_operations)
        assert progressed.wait(5)  # These operations finish with rmtree still gated.
        assert not release.is_set()
    assert not deleting.exists()
    assert executor.check_start_app("delete-me") == ClientStatus.STOPPED


def test_failed_run_delete_does_not_leave_job_fenced(tmp_path):
    _, workspace, client, _ = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    from nvflare.private.fed.client import client_executor as subject

    with (
        patch.object(subject.shutil, "rmtree", side_effect=OSError("storage unavailable")),
        pytest.raises(OSError, match="storage unavailable"),
    ):
        executor.delete_run("job-1", str(tmp_path / "job-1"))
    assert executor.check_start_app("job-1") == ClientStatus.STOPPED


@pytest.mark.parametrize("cancel_error", [None, RuntimeError("cancel failed")])
def test_retained_admin_abort_finishes_through_monitor(tmp_path, caplog, cancel_error):
    import logging

    _, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.poll.return_value = JobReturnCode.ABORTED
    observed = threading.Event()
    release_wait = threading.Event()

    def observe():
        observed.set()
        release_wait.wait()

    def cancel():
        release_wait.set()
        if cancel_error:
            raise cancel_error

    accepted.terminate.side_effect = cancel
    accepted.wait.side_effect = observe
    pending = _PendingJobHandle()
    pending.attach(accepted)
    resources = MagicMock()
    state = _RunState(monitor_args=(client, "job-1", {"gpu": 1}, "token", resources, str(tmp_path), ctx))
    state.handoff.set()
    executor.run_processes["job-1"] = {
        RunProcessKey.JOB_HANDLE: pending,
        RunProcessKey.STATUS: ClientStatus.STARTED,
        _RUN_STATE_KEY: state,
    }
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    executor._retain_cleanup_owner("job-1")
    caplog.clear()
    with caplog.at_level(logging.INFO):
        try:
            executor._start_monitor("job-1", state)
            assert observed.wait(5)
            executor.abort_app("job-1")
        finally:
            release_wait.set()
            if state.monitor and state.monitor.ident:
                join_thread(state.monitor)
    assert "backend completion confirmed" in caplog.text
    accepted.terminate.assert_called_once()
    accepted.wait.assert_called_once()
    assert executor.get_run_processes_keys() == []
    assert find_job_cert(str(tmp_path / "job-1")) is None
    resources.free_resources.assert_called_once()
    client.send_request_before_shutdown.assert_called_once()


@pytest.mark.parametrize("raises", [False, True], ids=["ineffective-cancel", "cancel-error"])
def test_admin_retries_finished_cancel_while_original_observation_is_still_running(tmp_path, raises, cleanup_deadlines):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    cancelled_again = threading.Event()
    initial_done = threading.Event()
    admin_done = threading.Event()
    initial_errors = []
    resources = MagicMock()

    def terminate(adapter, attempt):
        if attempt == 1:
            if raises:
                raise RuntimeError("first cancellation failed")
            return  # A backend can return without having stopped its runtime.
        adapter.exited.set()
        cancelled_again.set()

    adapter = ControlledProcessAdapter(terminate)
    cleanup_deadlines.observation_pending = adapter.wait_entered
    adapter.release_wait.clear()  # Keep the first honest observer alive even after exit.
    accepted = ProcessHandle(process_adapter=adapter)
    launcher = MagicMock()

    def launch(_meta, context):
        publish_job_handle(accepted, context)
        raise RuntimeError("startup failed after acceptance")

    launcher.launch_job.side_effect = launch

    def initial_launch():
        try:
            executor.start_app(
                client,
                "job-1",
                meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                {"gpu": 1},
                "token",
                resources,
                ctx,
            )
        except Exception as error:
            initial_errors.append(error)
        finally:
            initial_done.set()

    def admin_abort():
        executor.abort_app("job-1")
        admin_done.set()

    monitor = None
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            with ThreadGroup(adapter.exited, adapter.release_wait) as threads:
                threads.start(initial_launch)
                assert adapter.wait_entered.wait(5)
                assert initial_done.wait(5)
                assert len(initial_errors) == 1 and isinstance(initial_errors[0], JobCleanupPending)
                pending = executor.run_processes["job-1"][RunProcessKey.JOB_HANDLE]
                monitor = executor.run_processes["job-1"][_RUN_STATE_KEY].monitor
                first_cancel = pending.cancel_call
                assert first_cancel.done.wait(5)
                assert monitor.is_alive()
                threads.start(admin_abort)
                assert cancelled_again.wait(5)
                assert admin_done.wait(5)
                assert adapter.terminate_count == 2
                assert adapter.wait_count == 1
                assert pending.cancel_call is not first_cancel
                assert executor.run_processes["job-1"][_RUN_STATE_KEY].monitor is monitor
                assert adapter.exited.is_set()
                assert find_job_cert(str(tmp_path / "job-1")) is not None
                resources.free_resources.assert_not_called()
        finally:
            if monitor is not None:
                join_thread(monitor)


def test_admin_retries_finished_observation_while_original_cancel_is_still_running(tmp_path, cleanup_deadlines):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    observed_again = threading.Event()
    initial_done = threading.Event()
    admin_done = threading.Event()
    initial_errors = []

    def terminate(adapter, _attempt):
        cancel_entered.set()
        release_cancel.wait()
        adapter.exited.set()

    def observe(_adapter, attempt):
        if attempt == 1:
            raise RuntimeError("first observation failed")
        observed_again.set()

    adapter = ControlledProcessAdapter(terminate, observe)
    cleanup_deadlines.cancel_pending = cancel_entered
    accepted = ProcessHandle(process_adapter=adapter)
    launcher = MagicMock()

    def launch(_meta, context):
        publish_job_handle(accepted, context)
        raise RuntimeError("startup failed after acceptance")

    launcher.launch_job.side_effect = launch

    def initial_launch():
        try:
            executor.start_app(
                client, "job-1", meta, SimpleNamespace(workspace=str(tmp_path), set=[]), None, None, None, ctx
            )
        except Exception as error:
            initial_errors.append(error)
        finally:
            initial_done.set()

    def abort():
        executor.abort_app("job-1")
        admin_done.set()

    monitor = None
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            with ThreadGroup(adapter.exited, adapter.release_wait, release_cancel) as threads:
                threads.start(initial_launch)
                assert cancel_entered.wait(5)
                assert initial_done.wait(5)
                assert len(initial_errors) == 1 and isinstance(initial_errors[0], JobCleanupPending)
                pending = executor.run_processes["job-1"][RunProcessKey.JOB_HANDLE]
                monitor = executor.run_processes["job-1"][_RUN_STATE_KEY].monitor
                first_cancel = pending.cancel_call
                assert not first_cancel.done.is_set()
                assert observed_again.wait(5)
                threads.start(abort)
                assert observed_again.wait(5)
                assert admin_done.wait(5)
                assert adapter.terminate_count == 1
                assert adapter.wait_count == 2
                assert pending.cancel_call is first_cancel
                assert executor.run_processes["job-1"][_RUN_STATE_KEY].monitor is monitor
        finally:
            if monitor is not None:
                join_thread(monitor)


def test_saved_abort_retries_transient_cancel_and_processor_frees_admission(tmp_path):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from nvflare.private.admin_defs import Message
    from nvflare.private.defs import TrainingTopic
    from nvflare.private.fed.client.scheduler_cmds import StartJobProcessor
    from nvflare.private.scheduler_constants import ShareableHeader
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    resources.allocate_resources.return_value = {"gpu": 1}
    launcher = MagicMock()
    handles = []

    def terminate(adapter, attempt):
        if attempt == 1:
            raise RuntimeError("transient saved abort failure")
        adapter.exited.set()

    adapter = ControlledProcessAdapter(terminate)
    accepted = ProcessHandle(process_adapter=adapter)

    def launch(_meta, context):
        executor.abort_app("job-1")  # Save the abort before publication.
        publish_job_handle(accepted, context)
        handles.append(executor.run_processes["job-1"][RunProcessKey.JOB_HANDLE])
        assert handles[-1].cancel_call.done.wait(5)  # Establish the completed transient failure.
        return accepted

    launcher.launch_job.side_effect = launch
    engine = MagicMock(spec=ClientEngine)
    engine.client_executor = executor
    engine.args = SimpleNamespace(workspace=str(tmp_path), set=[])
    engine.client = client
    engine.rank = 0
    engine.logger = MagicMock()
    engine.new_context.return_value = ctx
    engine.start_app.side_effect = lambda *args, **kwargs: ClientEngine.start_app(engine, *args, **kwargs)
    Path(workspace.get_app_dir("job-1")).mkdir()
    request = Message(topic=TrainingTopic.START_JOB, body={"gpu": 1})
    request.set_header(RequestHeader.JOB_ID, "job-1")
    request.set_header(RequestHeader.JOB_META, meta)
    request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, "token")
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch("nvflare.private.fed.client.scheduler_cmds._get_resource_manager", return_value=resources),
        patch("nvflare.private.fed.client.scheduler_cmds._get_resource_consumer"),
    ):
        try:
            reply = StartJobProcessor().process(request, engine)
            assert adapter.exited.is_set()
            assert adapter.terminate_count == 2
            assert adapter.wait_count == 1
            assert executor.get_run_processes_keys() == []
            resources.free_resources.assert_called_once()
            assert "transient saved abort failure" in reply.body
            assert "JobCleanupPending" not in reply.body
        finally:
            adapter.exited.set()
            adapter.release_wait.set()
            for handle in handles:
                join_thread(handle.cancel_call.thread)


def test_concurrent_retained_admin_abort_cannot_enter_cleanup_twice(tmp_path):
    _, workspace, client, _ = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    first_done = threading.Event()
    second_done = threading.Event()
    entries = []
    pending = _PendingJobHandle()
    pending.attach(MagicMock(spec=JobHandleSpec))
    executor.run_processes["job-1"] = {
        RunProcessKey.JOB_HANDLE: pending,
        RunProcessKey.STATUS: ClientStatus.STARTED,
    }
    executor._retain_cleanup_owner("job-1")

    def cleanup(*args, **kwargs):
        entries.append((args, kwargs))
        cleanup_entered.set()
        release_cleanup.wait()  # Keep the first admin request inside cleanup.
        return False

    def abort(done):
        executor.abort_app("job-1")
        done.set()

    with patch.object(executor, "_cleanup_failed_launch", side_effect=cleanup), ThreadGroup(release_cleanup) as threads:
        threads.start(abort, first_done)
        assert cleanup_entered.wait(5)
        threads.start(abort, second_done)
        assert second_done.wait(5)
        assert not first_done.is_set()
        assert len(entries) == 1
        assert client.cell.fire_and_forget.call_count == 1
        release_cleanup.set()
        assert first_done.wait(5)


def test_late_launch_cleanup_finalizes_without_admin_abort(tmp_path, cleanup_deadlines):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    adapter = ControlledProcessAdapter(lambda *_: None)  # Cancellation was ineffective.
    cleanup_deadlines.observation_pending = adapter.wait_entered
    handle = ProcessHandle(process_adapter=adapter)
    launcher = MagicMock()
    run_dir = tmp_path / "job-1"
    write_job_cert(str(run_dir), b"cert", b"key")
    checkpoint = run_dir / "checkpoint"
    checkpoint.write_text("preserved")

    def launch(_meta, context):
        publish_job_handle(handle, context)
        raise ValueError("launch failed after publication")

    launcher.launch_job.side_effect = launch
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            with pytest.raises(JobCleanupPending):
                executor.start_app(
                    client,
                    "job-1",
                    meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    {"gpu": 1},
                    "token",
                    resources,
                    ctx,
                )
            assert adapter.wait_entered.wait(5)
            state = executor.run_processes["job-1"][_RUN_STATE_KEY]
            assert state.phase == _RunPhase.CLEANUP_PENDING
            assert adapter.terminate_count == adapter.wait_count == 1
            resources.free_resources.assert_not_called()
            assert find_job_cert(str(run_dir)) is not None
            # A competing monitor invocation cannot become another release owner.
            executor._wait_child_process_finish(*state.monitor_args)
            assert adapter.wait_count == 1
        finally:
            adapter.exited.set()
            if "state" in locals():
                join_thread(state.monitor)
    assert state.phase == _RunPhase.RELEASED
    assert executor.get_run_processes_keys() == []
    resources.free_resources.assert_called_once()
    assert find_job_cert(str(run_dir)) is None
    assert checkpoint.read_text() == "preserved"
    client.send_request_before_shutdown.assert_called_once()
    assert sum(c.args[0] == EventType.JOB_COMPLETED for c in ctx.get_engine().fire_event.call_args_list) == 1


@pytest.mark.parametrize("launch_outcome", ["completed-cj", "lost-launch-reply", "shutdown-during-launch"])
@pytest.mark.parametrize("unsettled", ["coordinator", "worker"])
def test_composed_handle_retains_shared_ownership_until_physical_cleanup(
    tmp_path, cleanup_deadlines, launch_outcome, unsettled
):
    from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskExecutionStatus
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from nvflare.app_opt.job_launcher.slurm.task_service import _Job
    from nvflare.private.fed.client import task_runtime
    from nvflare.private.fed.client.task_runtime import JobHandleWithTaskWorkers
    from nvflare.private.fed.task_worker.task_service import _Attempt

    launch_failed = launch_outcome != "completed-cj"
    late_handle = launch_outcome == "shutdown-during-launch"
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    settled, pending = threading.Event(), threading.Event()
    native = Mock(spec=ProcessHandle)
    native.poll.return_value = JobReturnCode.SUCCESS
    native._is_cleanup_complete.side_effect = lambda: unsettled == "worker" or settled.is_set()
    owner = _Job(SimpleNamespace(poll_interval=0.01), "job-1", "site-1", "token", "site-1.job-1")
    owner.manage_cj = True
    if late_handle:
        owner.cj_launch_pending = True
    else:
        owner.attach_cj(native)
    owner.on_done = Mock()
    if unsettled == "worker":
        worker = Mock(execution_id="slurm:allocation-1")
        worker.poll.side_effect = lambda: TaskExecutionStatus(TaskExecutionPhase.TERMINAL, settled=settled.is_set())
        owner.attempts["attempt-1"] = _Attempt(None, handle=worker)
    wrapped = JobHandleWithTaskWorkers(native, owner, cleanup_timeout=0.01, allow_restart=False)
    wait = wrapped.wait

    def observe():
        try:
            return wait()
        except TimeoutError:
            pending.set()
            raise

    wrapped.wait = observe
    cleanup_deadlines.observation_pending = pending
    run_dir = tmp_path / "job-1"
    write_job_cert(str(run_dir), b"cert", b"key")
    launcher = MagicMock()

    def launch(_meta, context):
        if late_handle:
            engine = context.get_engine()
            engine._task_execution_state = task_runtime._TaskExecutionState()
            service = Mock(add_job=Mock(return_value=owner), shutdown=Mock(side_effect=lambda **_: owner.release()))
            binding = task_runtime.TaskExecutionBinding(
                owner.backend,
                Mock(get_or_create=Mock(return_value=service)),
                "OWNER",
                "TOKEN",
                meta,
                task_runtime.JobLaunchOptions(),
            )

            def create(_meta, launch_context):
                task_runtime.shutdown_task_execution(engine)
                publish_job_handle(native, launch_context)
                return native

            with patch.object(task_runtime, "JobHandleWithTaskWorkers", return_value=wrapped):
                return task_runtime._launch_job_with_task_workers(
                    binding, Mock(launch_job=create), context, cleanup_timeout=0.01
                )
        publish_job_handle(wrapped, context)
        if launch_failed:
            raise ValueError("start response lost")
        return wrapped

    launcher.launch_job.side_effect = launch
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        with ThreadGroup(settled) as threads:
            owner.thread = threads.start(owner.run)
            try:
                args = (client, "job-1", meta, SimpleNamespace(workspace=str(tmp_path), set=[]))
                if launch_failed:
                    with pytest.raises(JobCleanupPending):
                        executor.start_app(*args, {"slot": 1}, "reservation", resources, ctx)
                else:
                    executor.start_app(*args, {"slot": 1}, "reservation", resources, ctx)
                assert pending.wait(5), "cleanup must not report completion while the physical owner is alive"
                state = executor.run_processes["job-1"][_RUN_STATE_KEY]
                assert executor.get_run_processes_keys() == ["job-1"]
                assert executor.run_processes["job-1"][RunProcessKey.JOB_HANDLE]._job_handle is wrapped
                assert find_job_cert(str(run_dir)) is not None
                with pytest.raises(RuntimeError, match="still registered"):
                    executor.delete_run("job-1", str(run_dir))
                resources.free_resources.assert_not_called()
                client.send_request_before_shutdown.assert_not_called()
                assert owner.thread.is_alive()
            finally:
                settled.set()
                owner.release()
                with owner.lock:
                    owner.next_cj_observation = 0
                owner.wake.set()
                join_thread(owner.thread)
                if "state" in locals():
                    join_thread(state.monitor)
    assert executor.get_run_processes_keys() == []
    assert find_job_cert(str(run_dir)) is None
    resources.free_resources.assert_called_once()
    client.send_request_before_shutdown.assert_called_once()
    outcome = client.send_request_before_shutdown.call_args.kwargs["request"].payload
    assert outcome[JobOutcomeMsgKey.CODE] == (ProcessExitCode.INFRASTRUCTURE_ERROR if launch_failed else 0)
    assert sum(call.args[0] == EventType.JOB_COMPLETED for call in ctx.get_engine().fire_event.call_args_list) == 1


def test_monitor_retries_wait_on_the_same_thread_after_transient_query_errors(tmp_path):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    adapter_threads = []
    entered_retry = threading.Event()
    release_retry = threading.Event()
    resources = MagicMock()
    executor.job_cleanup_poll_interval = 17.0
    backoff_entered = [threading.Event(), threading.Event()]
    release_backoff = [threading.Event(), threading.Event()]
    backoffs = []

    class Backoff:
        def wait(self, timeout):
            index = len(backoffs)
            backoffs.append(timeout)
            assert index < 2, "observer did not stop retrying after success"
            backoff_entered[index].set()
            release_backoff[index].wait()

        def clear(self):
            pass

        def set(self):
            for release in release_backoff:
                release.set()

    def observe(_adapter, attempt):
        adapter_threads.append(threading.get_ident())
        if attempt <= 2:
            raise RuntimeError("transient observation failure")
        entered_retry.set()
        release_retry.wait()

    adapter = ControlledProcessAdapter(lambda *_: None, observe)
    launcher = MagicMock()

    def launch(*_args):
        executor.run_processes["job-1"][_RUN_STATE_KEY].wake = Backoff()
        return ProcessHandle(process_adapter=adapter)

    launcher.launch_job.side_effect = launch
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            executor.start_app(
                client,
                "job-1",
                meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                {"gpu": 1},
                "token",
                resources,
                ctx,
            )
            state = executor.run_processes["job-1"][_RUN_STATE_KEY]
            for index in range(2):
                assert backoff_entered[index].wait(5)
                assert adapter.wait_count == index + 1
                resources.free_resources.assert_not_called()
                release_backoff[index].set()
            assert entered_retry.wait(5)
            assert backoffs == [17.0, 17.0]
            assert adapter.wait_count == 3
            assert len(set(adapter_threads)) == 1
            assert adapter.terminate_count == 0  # An observation error doesn't request cancellation.
            assert executor.get_status("job-1") == ClientStatus.STARTING
            resources.free_resources.assert_not_called()
        finally:
            for release in release_backoff:
                release.set()
            release_retry.set()
            adapter.exited.set()
            if "state" in locals():
                join_thread(state.monitor)
    resources.free_resources.assert_called_once()
    assert executor.get_run_processes_keys() == []
    assert find_job_cert(str(tmp_path / "job-1")) is None


@pytest.mark.parametrize("release_error", [False, True])
def test_monitor_commits_removal_before_slow_release_and_fences_the_directory(tmp_path, caplog, release_error):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    adapter = ControlledProcessAdapter(lambda *_: None)
    launcher = MagicMock()
    launcher.launch_job.return_value = ProcessHandle(process_adapter=adapter)
    releasing = threading.Event()
    release_resources = threading.Event()
    resources = MagicMock()

    def free(**_kwargs):
        releasing.set()
        release_resources.wait()
        if release_error:
            raise RuntimeError("admission backend unavailable")

    resources.free_resources.side_effect = free
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            executor.start_app(
                client,
                "job-1",
                meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                {"gpu": 1},
                "token",
                resources,
                ctx,
            )
            state = executor.run_processes["job-1"][_RUN_STATE_KEY]
            assert adapter.wait_entered.wait(5)
            adapter.exited.set()
            assert releasing.wait(5)
            assert executor.get_run_processes_keys() == ["job-1"]
            assert "job-1" not in executor.run_processes
            assert executor.get_status("job-1") == ClientStatus.STOPPED
            with pytest.raises(RuntimeError, match="in progress"):
                executor.check_start_app("job-1")
            with pytest.raises(RuntimeError, match="in progress"):
                executor.delete_run("job-1", str(tmp_path / "job-1"))
            executor._wait_child_process_finish(*state.monitor_args)
            assert resources.free_resources.call_count == 1
        finally:
            adapter.exited.set()
            release_resources.set()
            if "state" in locals():
                join_thread(state.monitor)
    assert executor._deleting_runs == set()
    assert executor.check_start_app("job-1") == ClientStatus.STOPPED
    assert find_job_cert(str(tmp_path / "job-1")) is None
    client.send_request_before_shutdown.assert_called_once()
    if release_error:
        assert "admission release failed; operator reconciliation required" in caplog.text


def test_cleanup_pending_age_and_reminders_are_rate_limited(tmp_path, monkeypatch, caplog):
    import logging

    clock = isolate_time(monkeypatch, "nvflare.private.fed.client.client_executor")
    now = [100.0]
    monkeypatch.setattr(clock, "monotonic", lambda: now[0])
    executor = JobExecutor(MagicMock(), "startup")
    executor.run_processes["job-1"] = {RunProcessKey.JOB_HANDLE: _PendingJobHandle()}
    executor._retain_cleanup_owner("job-1")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        now[0] = 399.0
        assert executor.get_run_processes_keys() == ["job-1"]
        assert executor.check_status("job-1") == "cleanup pending (age 299s)"
        assert not caplog.records
        now[0] = 400.0
        for _ in range(5):
            executor.get_run_processes_keys()
        assert len(caplog.records) == 1
        assert "age 300s" in caplog.text
        now[0] = 700.0
        executor.get_run_processes_keys()
        assert len(caplog.records) == 2
        assert "age 600s" in caplog.text


def test_monitor_finishing_before_pending_error_reaches_processor_does_not_double_free(tmp_path, cleanup_deadlines):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from nvflare.private.admin_defs import Message
    from nvflare.private.defs import TrainingTopic
    from nvflare.private.fed.client.scheduler_cmds import StartJobProcessor
    from nvflare.private.scheduler_constants import ShareableHeader
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    resources.allocate_resources.return_value = {"gpu": 1}
    adapter = ControlledProcessAdapter(lambda *_: None)
    cleanup_deadlines.observation_pending = adapter.wait_entered
    launcher = MagicMock()
    retain = executor._retain_cleanup_owner
    monitors = []

    def launch(_meta, context):
        publish_job_handle(ProcessHandle(process_adapter=adapter), context)
        raise ValueError("startup failed")

    def complete_before_reply(job_id, expected_owner=None):
        retain(job_id, expected_owner)
        state = executor.run_processes[job_id][_RUN_STATE_KEY]
        monitors.append(state.monitor)
        adapter.exited.set()
        join_thread(state.monitor)  # Force release before JobCleanupPending is raised.

    launcher.launch_job.side_effect = launch
    engine = MagicMock(spec=ClientEngine)
    engine.client_executor = executor
    engine.args = SimpleNamespace(workspace=str(tmp_path), set=[])
    engine.client = client
    engine.rank = 0
    engine.logger = MagicMock()
    engine.new_context.return_value = ctx
    engine.start_app.side_effect = lambda *args, **kwargs: ClientEngine.start_app(engine, *args, **kwargs)
    Path(workspace.get_app_dir("job-1")).mkdir()
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    request = Message(topic=TrainingTopic.START_JOB, body={"gpu": 1})
    request.set_header(RequestHeader.JOB_ID, "job-1")
    request.set_header(RequestHeader.JOB_META, meta)
    request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, "token")
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch("nvflare.private.fed.client.scheduler_cmds._get_resource_manager", return_value=resources),
        patch("nvflare.private.fed.client.scheduler_cmds._get_resource_consumer"),
        patch.object(executor, "_retain_cleanup_owner", side_effect=complete_before_reply),
    ):
        try:
            reply = StartJobProcessor().process(request, engine)
            assert "JobCleanupPending" in reply.body
            resources.free_resources.assert_called_once()
            assert executor.get_run_processes_keys() == []
            assert find_job_cert(str(tmp_path / "job-1")) is None
            client.send_request_before_shutdown.assert_called_once()
        finally:
            adapter.exited.set()
            for monitor in monitors:
                join_thread(monitor)


def test_late_cleanup_cannot_mark_a_reused_job_id_pending(tmp_path, cleanup_deadlines):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    adapter = ControlledProcessAdapter(lambda *_: None)
    cleanup_deadlines.observation_pending = adapter.wait_entered
    resources = MagicMock()
    launcher = MagicMock()
    retain = executor._retain_cleanup_owner
    replacement_state = _RunState(phase=_RunPhase.MONITORING)
    replacement = {RunProcessKey.JOB_HANDLE: _PendingJobHandle(), _RUN_STATE_KEY: replacement_state}
    monitors = []

    def launch(_meta, context):
        publish_job_handle(ProcessHandle(process_adapter=adapter), context)
        raise ValueError("startup failed")

    def retain_after_reuse(job_id, expected_owner=None):
        state = executor.run_processes[job_id][_RUN_STATE_KEY]
        monitors.append(state.monitor)
        adapter.exited.set()
        join_thread(state.monitor)
        assert executor.get_run_processes_keys() == []
        with executor.lock:
            executor.run_processes[job_id] = replacement
        retain(job_id, expected_owner)

    launcher.launch_job.side_effect = launch
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(executor, "_retain_cleanup_owner", side_effect=retain_after_reuse),
    ):
        try:
            with pytest.raises(JobCleanupPending):
                executor.start_app(
                    client,
                    "job-1",
                    meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    {"gpu": 1},
                    "token",
                    resources,
                    ctx,
                )
            assert executor.run_processes["job-1"] is replacement
            assert replacement_state.phase == _RunPhase.MONITORING
            assert replacement_state.cleanup_since is None
            resources.free_resources.assert_called_once()
        finally:
            adapter.exited.set()
            for monitor in monitors:
                join_thread(monitor)


@pytest.mark.parametrize("report_error", [False, True], ids=["acknowledged", "report-error"])
def test_finalizing_job_stays_in_heartbeat_until_outcome_report_returns(tmp_path, report_error):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.poll.return_value = JobReturnCode.SUCCESS
    waiter_entered = threading.Event()
    process_exited = threading.Event()
    report_entered = threading.Event()
    release_report = threading.Event()
    resources = MagicMock()
    launcher = MagicMock()
    launcher.launch_job.return_value = accepted

    def observe():
        waiter_entered.set()
        process_exited.wait()

    def report(**kwargs):
        assert kwargs["request"].payload[JobOutcomeMsgKey.CODE] == JobReturnCode.SUCCESS
        report_entered.set()
        release_report.wait()
        if report_error:
            raise RuntimeError("outcome connection failed")
        reply = MagicMock()
        reply.get_header.return_value = ReturnCode.OK
        return reply

    accepted.wait.side_effect = observe
    client.send_request_before_shutdown.side_effect = report
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            executor.start_app(
                client,
                "job-1",
                meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                {"gpu": 1},
                "token",
                resources,
                ctx,
            )
            state = executor.run_processes["job-1"][_RUN_STATE_KEY]
            assert waiter_entered.wait(5)
            process_exited.set()
            assert report_entered.wait(5)
            # Release already committed, but the server still awaits this client's outcome.
            with executor.lock:
                assert "job-1" not in executor.run_processes
            assert executor.get_run_processes_keys() == ["job-1"]
            resources.free_resources.assert_called_once()
            assert find_job_cert(str(tmp_path / "job-1")) is None
            with pytest.raises(RuntimeError, match="in progress"):
                executor.check_start_app("job-1")
        finally:
            process_exited.set()
            release_report.set()
            if "state" in locals():
                join_thread(state.monitor)
    assert executor.get_run_processes_keys() == []
    resources.free_resources.assert_called_once()
    client.send_request_before_shutdown.assert_called_once()


def test_transient_observation_error_preserves_normal_cj_recovery(tmp_path, caplog, monkeypatch):
    observed = []
    clock = ManualClock()
    monkeypatch.setattr("nvflare.private.fed.client.client_executor.time", clock)
    replacement_reports = []

    def observe(executor, index, _handle):
        observed.append(index)
        if len(observed) == 1:
            raise RuntimeError("transient backend observation failure")
        if observed == [0, 0, 1]:
            clock.advance(301)
            replacement_reports.extend(executor.get_run_processes_keys())
            raise RuntimeError("first observation failure on replacement")

    launcher, client, _ = _run_recovery(tmp_path, [(True, 104), (True, 0)], on_wait=observe)
    assert observed == [0, 0, 1, 1]
    assert launcher.launch_job.call_count == 2
    assert replacement_reports == ["job-1"]
    assert "still pending (age" not in caplog.text
    assert "cannot observe CJ completion: RuntimeError: first observation failure on replacement" in caplog.text
    assert client.send_request_before_shutdown.call_args.kwargs["request"].payload[JobOutcomeMsgKey.CODE] == 0


@pytest.mark.parametrize("heartbeat_cleanup", [False, True], ids=["admin-abort", "heartbeat-abort"])
def test_transient_observation_error_preserves_abort_grace(tmp_path, monkeypatch, heartbeat_cleanup):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.poll.return_value = JobReturnCode.SUCCESS
    second_wait = threading.Event()
    release_wait = threading.Event()
    attempts = []
    resources = MagicMock()
    launcher = MagicMock()
    launcher.launch_job.return_value = accepted
    clock = isolate_time(monkeypatch, "nvflare.private.fed.client.client_executor")
    now = [0.0]
    clock.time = lambda: now[0]
    clock.sleep = lambda _: now.__setitem__(0, 10.1)
    events = []
    client.cell.fire_and_forget.side_effect = lambda **_: events.append(("abort", now[0]))

    def observe():
        attempts.append(threading.get_ident())
        if len(attempts) == 1:
            raise RuntimeError("transient backend observation failure")
        second_wait.set()
        release_wait.wait()

    def terminate():
        events.append(("stop", now[0]))
        release_wait.set()

    accepted.wait.side_effect = observe
    accepted.terminate.side_effect = terminate
    accepted._terminate_for_heartbeat_cleanup = MagicMock(side_effect=terminate)
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            executor.start_app(
                client,
                "job-1",
                meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                {"gpu": 1},
                "token",
                resources,
                ctx,
            )
            state = executor.run_processes["job-1"][_RUN_STATE_KEY]
            executor.notify_job_status("job-1", ClientStatus.STARTED)
            assert second_wait.wait(5)
            assert state.phase == _RunPhase.OBSERVATION_PENDING
            executor.abort_app("job-1", heartbeat_cleanup=heartbeat_cleanup)
            assert release_wait.wait(5)
            assert events == [("abort", 0.0), ("stop", 10.1)]
            assert accepted._terminate_for_heartbeat_cleanup.call_count == int(heartbeat_cleanup)
            assert accepted.terminate.call_count == int(not heartbeat_cleanup)
            assert len(set(attempts)) == 1
        finally:
            release_wait.set()
            if "state" in locals():
                join_thread(state.monitor)
    resources.free_resources.assert_called_once()
    assert executor.get_run_processes_keys() == []


@pytest.mark.parametrize("raises", [False, True], ids=["ineffective-stop", "stop-error"])
def test_heartbeat_retries_completed_cancel_after_transient_observation_error(tmp_path, cleanup_deadlines, raises):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    observer_alive = threading.Event()
    observed_on = []

    def observe(_adapter, attempt):
        observed_on.append(threading.get_ident())
        if attempt == 1:
            raise RuntimeError("transient observation failure")
        observer_alive.set()

    def terminate(adapter, attempt):
        if attempt < 3:
            if raises:
                raise RuntimeError("transient stop failure")
            return
        adapter.exited.set()

    adapter = ControlledProcessAdapter(terminate, observe)
    adapter.release_wait.clear()  # Keep the observer alive through all six heartbeats.
    cleanup_deadlines.observation_pending = observer_alive
    launcher = MagicMock()
    launcher.launch_job.return_value = ProcessHandle(process_adapter=adapter)
    engine = SimpleNamespace(client_executor=executor)
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            executor.start_app(
                client,
                "job-1",
                meta,
                SimpleNamespace(workspace=str(tmp_path), set=[]),
                {"gpu": 1},
                "token",
                resources,
                ctx,
            )
            state = executor.run_processes["job-1"][_RUN_STATE_KEY]
            assert observer_alive.wait(5)
            assert state.phase == _RunPhase.OBSERVATION_PENDING
            calls = []
            for _ in range(6):
                ClientEngine.abort_app(engine, "job-1", heartbeat_cleanup=True)
                if not adapter.exited.is_set():
                    pending = executor.run_processes["job-1"][RunProcessKey.JOB_HANDLE]
                    assert pending.cancel_call.done.wait(5)
                calls.append(adapter.terminate_count)
                if adapter.exited.is_set():
                    adapter.release_wait.set()
                    join_thread(state.monitor)
                else:
                    resources.free_resources.assert_not_called()
            assert calls == [1, 2, 3, 3, 3, 3]
            assert adapter.exited.is_set()
            assert len(set(observed_on)) == 1
            assert adapter.wait_count == 2  # Failed observation plus one live observer.
            resources.free_resources.assert_called_once()
        finally:
            adapter.exited.set()
            adapter.release_wait.set()
            if "state" in locals():
                join_thread(state.monitor)
    assert executor.get_run_processes_keys() == []
    resources.free_resources.assert_called_once()
    assert find_job_cert(str(tmp_path / "job-1")) is None
    client.send_request_before_shutdown.assert_called_once()


@pytest.mark.parametrize("late_cancel", [False, True], ids=["caller-rolls-back", "monitor-keeps-ownership"])
def test_monitor_start_failure_keeps_release_with_caller_until_handoff(tmp_path, cleanup_deadlines, late_cancel):
    from nvflare.private.admin_defs import Message
    from nvflare.private.defs import TrainingTopic
    from nvflare.private.fed.client.scheduler_cmds import StartJobProcessor
    from nvflare.private.scheduler_constants import ShareableHeader

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    resources.allocate_resources.return_value = {"gpu": 1}
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.poll.return_value = JobReturnCode.SUCCESS
    second_cancel_entered = threading.Event()
    finish_second_cancel = threading.Event()

    def terminate():
        if accepted.terminate.call_count == 2:
            second_cancel_entered.set()
            finish_second_cancel.wait()

    accepted.terminate.side_effect = terminate
    launcher = MagicMock()
    owners = []
    failed_start = []

    def launch(*_args):
        owners.append(executor.run_processes["job-1"])
        return accepted

    def start(thread):
        if getattr(thread._target, "__name__", "") == "_wait_child_process_finish" and not failed_start:
            failed_start.append(True)
            raise RuntimeError("monitor startup failed")
        _start_thread(thread)

    def before_decision():
        state = owners[0][_RUN_STATE_KEY]
        # Let any premature handoff run all the way through release before the
        # caller decides rollback. Correct code keeps the monitor behind the gate.
        if state.handoff.is_set():
            join_thread(state.monitor)
        if late_cancel:
            cleanup_deadlines.cancel_pending = second_cancel_entered
            ClientEngine.abort_app(SimpleNamespace(client_executor=executor), "job-1")
            assert second_cancel_entered.is_set()

    cleanup_deadlines.before_observation_result = before_decision
    launcher.launch_job.side_effect = launch
    engine = MagicMock(spec=ClientEngine)
    engine.client_executor = executor
    engine.args = SimpleNamespace(workspace=str(tmp_path), set=[])
    engine.client = client
    engine.rank = 0
    engine.logger = MagicMock()
    engine.new_context.return_value = ctx
    engine.start_app.side_effect = lambda *args, **kwargs: ClientEngine.start_app(engine, *args, **kwargs)
    Path(workspace.get_app_dir("job-1")).mkdir()
    request = Message(topic=TrainingTopic.START_JOB, body={"gpu": 1})
    request.set_header(RequestHeader.JOB_ID, "job-1")
    request.set_header(RequestHeader.JOB_META, meta)
    request.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, "token")
    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch("nvflare.private.fed.client.scheduler_cmds._get_resource_manager", return_value=resources),
        patch("nvflare.private.fed.client.scheduler_cmds._get_resource_consumer"),
        patch.object(threading.Thread, "start", start),
    ):
        try:
            reply = StartJobProcessor().process(request, engine)
            assert "monitor startup failed" in reply.body
            if late_cancel:
                assert "JobCleanupPending" in reply.body
                assert executor.get_run_processes_keys() == ["job-1"]
                resources.free_resources.assert_not_called()
            else:
                assert "JobCleanupPending" not in reply.body
        finally:
            finish_second_cancel.set()
            if owners and owners[0][_RUN_STATE_KEY].monitor:
                join_thread(owners[0][_RUN_STATE_KEY].monitor)
    resources.free_resources.assert_called_once()
    assert client.send_request_before_shutdown.call_count == int(late_cancel)
    assert executor.get_run_processes_keys() == []


@pytest.mark.parametrize("launch_failed", [False, True], ids=["normal-launch", "failed-launch"])
def test_monitor_rechecks_cancel_installed_after_completion_query(tmp_path, cleanup_deadlines, launch_failed):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    commit_progress = threading.Event()
    resources.free_resources.side_effect = lambda **_kwargs: commit_progress.set()
    query_entered = threading.Event()
    release_query = threading.Event()
    second_cancel = threading.Event()
    release_cancel = threading.Event()

    def terminate(_adapter, attempt):
        if not launch_failed or attempt == 2:
            second_cancel.set()
            release_cancel.wait()

    adapter = ControlledProcessAdapter(terminate)
    cleanup_deadlines.observation_pending = adapter.wait_entered

    def poll():
        query_entered.set()
        release_query.wait()
        return 0

    adapter.poll = poll
    launcher = MagicMock()

    def launch(_meta, context):
        handle = ProcessHandle(process_adapter=adapter)
        if launch_failed:
            publish_job_handle(handle, context)
            raise RuntimeError("launch failed after publication")
        return handle

    launcher.launch_job.side_effect = launch
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:

            def start_app():
                executor.start_app(
                    client,
                    "job-1",
                    meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    {"gpu": 1},
                    "token",
                    resources,
                    ctx,
                )

            if launch_failed:
                with pytest.raises(JobCleanupPending):
                    start_app()
            else:
                start_app()
                rc_file = tmp_path / "job-1" / FLMetaKey.PROCESS_RC_FILE
                rc_file.write_text(str(int(ProcessExitCode.UNSAFE_COMPONENT)))
            owner = executor.run_processes["job-1"]
            state = owner[_RUN_STATE_KEY]
            adapter.exited.set()
            assert query_entered.wait(5)
            cleanup_deadlines.cancel_pending = second_cancel
            executor.abort_app("job-1")
            assert second_cancel.wait(5)
            fresh_cancel = owner[RunProcessKey.JOB_HANDLE].cancel_call
            original_wait = fresh_cancel.done.wait

            def wait_for_cancel(timeout=None):
                if threading.current_thread() is state.monitor:
                    commit_progress.set()
                return original_wait(timeout)

            fresh_cancel.done.wait = wait_for_cancel
            release_query.set()
            # Valid code waits on the new cancel; removing the recheck instead
            # reaches free_resources. Both paths signal this same progress gate.
            assert commit_progress.wait(5)
            resources.free_resources.assert_not_called()
            assert executor.run_processes["job-1"] is owner
        finally:
            release_query.set()
            release_cancel.set()
            adapter.exited.set()
            if "state" in locals():
                join_thread(state.monitor)
    resources.free_resources.assert_called_once()
    assert executor.get_run_processes_keys() == []
    expected = JobReturnCode.ABORTED if launch_failed else ProcessExitCode.UNSAFE_COMPONENT
    client.send_request_before_shutdown.assert_called_once()
    assert client.send_request_before_shutdown.call_args.kwargs["request"].payload[JobOutcomeMsgKey.CODE] == expected


@pytest.mark.parametrize("aborted", [False, True], ids=["launch-failed", "admin-aborted"])
def test_failed_launch_outcome_is_mapped_after_late_completion(tmp_path, cleanup_deadlines, aborted):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()

    def terminate(adapter, attempt):
        if attempt > 1:
            adapter.exited.set()

    adapter = ControlledProcessAdapter(terminate)
    adapter.poll = lambda: 0 if adapter.exited.is_set() else None
    cleanup_deadlines.observation_pending = adapter.wait_entered
    launcher = MagicMock()

    def launch(_meta, context):
        publish_job_handle(ProcessHandle(process_adapter=adapter), context)
        raise RuntimeError("launch failed after publication")

    launcher.launch_job.side_effect = launch
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        try:
            with pytest.raises(JobCleanupPending):
                executor.start_app(
                    client,
                    "job-1",
                    meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    {"gpu": 1},
                    "token",
                    resources,
                    ctx,
                )
            process = executor.run_processes["job-1"]
            state = process[_RUN_STATE_KEY]
            write_terminal_handoff(
                tmp_path / "job-1", _participant_summary(), launch_generation=process[_RESTART_KEY].launch_id
            )
            client.send_request_before_shutdown.assert_not_called()
            if aborted:
                executor.abort_app("job-1")
            else:
                adapter.exited.set()
        finally:
            adapter.exited.set()
            if "state" in locals():
                join_thread(state.monitor)
    resources.free_resources.assert_called_once()
    assert find_job_cert(str(tmp_path / "job-1")) is None
    client.send_request_before_shutdown.assert_called_once()
    expected = JobReturnCode.ABORTED if aborted else ProcessExitCode.INFRASTRUCTURE_ERROR
    payload = client.send_request_before_shutdown.call_args.kwargs["request"].payload
    assert payload[JobOutcomeMsgKey.CODE] == expected
    summary = json.loads(payload[JobOutcomeMsgKey.RESOURCE_REPORT][JobOutcomeMsgKey.PARTICIPANT_SUMMARY])
    assert summary["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": 1.5}
    assert executor.get_run_processes_keys() == []


def test_heartbeat_retries_monitor_startup_while_reusing_inflight_cancel(tmp_path, cleanup_deadlines):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    startup_failed = threading.Event()
    monitor_attempts = []

    def terminate(adapter, _attempt):
        cancel_entered.set()
        release_cancel.wait()
        adapter.exited.set()

    adapter = ControlledProcessAdapter(terminate)
    cleanup_deadlines.cancel_pending = cancel_entered
    cleanup_deadlines.observation_pending = startup_failed
    launcher = MagicMock()
    launcher.launch_job.return_value = ProcessHandle(process_adapter=adapter)

    def start(thread):
        if getattr(thread._target, "__name__", "") == "_wait_child_process_finish":
            monitor_attempts.append(thread)
            if len(monitor_attempts) <= 2:
                startup_failed.set()
                raise RuntimeError("cannot start monitor")
        _start_thread(thread)

    with (
        patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher),
        patch.object(threading.Thread, "start", start),
    ):
        try:
            with pytest.raises(JobCleanupPending):
                executor.start_app(
                    client,
                    "job-1",
                    meta,
                    SimpleNamespace(workspace=str(tmp_path), set=[]),
                    {"gpu": 1},
                    "token",
                    resources,
                    ctx,
                )
            state = executor.run_processes["job-1"][_RUN_STATE_KEY]
            assert state.monitor is None
            assert adapter.terminate_count == 1
            # Simulate a heartbeat cleanup after the server has forgotten the job.
            ClientEngine.abort_app(SimpleNamespace(client_executor=executor), "job-1", heartbeat_cleanup=True)
            assert adapter.wait_entered.wait(5)
            assert len(monitor_attempts) == 3
            assert state.monitor is monitor_attempts[-1]
            assert adapter.terminate_count == 1
            assert adapter.wait_count == 1
            resources.free_resources.assert_not_called()
        finally:
            release_cancel.set()
            adapter.exited.set()
            if "state" in locals() and state.monitor and state.monitor.ident:
                join_thread(state.monitor)
    resources.free_resources.assert_called_once()
    assert executor.get_run_processes_keys() == []


@pytest.mark.parametrize("status", [ClientStatus.STARTING, ClientStatus.STARTED, ClientStatus.STOPPED])
def test_heartbeat_reuses_ordinary_abort_across_observation_error(tmp_path, cleanup_deadlines, monkeypatch, status):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    first_stop = threading.Event()
    finish_first_stop = threading.Event()
    ordinary_abort_returned = threading.Event()
    observing_again = threading.Event()
    cleanup_deadlines.cancel_pending = first_stop

    def terminate(adapter, attempt):
        if attempt == 1:
            first_stop.set()
            finish_first_stop.wait()
        adapter.exited.set()

    def observe(_adapter, attempt):
        if attempt == 1:
            first_stop.wait()
            raise RuntimeError("transient wait error while abort is running")
        observing_again.set()

    adapter = ControlledProcessAdapter(terminate, observe)
    runtime = ProcessHandle(process_adapter=adapter)
    launcher = MagicMock()
    launcher.launch_job.return_value = runtime
    clock = isolate_time(monkeypatch, "nvflare.private.fed.client.client_executor")
    clock.time = Mock(side_effect=[0.0, 11.0])  # Expire the normal teardown grace.
    engine = SimpleNamespace(client_executor=executor)

    def ordinary_abort():
        ClientEngine.abort_app(engine, "job-1")
        ordinary_abort_returned.set()

    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        executor.start_app(
            client,
            "job-1",
            meta,
            SimpleNamespace(workspace=str(tmp_path), set=[]),
            {"gpu": 1},
            "token",
            resources,
            ctx,
        )
        state = executor.run_processes["job-1"][_RUN_STATE_KEY]
        executor.notify_job_status("job-1", status)
        try:
            with ThreadGroup(first_stop, finish_first_stop, adapter.exited) as threads:
                threads.start(ordinary_abort)
                assert first_stop.wait(5)
                assert ordinary_abort_returned.wait(5), "ordinary abort waited past its stop-call budget"
                assert observing_again.wait(5)
                cleanup_deadlines.cancel_pending = None
                ClientEngine.abort_app(engine, "job-1", heartbeat_cleanup=True)
                assert adapter.terminate_count == 1
                assert not adapter.exited.is_set()
                resources.free_resources.assert_not_called()
                pending = executor.run_processes["job-1"][RunProcessKey.JOB_HANDLE]
                cancel = pending.cancel_call
                monitor_waiting = threading.Event()
                original_wait = cancel.done.wait

                def wait_for_stop(timeout=None):
                    if threading.current_thread() is state.monitor:
                        monitor_waiting.set()
                    return original_wait(timeout)

                with patch.object(cancel.done, "wait", side_effect=wait_for_stop):
                    adapter.exited.set()  # Independent exit while the abort call is still blocked.
                    assert monitor_waiting.wait(5)
                    resources.free_resources.assert_not_called()
                    assert executor.get_run_processes_keys() == ["job-1"]
        finally:
            join_thread(state.monitor)
    resources.free_resources.assert_called_once()
    assert executor.get_run_processes_keys() == []


@pytest.mark.parametrize("blocked", ["cancel", "observation"])
def test_pending_cleanup_returns_to_heartbeat_while_backend_is_blocked(tmp_path, cleanup_deadlines, blocked):
    from nvflare.app_common.job_launcher.process_launcher import ProcessHandle
    from tests.client_executor_test_utils import ControlledProcessAdapter

    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    resources = MagicMock()
    cancellation_entered = threading.Event()
    release_cancel = threading.Event()
    heartbeat_returned = threading.Event()

    def terminate(adapter, attempt):
        if attempt == 1:
            return  # The first stop returned without stopping the runtime.
        cancellation_entered.set()
        if blocked == "cancel":
            release_cancel.wait()
            adapter.exited.set()

    adapter = ControlledProcessAdapter(terminate)
    cleanup_deadlines.observation_pending = adapter.wait_entered
    launcher = MagicMock()

    def launch(_meta, fl_ctx):
        publish_job_handle(ProcessHandle(process_adapter=adapter), fl_ctx)
        raise RuntimeError("launch failed after publication")

    launcher.launch_job.side_effect = launch
    engine = SimpleNamespace(client_executor=executor)
    engine.abort_app = ClientEngine.abort_app.__get__(engine)
    communicator = Communicator(client_config={"client_name": "site-1"})

    def receive_heartbeat():
        communicator._clean_up_runs(engine, ["job-1"])
        heartbeat_returned.set()

    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        state = None
        try:
            with ThreadGroup(release_cancel, adapter.exited) as threads:
                with pytest.raises(JobCleanupPending):
                    executor.start_app(
                        client,
                        "job-1",
                        meta,
                        SimpleNamespace(workspace=str(tmp_path), set=[]),
                        {"gpu": 1},
                        "token",
                        resources,
                        ctx,
                    )
                state = executor.run_processes["job-1"][_RUN_STATE_KEY]
                # The first start bound expired using the gated test wait. Any
                # later wait would exceed the heartbeat's independent watchdog.
                executor.job_launch_cleanup_timeout = 30
                threads.start(receive_heartbeat)
                assert cancellation_entered.wait(5)
                assert heartbeat_returned.wait(5), "heartbeat waited for backend cleanup"
                assert not adapter.exited.is_set()
                assert executor.get_run_processes_keys() == ["job-1"]
                resources.free_resources.assert_not_called()
        finally:
            if state is not None:
                join_thread(state.monitor)
    resources.free_resources.assert_called_once()
