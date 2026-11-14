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
)
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobProcessArgs, JobReturnCode, publish_job_handle
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.core_cell import FQCN
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.private.defs import CellChannel, CellChannelTopic, JobOutcomeMsgKey, RequestHeader
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.client_executor import (
    _ABORT_REQUESTED_KEY,
    _CLEANUP_PENDING_KEY,
    _RESOURCE_OWNER_KEY,
    _RESTART_KEY,
    REPORTABLE_JOB_FAILURES,
    JobCleanupPending,
    JobExecutor,
    _PendingJobHandle,
)
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.client.communicator import Communicator
from nvflare.private.fed.utils.job_cert_utils import find_job_cert, write_job_cert
from tests.timing_utils import ThreadGroup, isolate_time

_start_thread = threading.Thread.start


def _skip_completion_monitor(thread):
    if getattr(thread._target, "__name__", "") != "_wait_child_process_finish":
        _start_thread(thread)


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


def test_abort_app_terminates_starting_job_without_worker_command():
    client = MagicMock()
    job_executor = JobExecutor(client=client, startup="startup")
    job_handle = MagicMock()
    job_executor.run_processes["job-1"] = {
        RunProcessKey.JOB_HANDLE: job_handle,
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
    job_handle = MagicMock()
    job_executor.run_processes["job-1"] = {
        RunProcessKey.JOB_HANDLE: job_handle,
        RunProcessKey.STATUS: ClientStatus.STOPPED,
    }

    with (
        patch("nvflare.private.fed.client.client_executor.time.time", side_effect=[10.0, 10.0, 20.1]),
        patch("nvflare.private.fed.client.client_executor.time.sleep") as sleep,
    ):
        job_executor._terminate_job(job_handle, "job-1", heartbeat_cleanup=True)

    sleep.assert_called_once_with(0.05)
    job_handle.terminate.assert_called_once_with(heartbeat_cleanup=True)


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


def test_start_app_removes_pending_handle_when_launch_fails(tmp_path):
    job_id = "job-1"
    job_meta, workspace, client, fl_ctx = _make_start_app_inputs(tmp_path, job_id)
    executor = JobExecutor(client=client, startup=workspace.get_startup_kit_dir())
    launcher = MagicMock()

    def launch_job(*_args):
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

    assert "job-1" not in job_executor.run_processes
    fl_ctx.set_prop.assert_any_call(FLContextKey.CURRENT_JOB_ID, "job-1", private=True, sticky=False)
    fl_ctx.set_prop.assert_any_call(FLContextKey.CLIENT_NAME, "site-1", private=True, sticky=False)
    engine.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, fl_ctx)


def test_wait_child_process_destroys_job_credential_when_worker_exits(tmp_path):
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.return_value.get_header.return_value = ReturnCode.OK
    job_executor = JobExecutor(client=client, startup="startup")
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


def test_wait_child_process_retains_job_credential_when_wait_raises(tmp_path):
    client = MagicMock()
    client.client_name = "site-1"
    job_executor = JobExecutor(client=client, startup="startup")
    job_handle = MagicMock()
    job_handle.wait.side_effect = RuntimeError("launcher lost the job")
    job_executor.run_processes = {"job-1": {RunProcessKey.JOB_HANDLE: job_handle}}
    run_dir = tmp_path / "job-1"
    run_dir.mkdir()
    write_job_cert(str(run_dir), b"cert", b"key")

    resources = MagicMock()
    job_executor._wait_child_process_finish(
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


def test_wait_child_process_destroys_job_credential_without_job_handle(tmp_path):
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

    assert find_job_cert(str(run_dir)) is None


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


def test_wait_child_process_cleans_up_when_terminal_outcome_report_fails():
    client = MagicMock()
    client.client_name = "site-1"
    client.send_request_before_shutdown.side_effect = RuntimeError("network unavailable")
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
    from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher
    from nvflare.app_opt.job_launcher.docker_launcher import ClientDockerJobLauncher
    from nvflare.app_opt.job_launcher.slurm.launcher import ServerSlurmJobLauncher
    from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy

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
    policy = TaskExecutionPolicy(SimpleNamespace(launch_mode="slurm" if backend == "slurm_server" else "k8s"))
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        with patch.object(TaskExecutionPolicy, "from_workspace", return_value=policy):
            with pytest.raises(ValueError, match="supported Kubernetes or Slurm client-job launcher"):
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
    from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher

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
    import importlib

    from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy

    launcher = object.__new__(getattr(importlib.import_module(module_name), class_name))
    launcher.launch_job = Mock(side_effect=RuntimeError("native launch boundary reached"))
    job_meta, workspace, client, context = _make_start_app_inputs(tmp_path)
    job_meta["resource_spec"] = {"site-1": {"num_of_gpus": gpus}}
    job_meta["cj_resource_spec"] = {"site-1": {}}
    Path(workspace.get_job_meta_path("job-1")).write_text(json.dumps(job_meta))
    config_dir = Path(workspace.get_app_config_dir("job-1"))
    config_dir.mkdir(parents=True)
    (config_dir / "config_fed_client.json").write_text('{"execution_lifetime":"task"}')
    policy = TaskExecutionPolicy(SimpleNamespace(launch_mode=mode))
    with patch("nvflare.private.fed.client.client_executor.get_job_launcher", return_value=launcher):
        with patch.object(TaskExecutionPolicy, "from_workspace", return_value=policy):
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
    launcher.launch_job.assert_called_once()


def test_shared_admission_keeps_job_site_independent_of_another_sites_task_policy(tmp_path):
    from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy

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
    from nvflare.private.fed.client.client_app_runner import ClientAppRunner
    from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator

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
        accepted.wait.side_effect = RuntimeError("observation unavailable")
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
        executor._wait_child_process_finish(client, "job-1", {"gpu": 1}, "token", resources, str(tmp_path), ctx)
    assert executor.get_run_processes_keys() == ["job-1"]
    assert find_job_cert(str(run_dir)) is not None
    resources.free_resources.assert_not_called()
    client.send_request_before_shutdown.assert_not_called()
    assert len(calls) == (2 if replacement else 1)
    if failure != "query":
        accepted.terminate.assert_called_once()
        accepted.wait.assert_called_once()  # Cancellation failure cannot skip observation.


@pytest.mark.parametrize("replacement", [False, True], ids=["initial", "replacement"])
@pytest.mark.parametrize("failure", ["raise", "none", "event", "saved-abort"])
def test_failed_launch_observes_published_handle_even_when_cancel_fails(tmp_path, replacement, failure):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    launcher = MagicMock()
    resources = MagicMock()
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.terminate.side_effect = RuntimeError("cancellation unavailable")
    accepted.wait.side_effect = RuntimeError("observation unavailable")
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
            executor._wait_child_process_finish(client, "job-1", {"gpu": 1}, "token", resources, str(tmp_path), ctx)
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
    assert process[_RESOURCE_OWNER_KEY] == ({"gpu": 1}, "token", resources)
    assert process[_CLEANUP_PENDING_KEY] is True
    executor.abort_app("job-1", heartbeat_cleanup=True)  # Retained owners must not stack control calls.
    assert ctx.get_prop(FLContextKey.JOB_HANDLE_CALLBACK) is None
    assert find_job_cert(str(tmp_path / "job-1")) is not None
    resources.free_resources.assert_not_called()
    client.send_request_before_shutdown.assert_not_called()
    accepted.terminate.assert_called_once()
    accepted.wait.assert_called_once()
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
def test_launch_cleanup_timeout_retains_owner_and_does_not_block_observation(tmp_path, blocked):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    executor.job_launch_cleanup_timeout = 0.02
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
                aborted = threading.Event()

                def abort_again():
                    executor.abort_app("job-1")
                    aborted.set()

                threads.start(abort_again)
                assert aborted.wait(2)
                accepted.terminate.assert_called_once()
                accepted.wait.assert_called_once()
        finally:
            release.set()
            pending = executor.run_processes.get("job-1", {}).get(RunProcessKey.JOB_HANDLE)
            if pending and pending.cleanup_calls:
                for call in pending.cleanup_calls:
                    call.thread.join(2)
                    assert not call.thread.is_alive()
    accepted.terminate.assert_called_once()
    accepted.wait.assert_called_once()
    # Late completion does not silently unregister or resubmit a retained run.
    assert executor.get_run_processes_keys() == ["job-1"]


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
        accepted.wait.side_effect = RuntimeError("query unavailable")
    launcher = MagicMock()
    launcher.launch_job.return_value = accepted

    def start(thread):
        if getattr(thread._target, "__name__", "") == "_wait_child_process_finish":
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
    accepted.wait.assert_called_once()


@pytest.mark.parametrize("failed_control", [1, 2], ids=["cancel-thread", "observe-thread"])
def test_cleanup_control_start_failure_cannot_skip_observation_or_drop_owner(tmp_path, failed_control):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    launcher = MagicMock()
    control_calls = []

    def launch(_meta, context):
        publish_job_handle(accepted, context)
        raise ValueError("startup failure")

    def start(thread):
        control_calls.append(thread)
        if len(control_calls) == failed_control:
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


def test_admin_abort_after_monitor_error_still_cancels_and_observes(tmp_path):
    meta, workspace, client, ctx = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.wait.side_effect = RuntimeError("query unavailable")
    pending = _PendingJobHandle()
    pending.attach(accepted)
    resources = MagicMock()
    owner = {
        RunProcessKey.JOB_HANDLE: pending,
        RunProcessKey.STATUS: ClientStatus.STARTED,
        _RESOURCE_OWNER_KEY: ({"gpu": 1}, "token", resources),
    }
    executor.run_processes["job-1"] = owner
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    executor._wait_child_process_finish(client, "job-1", {"gpu": 1}, "token", resources, str(tmp_path), ctx)
    assert executor.check_status("job-1") == "cleanup pending"
    assert executor.get_status("job-1") == ClientStatus.CLEANUP_PENDING
    accepted.terminate.assert_not_called()
    executor.abort_app("job-1")
    assert client.cell.fire_and_forget.call_args.kwargs["topic"] == AdminCommandNames.ABORT
    accepted.terminate.assert_called_once()
    assert accepted.wait.call_count == 2
    executor.abort_app("job-1")  # Completed failures may be retried by another explicit request.
    assert accepted.terminate.call_count == 2
    assert accepted.wait.call_count == 3
    assert executor.run_processes["job-1"] is owner
    assert find_job_cert(str(tmp_path / "job-1")) is not None
    resources.free_resources.assert_not_called()


def test_retained_heartbeat_is_quiet_and_does_not_issue_new_control_calls(tmp_path, caplog):
    _, workspace, client, _ = _make_start_app_inputs(tmp_path)
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    handle = MagicMock(spec=JobHandleSpec)
    executor.run_processes["job-1"] = {
        RunProcessKey.JOB_HANDLE: handle,
        RunProcessKey.STATUS: ClientStatus.STARTED,
    }
    executor._retain_cleanup_owner("job-1")
    caplog.clear()
    for _ in range(3):
        executor.abort_app("job-1", heartbeat_cleanup=True)
    handle.terminate.assert_not_called()
    client.cell.fire_and_forget.assert_not_called()
    assert not caplog.records


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
def test_retained_admin_abort_logs_observed_completion_without_releasing_owner(tmp_path, caplog, cancel_error):
    import logging

    _, workspace, client, _ = _make_start_app_inputs(tmp_path)
    client.cell.get_fqcn.return_value = "site-1"
    executor = JobExecutor(client, workspace.get_startup_kit_dir())
    accepted = MagicMock(spec=JobHandleSpec)
    accepted.terminate.side_effect = cancel_error
    pending = _PendingJobHandle()
    pending.attach(accepted)
    resources = MagicMock()
    owner = {
        RunProcessKey.JOB_HANDLE: pending,
        RunProcessKey.STATUS: ClientStatus.STARTED,
        _RESOURCE_OWNER_KEY: ({"gpu": 1}, "token", resources),
    }
    executor.run_processes["job-1"] = owner
    write_job_cert(str(tmp_path / "job-1"), b"cert", b"key")
    executor._retain_cleanup_owner("job-1")
    caplog.clear()
    with caplog.at_level(logging.INFO):
        executor.abort_app("job-1")
    assert "backend completion observed after admin abort" in caplog.text
    assert "registration, admission resources and credentials remain retained" in caplog.text
    accepted.terminate.assert_called_once()
    accepted.wait.assert_called_once()
    assert executor.run_processes["job-1"] is owner
    assert executor.get_status("job-1") == ClientStatus.CLEANUP_PENDING
    assert find_job_cert(str(tmp_path / "job-1")) is not None
    resources.free_resources.assert_not_called()
