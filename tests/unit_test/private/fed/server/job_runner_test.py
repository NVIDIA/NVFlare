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

from contextlib import nullcontext
from unittest.mock import MagicMock, call, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, RunProcessKey
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.fed.server.job_runner import JobRunner
from nvflare.private.fed.server.message_send import ClientReply


def _make_runner_inputs(num_clients=1):
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.log_warning = MagicMock()
    runner.fire_event = MagicMock()

    fl_ctx = MagicMock()
    engine = MagicMock()
    fl_ctx.get_engine.return_value = engine

    client_obj = MagicMock()
    client_obj.to_dict.return_value = {"name": "site-1"}
    engine.get_job_clients.return_value = {"token-1": client_obj}
    engine.start_app_on_server.return_value = ""
    engine.start_client_job.return_value = [MagicMock()]

    job = MagicMock()
    job.job_id = "job-1"
    job.meta = {}
    job.min_sites = 0  # no minimum by default
    job.required_sites = None  # no required sites by default

    client_sites = {"site-1": MagicMock()}
    return runner, fl_ctx, engine, job, client_sites


# ---------------------------------------------------------------------------
# strict flag wiring
# ---------------------------------------------------------------------------


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=False)
def test_start_run_passes_strict_false_when_flag_disabled(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = []  # no timeouts
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    mock_get_bool.assert_called_once()
    mock_check_replies.assert_called_once()
    assert mock_check_replies.call_args.kwargs["strict"] is False


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_passes_strict_true_when_flag_enabled(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = []  # no timeouts
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    mock_get_bool.assert_called_once()
    mock_check_replies.assert_called_once()
    assert mock_check_replies.call_args.kwargs["strict"] is True


# ---------------------------------------------------------------------------
# timeout exclusion in _start_run
# ---------------------------------------------------------------------------


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_proceeds_when_timed_out_clients_within_min_sites(mock_get_bool, mock_check_replies):
    """When some clients time out but active count >= min_sites, job proceeds with a warning."""
    mock_check_replies.return_value = ["site-2"]  # site-2 timed out
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()
    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1  # require at least 1; site-1 is still active

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    runner.log_warning.assert_called_once()
    warning_msg = runner.log_warning.call_args[0][1]
    assert "site-2" in warning_msg
    assert "timed out" in warning_msg


@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=False)
def test_start_run_non_strict_excludes_timed_out_clients_from_meta(mock_get_bool):
    """Even when strict checking is disabled, JOB_CLIENTS should include only active clients."""
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}

    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}

    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}

    ok_reply = Message(topic="reply", body="ok")
    ok_reply.set_header(MsgHeader.RETURN_CODE, ReturnCode.OK)
    req1 = Message(topic="req", body="")
    req2 = Message(topic="req", body="")
    engine.start_client_job.return_value = [
        ClientReply(client_token="token-site-1", client_name="site-1", req=req1, reply=ok_reply),
        ClientReply(client_token="token-site-2", client_name="site-2", req=req2, reply=None),
    ]

    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert job.meta[JobMetaKey.JOB_CLIENTS] == [{"name": "site-1"}]


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_raises_when_timed_out_clients_breach_min_sites(mock_get_bool, mock_check_replies):
    """When timeouts cause active count to fall below min_sites, _start_run raises."""
    mock_check_replies.return_value = ["site-1", "site-2"]  # both timed out
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()
    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 2  # need at least 2; 0 active after timeouts

    with pytest.raises(RuntimeError, match="min_sites"):
        runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_updates_job_clients_meta_after_timeout_exclusion(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = ["site-2"]
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}

    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}

    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}
    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert job.meta[JobMetaKey.JOB_CLIENTS] == [{"name": "site-1"}]


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_keeps_job_clients_meta_when_no_timeouts(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = []
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}

    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}

    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}
    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert job.meta[JobMetaKey.JOB_CLIENTS] == [{"name": "site-1"}, {"name": "site-2"}]


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_sets_job_clients_meta_before_start_client_job(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = []
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}

    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}

    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}

    seen_job_clients_meta = {}

    def _start_client_job_side_effect(passed_job, passed_client_sites, passed_fl_ctx):
        seen_job_clients_meta["value"] = passed_job.meta.get(JobMetaKey.JOB_CLIENTS)
        return [MagicMock()]

    engine.start_client_job.side_effect = _start_client_job_side_effect

    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert seen_job_clients_meta["value"] == [{"name": "site-1"}, {"name": "site-2"}]


def test_job_complete_process_saves_workspace_before_publishing_aborted_status():
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event = MagicMock()
    runner.log_debug = MagicMock()
    runner._save_workspace = MagicMock()
    runner.ask_to_stop = False

    engine = MagicMock()
    engine.run_processes = {}
    engine.exception_run_processes = {}

    job_manager = MagicMock()
    engine.get_component.return_value = job_manager

    completion_ctx = MagicMock()
    engine.new_context.return_value = nullcontext(completion_ctx)

    job = MagicMock()
    job.job_id = "job-1"
    job.run_aborted = True
    runner.running_jobs = {"job-1": job}

    parent = MagicMock()
    parent.attach_mock(runner._save_workspace, "save_workspace")
    parent.attach_mock(job_manager.set_status, "set_status")
    parent.attach_mock(runner.fire_event, "fire_event")

    def _stop_after_first_pass(_):
        runner.ask_to_stop = True

    with patch("nvflare.private.fed.server.job_runner.time.sleep", side_effect=_stop_after_first_pass):
        runner._job_complete_process(engine)

    completion_ctx.set_prop.assert_called_once_with(FLContextKey.CURRENT_JOB_ID, "job-1")
    assert parent.mock_calls == [
        call.save_workspace(completion_ctx),
        call.set_status("job-1", RunStatus.FINISHED_ABORTED, completion_ctx),
        call.fire_event(EventType.JOB_ABORTED, completion_ctx),
        call.fire_event(EventType.JOB_COMPLETED, completion_ctx),
    ]
    engine.remove_exception_process.assert_called_once_with("job-1")
    assert "job-1" not in runner.running_jobs


def test_get_finished_job_status_maps_aborted_launcher_return_code_to_finished_aborted():
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.abort_client_run = MagicMock()

    engine = MagicMock()
    engine.client_manager.clients = {}
    engine.exception_run_processes = {
        "job-1": {
            RunProcessKey.PARTICIPANTS: {},
            RunProcessKey.PROCESS_RETURN_CODE: JobReturnCode.ABORTED,
        }
    }

    job = MagicMock()
    job.job_id = "job-1"
    job_manager = MagicMock()
    fl_ctx = MagicMock()

    status = runner._get_finished_job_status(engine, job, fl_ctx)

    assert status == RunStatus.FINISHED_ABORTED
    job_manager.set_status.assert_not_called()
    runner.abort_client_run.assert_called_once_with("job-1", [], fl_ctx)


@pytest.mark.parametrize(
    "failure_code",
    [ProcessExitCode.CONFIG_ERROR, ProcessExitCode.EXCEPTION, JobReturnCode.EXECUTION_ERROR],
)
def test_get_finished_job_status_maps_exception_return_code_to_finished_execution_exception(failure_code):
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.abort_client_run = MagicMock()

    engine = MagicMock()
    engine.client_manager.clients = {}
    engine.exception_run_processes = {
        "job-1": {
            RunProcessKey.PARTICIPANTS: {},
            RunProcessKey.PROCESS_RETURN_CODE: failure_code,
        }
    }

    job = MagicMock()
    job.job_id = "job-1"
    job_manager = MagicMock()
    fl_ctx = MagicMock()

    status = runner._get_finished_job_status(engine, job, fl_ctx)

    assert status == RunStatus.FINISHED_EXECUTION_EXCEPTION
    job_manager.set_status.assert_not_called()
    runner.abort_client_run.assert_called_once_with("job-1", [], fl_ctx)


@pytest.mark.parametrize(
    "failure_code",
    [ProcessExitCode.CONFIG_ERROR, ProcessExitCode.EXCEPTION, JobReturnCode.EXECUTION_ERROR],
)
def test_get_finished_job_status_exception_return_code_overrides_clean_sj_finish(failure_code):
    """fail_run sets PROCESS_RETURN_CODE=EXCEPTION before _stop_run aborts the SJ.
    The SJ's finally clause then sends UPDATE_RUN_STATUS with execution_error=False
    and PROCESS_FINISHED=True. The EXCEPTION code must still win so list_jobs shows
    FINISHED:EXECUTION_EXCEPTION rather than FINISHED:COMPLETED.
    """
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.abort_client_run = MagicMock()

    engine = MagicMock()
    engine.client_manager.clients = {}
    engine.exception_run_processes = {
        "job-1": {
            RunProcessKey.PARTICIPANTS: {},
            RunProcessKey.PROCESS_RETURN_CODE: failure_code,
            RunProcessKey.PROCESS_FINISHED: True,
            RunProcessKey.PROCESS_EXE_ERROR: False,
        }
    }

    job = MagicMock()
    job.job_id = "job-1"
    job_manager = MagicMock()
    fl_ctx = MagicMock()

    status = runner._get_finished_job_status(engine, job, fl_ctx)

    assert status == RunStatus.FINISHED_EXECUTION_EXCEPTION
    job_manager.set_status.assert_not_called()


def test_fail_run_records_exception_process_without_setting_aborted_status():
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner._stop_run = MagicMock()

    engine = MagicMock()
    engine.lock = MagicMock()
    engine.run_processes = {"job-1": {RunProcessKey.PARTICIPANTS: {}}}
    engine.exception_run_processes = {}
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    job = MagicMock()
    job.job_id = "job-1"
    runner.running_jobs = {"job-1": job}

    assert runner.fail_run("job-1", ProcessExitCode.EXCEPTION, fl_ctx) == ""

    engine.lock.__enter__.assert_called_once()
    assert engine.exception_run_processes["job-1"][RunProcessKey.PROCESS_RETURN_CODE] == ProcessExitCode.EXCEPTION
    runner._stop_run.assert_called_once_with("job-1", fl_ctx)
    fl_ctx.set_prop.assert_called_once_with(FLContextKey.CURRENT_JOB_ID, "job-1")


def test_fail_run_preserves_existing_exception_process_entry_under_engine_lock():
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner._stop_run = MagicMock()

    live_run_process = {RunProcessKey.PARTICIPANTS: {"token-1": MagicMock()}}
    existing_exception_process = {
        RunProcessKey.PARTICIPANTS: {},
        RunProcessKey.PROCESS_RETURN_CODE: JobReturnCode.ABORTED,
    }
    engine = MagicMock()
    engine.lock = MagicMock()
    engine.run_processes = {"job-1": live_run_process}
    engine.exception_run_processes = {"job-1": existing_exception_process}
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine

    job = MagicMock()
    job.job_id = "job-1"
    runner.running_jobs = {"job-1": job}

    assert runner.fail_run("job-1", ProcessExitCode.EXCEPTION, fl_ctx) == ""

    engine.lock.__enter__.assert_called_once()
    assert engine.exception_run_processes["job-1"] is existing_exception_process
    assert existing_exception_process[RunProcessKey.PROCESS_RETURN_CODE] == ProcessExitCode.EXCEPTION
    assert live_run_process.get(RunProcessKey.PROCESS_RETURN_CODE) is None


def test_stop_run_does_not_publish_terminal_status_before_completion():
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner.fire_event = MagicMock()
    runner._stop_run = MagicMock()

    fl_ctx = MagicMock()

    job = MagicMock()
    job.job_id = "job-1"
    job.run_aborted = False
    runner.running_jobs = {"job-1": job}

    assert runner.stop_run("job-1", fl_ctx) == ""

    runner._stop_run.assert_called_once_with("job-1", fl_ctx)
    assert job.run_aborted is True
    fl_ctx.get_engine.assert_not_called()
    runner.fire_event.assert_not_called()


def test_job_complete_process_fires_job_aborted_for_aborted_launcher_return_code():
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event = MagicMock()
    runner.log_debug = MagicMock()
    runner.log_info = MagicMock()
    runner.abort_client_run = MagicMock()
    runner._save_workspace = MagicMock()
    runner.ask_to_stop = False

    engine = MagicMock()
    engine.run_processes = {}
    engine.client_manager.clients = {}
    engine.exception_run_processes = {
        "job-1": {
            RunProcessKey.PARTICIPANTS: {},
            RunProcessKey.PROCESS_RETURN_CODE: JobReturnCode.ABORTED,
        }
    }

    job_manager = MagicMock()
    engine.get_component.return_value = job_manager

    completion_ctx = MagicMock()
    engine.new_context.return_value = nullcontext(completion_ctx)

    job = MagicMock()
    job.job_id = "job-1"
    job.run_aborted = False
    runner.running_jobs = {"job-1": job}

    def _stop_after_first_pass(_):
        runner.ask_to_stop = True

    with patch("nvflare.private.fed.server.job_runner.time.sleep", side_effect=_stop_after_first_pass):
        runner._job_complete_process(engine)

    runner._save_workspace.assert_called_once_with(completion_ctx)
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_ABORTED, completion_ctx)
    assert runner.fire_event.call_args_list == [
        call(EventType.JOB_ABORTED, completion_ctx),
        call(EventType.JOB_COMPLETED, completion_ctx),
    ]
    engine.remove_exception_process.assert_called_once_with("job-1")
    assert "job-1" not in runner.running_jobs


def test_job_complete_process_keeps_processing_jobs_when_save_workspace_fails():
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event = MagicMock()
    runner.log_debug = MagicMock()
    runner.log_exception = MagicMock()
    runner._save_workspace = MagicMock(side_effect=[RuntimeError("storage unavailable"), None])
    runner.ask_to_stop = False

    engine = MagicMock()
    engine.run_processes = {}
    engine.exception_run_processes = {}

    job_manager = MagicMock()
    engine.get_component.return_value = job_manager

    first_ctx = MagicMock()
    second_ctx = MagicMock()
    engine.new_context.side_effect = [nullcontext(first_ctx), nullcontext(second_ctx)]

    first_job = MagicMock()
    first_job.job_id = "job-1"
    first_job.run_aborted = True
    second_job = MagicMock()
    second_job.job_id = "job-2"
    second_job.run_aborted = True
    runner.running_jobs = {"job-1": first_job, "job-2": second_job}

    def _stop_after_first_pass(_):
        runner.ask_to_stop = True

    with patch("nvflare.private.fed.server.job_runner.time.sleep", side_effect=_stop_after_first_pass):
        runner._job_complete_process(engine)

    assert runner._save_workspace.call_args_list == [call(first_ctx), call(second_ctx)]
    runner.log_exception.assert_called_once()
    job_manager.set_status.assert_called_once_with("job-2", RunStatus.FINISHED_ABORTED, second_ctx)
    assert runner.fire_event.call_args_list == [
        call(EventType.JOB_ABORTED, second_ctx),
        call(EventType.JOB_COMPLETED, second_ctx),
    ]
    engine.remove_exception_process.assert_called_once_with("job-2")
    assert "job-1" in runner.running_jobs
    assert "job-2" not in runner.running_jobs


def test_job_complete_process_retries_save_without_recomputing_finished_status():
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event = MagicMock()
    runner.log_debug = MagicMock()
    runner.log_info = MagicMock()
    runner.log_exception = MagicMock()
    runner.abort_client_run = MagicMock()
    runner._save_workspace = MagicMock(side_effect=[RuntimeError("storage unavailable"), None])
    runner.ask_to_stop = False

    engine = MagicMock()
    engine.run_processes = {}
    engine.client_manager.clients = {}
    engine.exception_run_processes = {
        "job-1": {
            RunProcessKey.PARTICIPANTS: {},
            RunProcessKey.PROCESS_RETURN_CODE: JobReturnCode.ABORTED,
        }
    }

    job_manager = MagicMock()
    engine.get_component.return_value = job_manager

    first_ctx = MagicMock()
    second_ctx = MagicMock()
    engine.new_context.side_effect = [nullcontext(first_ctx), nullcontext(second_ctx)]

    job = MagicMock()
    job.job_id = "job-1"
    job.run_aborted = False
    runner.running_jobs = {"job-1": job}

    sleep_count = 0

    def _stop_after_second_pass(_):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            runner.ask_to_stop = True

    with patch("nvflare.private.fed.server.job_runner.time.sleep", side_effect=_stop_after_second_pass):
        runner._job_complete_process(engine)

    assert runner._save_workspace.call_args_list == [call(first_ctx), call(second_ctx)]
    runner.abort_client_run.assert_called_once_with("job-1", [], first_ctx)
    runner.log_exception.assert_called_once()
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_ABORTED, second_ctx)
    assert runner.fire_event.call_args_list == [
        call(EventType.JOB_ABORTED, second_ctx),
        call(EventType.JOB_COMPLETED, second_ctx),
    ]
    engine.remove_exception_process.assert_called_once_with("job-1")
    assert "job-1" not in runner.running_jobs
    assert "job-1" not in runner._finished_job_states


def test_job_complete_process_retries_status_publish_without_resaving_workspace():
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event = MagicMock()
    runner.log_debug = MagicMock()
    runner.log_exception = MagicMock()
    runner._save_workspace = MagicMock()
    runner.ask_to_stop = False

    engine = MagicMock()
    engine.run_processes = {}
    engine.exception_run_processes = {}

    job_manager = MagicMock()
    job_manager.set_status.side_effect = [RuntimeError("storage unavailable"), None]
    engine.get_component.return_value = job_manager

    first_ctx = MagicMock()
    second_ctx = MagicMock()
    engine.new_context.side_effect = [nullcontext(first_ctx), nullcontext(second_ctx)]

    job = MagicMock()
    job.job_id = "job-1"
    job.run_aborted = True
    runner.running_jobs = {"job-1": job}

    sleep_count = 0

    def _stop_after_second_pass(_):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            runner.ask_to_stop = True

    with patch("nvflare.private.fed.server.job_runner.time.sleep", side_effect=_stop_after_second_pass):
        runner._job_complete_process(engine)

    runner._save_workspace.assert_called_once_with(first_ctx)
    assert job_manager.set_status.call_args_list == [
        call("job-1", RunStatus.FINISHED_ABORTED, first_ctx),
        call("job-1", RunStatus.FINISHED_ABORTED, second_ctx),
    ]
    runner.log_exception.assert_called_once()
    assert runner.fire_event.call_args_list == [
        call(EventType.JOB_ABORTED, second_ctx),
        call(EventType.JOB_COMPLETED, second_ctx),
    ]
    assert "job-1" not in runner.running_jobs
    assert "job-1" not in runner._finished_job_states


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_raises_when_required_site_times_out(mock_get_bool, mock_check_replies):
    """A timed-out required site must abort the job even if active_count >= min_sites."""
    mock_check_replies.return_value = ["site-2"]  # site-2 timed out
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}
    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}
    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}

    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1  # still satisfied after site-2 drops out
    job.required_sites = ["site-2"]  # but site-2 is required

    with pytest.raises(RuntimeError, match="required client site-2 timed out"):
        runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_proceeds_when_non_required_site_times_out(mock_get_bool, mock_check_replies):
    """A timed-out non-required site proceeds normally when min_sites is still satisfied."""
    mock_check_replies.return_value = ["site-2"]  # site-2 timed out but is not required
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}
    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}
    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}

    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1
    job.required_sites = ["site-1"]  # site-1 is required, site-2 is not

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert job.meta[JobMetaKey.JOB_CLIENTS] == [{"name": "site-1"}]
    runner.log_warning.assert_called_once()


@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_integration_real_reply_check_updates_meta(mock_get_bool):
    """Integration-style check: _start_run + real check_client_replies timeout path."""
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()

    site1 = MagicMock()
    site1.name = "site-1"
    site1.to_dict.return_value = {"name": "site-1"}

    site2 = MagicMock()
    site2.name = "site-2"
    site2.to_dict.return_value = {"name": "site-2"}

    engine.get_job_clients.return_value = {"token-1": site1, "token-2": site2}

    ok_reply = Message(topic="reply", body="ok")
    ok_reply.set_header(MsgHeader.RETURN_CODE, ReturnCode.OK)
    req1 = Message(topic="req", body="")
    req2 = Message(topic="req", body="")
    engine.start_client_job.return_value = [
        ClientReply(client_token="token-site-1", client_name="site-1", req=req1, reply=ok_reply),
        ClientReply(client_token="token-site-2", client_name="site-2", req=req2, reply=None),
    ]

    client_sites = {"site-1": MagicMock(), "site-2": MagicMock()}
    job.min_sites = 1

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    assert job.meta[JobMetaKey.JOB_CLIENTS] == [{"name": "site-1"}]
    runner.log_warning.assert_called_once()
    warning_msg = runner.log_warning.call_args[0][1]
    assert "site-2" in warning_msg
    assert "timed out" in warning_msg
