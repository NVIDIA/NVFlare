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
from contextlib import nullcontext
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ConfigVarName, FLContextKey, RunProcessKey, SystemConfigs
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.utils.event import fire_event_to_components
from nvflare.app_common.job_schedulers.job_scheduler import DefaultJobScheduler
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.zip_utils import get_all_file_paths
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.fed.resource_stats.contract import capacity_to_resource_time
from nvflare.private.fed.resource_stats.handoff import (
    INTERNAL_HANDOFF_KIND,
    INTERNAL_HANDOFF_VERSION,
    terminal_handoff_path,
    write_terminal_handoff,
)
from nvflare.private.fed.server.job_runner import JobRunner, _FinishedJobState
from nvflare.private.fed.server.message_send import ClientReply
from nvflare.private.fed.utils.job_cert_utils import write_job_cert


def _patch_job_runner_sleep(side_effect):
    """Patch the ``time`` name bound inside the job_runner module only.

    Patching ``job_runner.time.sleep`` would mutate the process-global ``time``
    module, so a leftover daemon thread from another test in the same worker
    calling ``time.sleep`` would run the side effect and trip the loop's stop
    condition early (FLARE-3034).
    """
    return patch(
        "nvflare.private.fed.server.job_runner.time",
        **{"sleep.side_effect": side_effect, "monotonic.return_value": 0.0},
    )


def _make_runner_inputs(num_clients=1):
    runner = JobRunner(workspace_root="/tmp")
    runner.resource_stats.start_job = MagicMock()
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
    job.meta = {JobMetaKey.JOB_NAME.value: "hello-pt"}
    job.min_sites = 0  # no minimum by default
    job.required_sites = None  # no required sites by default

    client_sites = {"site-1": MagicMock()}
    return runner, fl_ctx, engine, job, client_sites


def _make_workspace_save_inputs(run_dir, result_root, log_root, audit_root):
    runner = JobRunner(workspace_root="/tmp")
    runner.log_debug = MagicMock()
    runner.log_warning = MagicMock()

    workspace = MagicMock()
    workspace.get_run_dir.return_value = run_dir
    workspace.get_result_root.return_value = result_root
    workspace.get_log_root.return_value = log_root
    workspace.get_audit_root.return_value = audit_root

    job_manager = MagicMock()
    engine = MagicMock()
    engine.get_component.return_value = job_manager

    fl_ctx = MagicMock()
    fl_ctx.get_prop.return_value = "job-1"
    fl_ctx.get_workspace.return_value = workspace
    fl_ctx.get_engine.return_value = engine
    return runner, fl_ctx, job_manager


# ---------------------------------------------------------------------------
# strict flag wiring
# ---------------------------------------------------------------------------


def test_fire_job_lifecycle_event_includes_job_id():
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event_with_data = MagicMock()
    fl_ctx = MagicMock()

    runner._fire_job_lifecycle_event(EventType.JOB_STARTED, "job-1", fl_ctx)

    runner.fire_event_with_data.assert_called_once_with(
        EventType.JOB_STARTED, fl_ctx, FLContextKey.EVENT_DATA, {JobMetaKey.JOB_ID.value: "job-1"}
    )


def test_lifecycle_event_delivers_explicit_job_id_to_scheduler():
    scheduler = DefaultJobScheduler(max_jobs=5)
    runner = JobRunner(workspace_root="/tmp")

    engine = MagicMock()
    engine.fire_event.side_effect = lambda event_type, ctx: fire_event_to_components(event_type, [scheduler], ctx)
    fl_ctx = FLContextManager(engine=engine).new_context()
    fl_ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "wrong-job")  # racy sticky value

    runner._fire_job_lifecycle_event(EventType.JOB_STARTED, "job-1", fl_ctx)

    assert scheduler.scheduled_jobs == ["job-1"]
    assert fl_ctx.get_prop(FLContextKey.EVENT_DATA) is None  # cleaned up after the fire

    runner._fire_job_lifecycle_event(EventType.JOB_COMPLETED, "job-1", fl_ctx)

    assert scheduler.scheduled_jobs == []


@pytest.mark.parametrize("operation", ["mark_run_aborted", "fail_run"])
def test_job_terminal_operations_do_not_change_other_contexts(operation):
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner._stop_run = MagicMock()
    engine = MagicMock()
    engine.exception_run_processes = {}
    engine.run_processes = {}
    manager = FLContextManager(engine=engine)
    contexts = []
    for job_id in ("job-A", "job-B"):
        ctx = manager.new_context()
        job = MagicMock(job_id=job_id)
        runner.running_jobs[job_id] = job
        if operation == "fail_run":
            assert runner.fail_run(job_id, ProcessExitCode.EXCEPTION, ctx) == ""
        else:
            assert runner.mark_run_aborted(job_id, ctx) == ""
        contexts.append(ctx)

    assert [ctx.get_prop(FLContextKey.CURRENT_JOB_ID) for ctx in contexts] == ["job-A", "job-B"]
    assert manager.new_context().get_prop(FLContextKey.CURRENT_JOB_ID) is None


@patch("nvflare.private.fed.server.job_runner.check_client_replies")
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=False)
def test_start_run_passes_strict_false_when_flag_disabled(mock_get_bool, mock_check_replies):
    mock_check_replies.return_value = []  # no timeouts
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()
    runner._fire_job_lifecycle_event = MagicMock()

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    mock_get_bool.assert_called_once()
    mock_check_replies.assert_called_once()
    assert mock_check_replies.call_args.kwargs["strict"] is False
    runner._fire_job_lifecycle_event.assert_called_once_with(EventType.JOB_STARTED, "job-1", fl_ctx)


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


@pytest.mark.parametrize(
    ("section", "file_name"),
    [(SystemConfigs.STARTUP_CONF, "fed_server.json"), (SystemConfigs.RESOURCES_CONF, "resources.json")],
)
def test_job_cert_valid_days_read_from_server_config(tmp_path, section, file_name):
    (tmp_path / file_name).write_text(json.dumps({ConfigVarName.JOB_CERT_VALID_DAYS: 7}))
    ConfigService.reset()
    ConfigService.initialize(section_files={section: file_name}, config_path=[str(tmp_path)])
    try:
        assert JobRunner(workspace_root=str(tmp_path)).job_cert_valid_days == 7
    finally:
        ConfigService.reset()


def test_save_workspace_skips_duplicate_missing_sources(tmp_path):
    missing = str(tmp_path / "missing")
    runner, fl_ctx, job_manager = _make_workspace_save_inputs(missing, missing, missing, missing)

    runner._save_workspace(fl_ctx)

    job_manager.save_workspace.assert_not_called()
    assert runner.log_warning.call_args_list == [
        call(fl_ctx, f"Skipping unavailable workspace archive source for job job-1: {missing}"),
        call(fl_ctx, "No workspace archive sources are available for finished job job-1"),
    ]


def test_save_workspace_archives_only_existing_deduplicated_sources(tmp_path):
    run_dir = tmp_path / "run"
    result_root = tmp_path / "result"
    missing = str(tmp_path / "missing")
    run_dir.mkdir()
    result_root.mkdir()
    runner, fl_ctx, job_manager = _make_workspace_save_inputs(str(run_dir), str(run_dir), str(result_root), missing)
    job_manager.save_workspace.return_value = "/store/job-1/workspace"

    runner._save_workspace(fl_ctx)

    job_manager.save_workspace.assert_called_once_with("job-1", [str(run_dir), str(result_root)], fl_ctx)
    runner.log_warning.assert_called_once_with(
        fl_ctx, f"Skipping unavailable workspace archive source for job job-1: {missing}"
    )
    assert not run_dir.exists()
    assert not result_root.exists()


def test_save_workspace_destroys_job_credential_before_archiving(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "app_server").mkdir()
    (run_dir / "app_server" / "result.txt").write_text("2")
    write_job_cert(str(run_dir), b"cert", b"key")
    runner, fl_ctx, job_manager = _make_workspace_save_inputs(str(run_dir), str(run_dir), str(run_dir), str(run_dir))
    archived = {}

    def _capture(job_id, sources, ctx):
        archived[job_id] = sorted(
            os.path.relpath(f, sources[0]) for f in get_all_file_paths(sources[0]) if os.path.isfile(f)
        )
        return "/store/job-1/workspace"

    job_manager.save_workspace.side_effect = _capture

    runner._save_workspace(fl_ctx)

    assert archived["job-1"] == [os.path.join("app_server", "result.txt")]
    assert not run_dir.exists()


def test_save_workspace_tolerates_source_disappearing_during_cleanup(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner, fl_ctx, job_manager = _make_workspace_save_inputs(*([str(run_dir)] * 4))

    def _archive_then_remove_source(*_args):
        run_dir.rmdir()
        return "/store/job-1/workspace"

    job_manager.save_workspace.side_effect = _archive_then_remove_source

    runner._save_workspace(fl_ctx)

    job_manager.save_workspace.assert_called_once_with("job-1", [str(run_dir)], fl_ctx)
    runner.log_warning.assert_called_once_with(
        fl_ctx, f"Workspace archive source disappeared before cleanup for job job-1: {run_dir}"
    )


def test_save_workspace_retries_cleanup_without_rewriting_archive(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner, fl_ctx, job_manager = _make_workspace_save_inputs(*([str(run_dir)] * 4))
    finished_state = _FinishedJobState(status=RunStatus.FINISHED_COMPLETED)

    cleanup_attempts = 0

    def _fail_cleanup_once(_path):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise PermissionError("denied")
        run_dir.rmdir()

    with patch("nvflare.private.fed.server.job_runner.shutil.rmtree", side_effect=_fail_cleanup_once):
        with pytest.raises(PermissionError, match="denied"):
            runner._save_workspace(fl_ctx, finished_state)
        runner._save_workspace(fl_ctx, finished_state)

    job_manager.save_workspace.assert_called_once_with("job-1", [str(run_dir)], fl_ctx)
    assert finished_state.workspace_archive_written is True
    assert finished_state.workspace_archive_sources == (str(run_dir),)
    assert cleanup_attempts == 2
    assert not run_dir.exists()


def test_save_workspace_retries_remaining_cleanup_after_earlier_source_was_removed(tmp_path):
    run_dir = tmp_path / "run"
    result_root = tmp_path / "result"
    run_dir.mkdir()
    result_root.mkdir()
    runner, fl_ctx, job_manager = _make_workspace_save_inputs(
        str(run_dir), str(run_dir), str(result_root), str(result_root)
    )
    finished_state = _FinishedJobState(status=RunStatus.FINISHED_COMPLETED)

    cleanup_calls = []
    result_cleanup_attempts = 0

    def _fail_second_source_once(path):
        nonlocal result_cleanup_attempts
        cleanup_calls.append(path)
        if path == str(run_dir):
            if not run_dir.exists():
                raise FileNotFoundError(path)
            run_dir.rmdir()
            return

        result_cleanup_attempts += 1
        if result_cleanup_attempts == 1:
            raise PermissionError("denied")
        result_root.rmdir()

    with patch("nvflare.private.fed.server.job_runner.shutil.rmtree", side_effect=_fail_second_source_once):
        with pytest.raises(PermissionError, match="denied"):
            runner._save_workspace(fl_ctx, finished_state)
        runner._save_workspace(fl_ctx, finished_state)

    job_manager.save_workspace.assert_called_once_with("job-1", [str(run_dir), str(result_root)], fl_ctx)
    assert finished_state.workspace_archive_sources == (str(run_dir), str(result_root))
    assert cleanup_calls == [str(run_dir), str(result_root), str(run_dir), str(result_root)]
    runner.log_warning.assert_called_once_with(
        fl_ctx, f"Workspace archive source disappeared before cleanup for job job-1: {run_dir}"
    )
    assert not run_dir.exists()
    assert not result_root.exists()


def test_save_workspace_uses_explicit_job_id_over_sticky_context_id(tmp_path):
    job_a_dir = tmp_path / "job-A"
    job_a_dir.mkdir()
    job_b_dir = tmp_path / "job-B"
    job_b_dir.mkdir()

    runner, fl_ctx, job_manager = _make_workspace_save_inputs("", "", "", "")
    fl_ctx.get_prop.return_value = "job-B"  # racy sticky CURRENT_JOB_ID points at another job
    workspace = fl_ctx.get_workspace.return_value
    for getter in (workspace.get_run_dir, workspace.get_result_root, workspace.get_log_root, workspace.get_audit_root):
        getter.side_effect = lambda job_id: str(tmp_path / job_id)

    runner._save_workspace(fl_ctx, job_id="job-A")

    job_manager.save_workspace.assert_called_once_with("job-A", [str(job_a_dir)], fl_ctx)
    assert not job_a_dir.exists()
    assert job_b_dir.exists()


def test_job_complete_process_retries_cleanup_without_resaving_workspace(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner, completion_ctx, job_manager = _make_workspace_save_inputs(*([str(run_dir)] * 4))
    runner.fire_event = MagicMock()
    runner.log_exception = MagicMock()
    runner.ask_to_stop = False
    job_manager.save_workspace.return_value = "/store/job-1/workspace"

    engine = completion_ctx.get_engine.return_value
    engine.run_processes = {}
    engine.exception_run_processes = {}
    engine.new_context.side_effect = [nullcontext(completion_ctx), nullcontext(completion_ctx)]

    job = MagicMock()
    job.job_id = "job-1"
    job.run_aborted = True
    runner.running_jobs = {"job-1": job}

    cleanup_attempts = 0

    def _fail_cleanup_once(_path):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise PermissionError("denied")
        run_dir.rmdir()

    sleep_count = 0

    def _stop_after_second_pass(_):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            runner.ask_to_stop = True

    with (
        patch("nvflare.private.fed.server.job_runner.shutil.rmtree", side_effect=_fail_cleanup_once),
        _patch_job_runner_sleep(_stop_after_second_pass),
    ):
        runner._job_complete_process(engine)

    job_manager.save_workspace.assert_called_once_with("job-1", [str(run_dir)], completion_ctx)
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_ABORTED, completion_ctx)
    runner.log_exception.assert_called_once()
    assert cleanup_attempts == 2
    assert not run_dir.exists()


def test_job_complete_process_publishes_status_after_persistent_cleanup_failure(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner, completion_ctx, job_manager = _make_workspace_save_inputs(*([str(run_dir)] * 4))
    runner.fire_event = MagicMock()
    runner.log_error = MagicMock()
    runner.log_exception = MagicMock()
    runner.ask_to_stop = False
    job_manager.save_workspace.return_value = "/store/job-1/workspace"

    engine = completion_ctx.get_engine.return_value
    engine.run_processes = {}
    engine.exception_run_processes = {}
    engine.new_context.side_effect = [nullcontext(completion_ctx), nullcontext(completion_ctx)]

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

    with (
        patch("nvflare.private.fed.server.job_runner.shutil.rmtree", side_effect=PermissionError("denied")) as rmtree,
        patch(
            "nvflare.private.fed.server.job_runner.time",
            **{"sleep.side_effect": _stop_after_second_pass, "monotonic.side_effect": [0.0, 60.0]},
        ),
    ):
        runner._job_complete_process(engine)

    job_manager.save_workspace.assert_called_once_with("job-1", [str(run_dir)], completion_ctx)
    assert rmtree.call_args_list == [call(str(run_dir)), call(str(run_dir))]
    runner.log_exception.assert_called_once()
    runner.log_error.assert_called_once_with(
        completion_ctx,
        "Workspace cleanup for finished job (job-1) kept failing for 60 seconds; publishing terminal status with "
        "archived artifacts: PermissionError: denied",
    )
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_ABORTED, completion_ctx)
    assert runner.fire_event.call_args_list == [
        call(EventType.JOB_ABORTED, completion_ctx),
        call(EventType.JOB_COMPLETED, completion_ctx),
    ]
    assert "job-1" not in runner.running_jobs
    assert "job-1" not in runner._finished_job_states
    assert run_dir.exists()


def test_job_complete_process_saves_workspace_before_publishing_aborted_status():
    runner = JobRunner(workspace_root="/tmp")
    runner._fire_job_lifecycle_event = MagicMock()
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
    parent.attach_mock(runner._fire_job_lifecycle_event, "fire_job_lifecycle_event")

    def _stop_after_first_pass(_):
        runner.ask_to_stop = True

    with _patch_job_runner_sleep(_stop_after_first_pass):
        runner._job_complete_process(engine)

    completion_ctx.set_prop.assert_called_once_with(FLContextKey.CURRENT_JOB_ID, "job-1", sticky=False)
    assert parent.mock_calls == [
        call.save_workspace(completion_ctx, ANY, "job-1"),
        call.set_status("job-1", RunStatus.FINISHED_ABORTED, completion_ctx),
        call.fire_job_lifecycle_event(EventType.JOB_ABORTED, "job-1", completion_ctx),
        call.fire_job_lifecycle_event(EventType.JOB_COMPLETED, "job-1", completion_ctx),
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
    "failure_code, expected_status",
    [
        (ProcessExitCode.CONFIG_ERROR, RunStatus.FINISHED_EXECUTION_EXCEPTION),
        (ProcessExitCode.EXCEPTION, RunStatus.FINISHED_EXECUTION_EXCEPTION),
        (JobReturnCode.EXECUTION_ERROR, RunStatus.FINISHED_EXECUTION_EXCEPTION),
        (ProcessExitCode.INFRASTRUCTURE_ERROR, RunStatus.FINISHED_ABNORMAL),
    ],
)
def test_get_finished_job_status_maps_failure_return_code(failure_code, expected_status):
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

    assert status == expected_status
    job_manager.set_status.assert_not_called()
    runner.abort_client_run.assert_called_once_with("job-1", [], fl_ctx)


@pytest.mark.parametrize(
    "failure_code",
    [
        ProcessExitCode.CONFIG_ERROR,
        ProcessExitCode.EXCEPTION,
        ProcessExitCode.UNSAFE_COMPONENT,
        JobReturnCode.EXECUTION_ERROR,
    ],
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
    fl_ctx.set_prop.assert_called_once_with(FLContextKey.CURRENT_JOB_ID, "job-1", sticky=False)


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


@pytest.mark.parametrize(
    "first_code, second_code, expected_code",
    [
        (ProcessExitCode.INFRASTRUCTURE_ERROR, ProcessExitCode.EXCEPTION, ProcessExitCode.INFRASTRUCTURE_ERROR),
        (ProcessExitCode.EXCEPTION, ProcessExitCode.INFRASTRUCTURE_ERROR, ProcessExitCode.INFRASTRUCTURE_ERROR),
        (ProcessExitCode.EXCEPTION, JobReturnCode.ABORTED, ProcessExitCode.EXCEPTION),
        (JobReturnCode.ABORTED, ProcessExitCode.EXCEPTION, ProcessExitCode.EXCEPTION),
    ],
)
def test_fail_run_preserves_failure_precedence(first_code, second_code, expected_code):
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner._stop_run = MagicMock()

    run_process = {
        RunProcessKey.PARTICIPANTS: {},
        RunProcessKey.PROCESS_RETURN_CODE: first_code,
    }
    engine = MagicMock()
    engine.lock = MagicMock()
    engine.exception_run_processes = {"job-1": run_process}
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    runner.running_jobs = {"job-1": MagicMock()}

    assert runner.fail_run("job-1", second_code, fl_ctx) == ""

    assert run_process[RunProcessKey.PROCESS_RETURN_CODE] == expected_code


def test_client_outcome_tracking_api():
    runner = JobRunner(workspace_root="/tmp")
    runner._pending_client_outcomes = {"job-1": {"site-1", "site-2"}, "job-2": set()}

    assert runner.get_client_outcome_jobs() == {"job-1", "job-2"}
    assert runner.get_client_outcome_jobs("site-1") == {"job-1"}
    assert runner.is_client_outcome_pending("job-1", "site-1")
    runner.resolve_client_outcome("job-1", "site-1")
    assert not runner.is_client_outcome_pending("job-1", "site-1")


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


def test_stop_all_runs_logs_info_for_untracked_run_process():
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner._stop_run = MagicMock()

    engine = MagicMock()
    engine.run_processes = {"job-1": {}}
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    runner.running_jobs = {}

    runner.stop_all_runs(fl_ctx)

    runner._stop_run.assert_called_once_with("job-1", fl_ctx)
    assert runner.log_info.call_args_list == [
        call(fl_ctx, "Job job-1 is not running. It can not be stopped."),
        call(fl_ctx, "Stop all the running jobs."),
    ]
    assert runner.ask_to_stop is True


def test_mark_run_aborted_logs_info_when_job_not_running():
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    fl_ctx = MagicMock()
    runner.running_jobs = {}

    msg = runner.mark_run_aborted("job-1", fl_ctx)

    assert msg == "Job job-1 is not running."
    runner.log_info.assert_called_once_with(fl_ctx, "Job job-1 is not running. It can not be stopped.")


def test_fail_run_ignores_job_already_finalized():
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner._stop_run = MagicMock()

    engine = MagicMock()
    engine.lock = MagicMock()
    engine.run_processes = {}
    engine.exception_run_processes = {}
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    runner.running_jobs = {}

    msg = runner.fail_run("job-1", ProcessExitCode.EXCEPTION, fl_ctx)

    assert msg == "Job job-1 is not running."
    runner.log_info.assert_called_once_with(fl_ctx, "Job job-1 is not running. It can not be failed.")
    runner._stop_run.assert_not_called()
    assert "job-1" not in engine.exception_run_processes


def test_fail_run_releases_client_outcome_barrier():
    runner = JobRunner(workspace_root="/tmp")
    runner.log_info = MagicMock()
    runner._stop_run = MagicMock()
    runner._pending_client_outcomes = {"job-1": {"site-1", "site-2"}}
    runner._client_outcome_deadlines = {"job-1": 123.0}

    engine = MagicMock()
    engine.lock = MagicMock()
    engine.run_processes = {
        "job-1": {
            RunProcessKey.PARTICIPANTS: {},
        }
    }
    engine.exception_run_processes = {}
    fl_ctx = MagicMock()
    fl_ctx.get_engine.return_value = engine
    job = MagicMock(job_id="job-1")
    runner.running_jobs = {"job-1": job}

    assert runner.fail_run("job-1", ProcessExitCode.EXCEPTION, fl_ctx) == ""

    assert "job-1" not in runner._pending_client_outcomes
    assert "job-1" not in runner._client_outcome_deadlines
    runner._stop_run.assert_called_once_with("job-1", fl_ctx)


@pytest.mark.parametrize(
    "failure_process",
    [
        {
            RunProcessKey.PARTICIPANTS: {},
            RunProcessKey.PROCESS_RETURN_CODE: ProcessExitCode.EXCEPTION,
        },
        {
            RunProcessKey.PARTICIPANTS: {},
            RunProcessKey.PROCESS_FINISHED: True,
            RunProcessKey.PROCESS_EXE_ERROR: True,
        },
    ],
    ids=["failure-return-code", "execution-error-without-return-code"],
)
def test_job_complete_process_releases_client_outcome_barrier_after_server_failure(failure_process):
    runner = JobRunner(workspace_root="/tmp")
    runner.logger = MagicMock()
    runner.fire_event = MagicMock()
    runner.log_debug = MagicMock()
    runner.log_info = MagicMock()
    runner.abort_client_run = MagicMock()
    runner._save_workspace = MagicMock()
    runner.ask_to_stop = False

    engine = MagicMock()
    engine.run_processes = {}
    engine.client_manager.clients = {}
    engine.exception_run_processes = {"job-1": failure_process}

    job_manager = MagicMock()
    engine.get_component.return_value = job_manager

    completion_ctx = MagicMock()
    engine.new_context.return_value = nullcontext(completion_ctx)

    job = MagicMock(job_id="job-1", run_aborted=False)
    runner.running_jobs = {"job-1": job}
    runner._pending_client_outcomes = {"job-1": {"site-1"}}
    runner._client_outcome_deadlines = {"job-1": 123.0}

    def _stop_after_first_pass(_):
        runner.ask_to_stop = True

    with _patch_job_runner_sleep(_stop_after_first_pass):
        runner._job_complete_process(engine)

    runner._save_workspace.assert_called_once_with(completion_ctx, ANY, "job-1")
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_EXECUTION_EXCEPTION, completion_ctx)
    runner.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, completion_ctx)
    engine.remove_exception_process.assert_called_once_with("job-1")
    assert "job-1" not in runner.running_jobs
    assert "job-1" not in runner._pending_client_outcomes
    assert "job-1" not in runner._client_outcome_deadlines
    runner.logger.info.assert_called_once_with(
        "Server failure for job (job-1); skipping client outcome wait for: ['site-1']"
    )


def test_job_complete_process_keeps_client_outcome_barrier_for_clean_unknown_return_code():
    runner = JobRunner(workspace_root="/tmp")
    runner.ask_to_stop = False

    engine = MagicMock()
    engine.run_processes = {}
    engine.exception_run_processes = {
        "job-1": {
            RunProcessKey.PROCESS_RETURN_CODE: JobReturnCode.UNKNOWN,
            RunProcessKey.PROCESS_FINISHED: True,
            RunProcessKey.PROCESS_EXE_ERROR: False,
        }
    }
    job_manager = MagicMock()
    engine.get_component.return_value = job_manager

    runner.running_jobs = {"job-1": MagicMock(job_id="job-1", run_aborted=False)}
    runner._pending_client_outcomes = {"job-1": {"site-1"}}

    def _stop_after_first_pass(_):
        runner.ask_to_stop = True

    with _patch_job_runner_sleep(_stop_after_first_pass):
        runner._job_complete_process(engine)

    job_manager.set_status.assert_not_called()
    assert runner._pending_client_outcomes["job-1"] == {"site-1"}


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
    runner._pending_client_outcomes = {"job-1": {"site-1"}}

    def _stop_after_first_pass(_):
        runner.ask_to_stop = True

    with _patch_job_runner_sleep(_stop_after_first_pass):
        runner._job_complete_process(engine)

    job_manager.set_status.assert_not_called()
    runner.ask_to_stop = False
    job.run_aborted = True
    with _patch_job_runner_sleep(_stop_after_first_pass):
        runner._job_complete_process(engine)

    runner._save_workspace.assert_called_once_with(completion_ctx, ANY, "job-1")
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_ABORTED, completion_ctx)
    assert runner.fire_event.call_args_list == [
        call(EventType.JOB_ABORTED, completion_ctx),
        call(EventType.JOB_COMPLETED, completion_ctx),
    ]
    engine.remove_exception_process.assert_called_once_with("job-1")
    assert "job-1" not in runner.running_jobs


@patch("nvflare.private.fed.server.job_runner.ConfigService.get_float_var", return_value=2.0)
def test_job_complete_process_finalizes_server_outcome_after_client_outcome_grace(mock_get_float):
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event = MagicMock()
    runner.logger = MagicMock()
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

    job = MagicMock(job_id="job-1", run_aborted=False)
    runner.running_jobs = {"job-1": job}
    runner._pending_client_outcomes = {"job-1": {"site-1"}}

    sleep_count = 0

    def _stop_after_second_pass(_):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            runner.ask_to_stop = True

    with patch(
        "nvflare.private.fed.server.job_runner.time",
        **{"sleep.side_effect": _stop_after_second_pass, "monotonic.side_effect": [0.0, 2.0]},
    ):
        runner._job_complete_process(engine)

    mock_get_float.assert_called_once()
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_COMPLETED, completion_ctx)
    assert "site-1" in runner.logger.warning.call_args[0][0]
    assert "job-1" not in runner._client_outcome_deadlines


def test_job_complete_process_contains_resource_stats_finalization_write_failure():
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event = MagicMock()
    runner.log_debug = MagicMock()
    runner.log_exception = MagicMock()
    runner._save_workspace = MagicMock()
    runner._accept_server_resource_report = MagicMock()
    runner._fire_job_lifecycle_event = MagicMock()
    runner.ask_to_stop = False
    runner.resource_stats.has_job = MagicMock(return_value=True)
    runner.resource_stats.finalize_job = MagicMock(side_effect=OSError("disk full"))
    runner.resource_stats.discard_job_artifacts = MagicMock()

    engine = MagicMock()
    engine.run_processes = {}
    engine.exception_run_processes = {}
    parent_cell = engine.get_cell.return_value
    parent_cell.get_fqcn.return_value = "server"
    parent_cell.get_traffic_stats.return_value = {"server.job-1": {"sent": 1, "received": 2}}
    parent_cell.pop_traffic_stats.side_effect = RuntimeError("counter unavailable")
    job_manager = MagicMock()
    engine.get_component.return_value = job_manager
    completion_ctx = MagicMock()
    engine.new_context.return_value = nullcontext(completion_ctx)

    job = MagicMock(job_id="job-1", run_aborted=True)
    job.meta = {}
    runner.running_jobs = {"job-1": job}

    def _stop_after_first_pass(_):
        runner.ask_to_stop = True

    with _patch_job_runner_sleep(_stop_after_first_pass) as clock:
        runner._job_complete_process(engine)

    runner.resource_stats.finalize_job.assert_called_once_with("job-1")
    clock.sleep.assert_called_once_with(1.0)
    runner._accept_server_resource_report.assert_called_once_with("job-1", completion_ctx)
    runner.resource_stats.discard_job_artifacts.assert_called_once_with("job-1")
    runner._save_workspace.assert_called_once_with(completion_ctx, ANY, "job-1")
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_ABORTED, completion_ctx)
    runner._fire_job_lifecycle_event.assert_has_calls(
        [
            call(EventType.JOB_ABORTED, "job-1", completion_ctx),
            call(EventType.JOB_COMPLETED, "job-1", completion_ctx),
        ]
    )
    assert "Failed to finalize resource statistics" in runner.log_exception.call_args[0][1]
    parent_cell.pop_traffic_stats.assert_called_once_with("server.job-1")


@pytest.mark.parametrize("aborted", [False, True])
def test_job_completion_retires_parent_peers_after_final_report_archive_and_status(aborted):
    order = []
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event = MagicMock()
    runner.log_debug = MagicMock()
    runner._get_finished_job_status = MagicMock(return_value=RunStatus.FINISHED_COMPLETED)
    runner._accept_server_resource_report = MagicMock(side_effect=lambda *_args: order.append("accept"))
    runner._save_workspace = MagicMock(side_effect=lambda *_args: order.append("archive"))
    runner.resource_stats = MagicMock()
    runner.resource_stats.finalize_job.side_effect = lambda *_args: order.append("finalize")
    participants = [{"name": "site-1", "fqcn": "relay-1.site-1"}]
    job = MagicMock(job_id="job-1", run_aborted=aborted)
    job.meta = {JobMetaKey.JOB_CLIENTS: participants}
    runner.running_jobs = {"job-1": job}
    engine = MagicMock()
    engine.run_processes = {}
    engine.exception_run_processes = {}
    job_manager = engine.get_component.return_value
    job_manager.set_status.side_effect = lambda *_args: order.append("status")
    engine.new_context.return_value = nullcontext(MagicMock())

    def retire_before_unregister(cell, job_id, sites):
        assert cell is engine.get_cell.return_value
        assert sites is participants
        assert job_id in runner.running_jobs
        order.append("retire")

    with (
        _patch_job_runner_sleep(lambda _seconds: setattr(runner, "ask_to_stop", True)),
        patch("nvflare.private.fed.server.job_runner.retire_job_traffic", side_effect=retire_before_unregister),
    ):
        runner._job_complete_process(engine)

    assert order == ["accept", "finalize", "archive", "status", "retire"]
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

    with _patch_job_runner_sleep(_stop_after_first_pass):
        runner._job_complete_process(engine)

    assert runner._save_workspace.call_args_list == [call(first_ctx, ANY, "job-1"), call(second_ctx, ANY, "job-2")]
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

    with _patch_job_runner_sleep(_stop_after_second_pass):
        runner._job_complete_process(engine)

    assert runner._save_workspace.call_args_list == [call(first_ctx, ANY, "job-1"), call(second_ctx, ANY, "job-1")]
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


def test_job_complete_process_publishes_terminal_status_after_workspace_save_grace_time():
    runner = JobRunner(workspace_root="/tmp")
    runner.fire_event = MagicMock()
    runner.log_debug = MagicMock()
    runner.log_error = MagicMock()
    runner.log_exception = MagicMock()
    runner.log_info = MagicMock()
    runner.abort_client_run = MagicMock()
    runner._save_workspace = MagicMock(side_effect=RuntimeError("storage unavailable"))
    runner.ask_to_stop = False

    engine = MagicMock()
    engine.run_processes = {}
    engine.client_manager.clients = {}
    engine.exception_run_processes = {
        "job-1": {
            RunProcessKey.PARTICIPANTS: {},
            RunProcessKey.PROCESS_RETURN_CODE: ProcessExitCode.EXCEPTION,
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

    with patch(
        "nvflare.private.fed.server.job_runner.time",
        **{"sleep.side_effect": _stop_after_second_pass, "monotonic.side_effect": [0.0, 60.0]},
    ):
        runner._job_complete_process(engine)

    assert runner._save_workspace.call_args_list == [call(first_ctx, ANY, "job-1"), call(second_ctx, ANY, "job-1")]
    runner.abort_client_run.assert_called_once_with("job-1", [], first_ctx)
    runner.log_exception.assert_called_once()
    runner.log_error.assert_called_once()
    job_manager.set_status.assert_called_once_with("job-1", RunStatus.FINISHED_EXECUTION_EXCEPTION, second_ctx)
    runner.fire_event.assert_called_once_with(EventType.JOB_COMPLETED, second_ctx)
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

    with _patch_job_runner_sleep(_stop_after_second_pass):
        runner._job_complete_process(engine)

    runner._save_workspace.assert_called_once_with(first_ctx, ANY, "job-1")
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


@patch("nvflare.private.fed.server.job_runner.check_client_replies", return_value=[])
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_continues_when_resource_stats_setup_fails(mock_get_bool, mock_check_replies):
    runner, fl_ctx, engine, job, client_sites = _make_runner_inputs()
    runner.resource_stats.start_job = MagicMock(side_effect=OSError("unsafe path"))

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    engine.start_app_on_server.assert_called_once()
    assert "Resource statistics are unavailable" in runner.log_warning.call_args[0][1]


@patch("nvflare.private.fed.server.job_runner.check_client_replies", return_value=[])
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_registers_resource_participants_without_copying_job_name(mock_get_bool, mock_check_replies):
    runner, fl_ctx, _engine, job, client_sites = _make_runner_inputs()
    fl_ctx.get_workspace.return_value.get_run_dir.return_value = "/tmp/run_job-1"

    runner._start_run(job_id=job.job_id, job=job, client_sites=client_sites, fl_ctx=fl_ctx)

    runner.resource_stats.start_job.assert_called_once_with("job-1", ["site-1"], "/tmp/run_job-1")


@patch("nvflare.private.fed.server.job_runner.check_client_replies", return_value=[])
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_start_run_keeps_selected_but_not_deployable_client_missing(mock_get_bool, mock_check_replies, tmp_path):
    runner, fl_ctx, engine, job, client_sites = _make_runner_inputs()
    runner.resource_stats = type(runner.resource_stats)()
    fl_ctx.get_workspace.return_value.get_run_dir.return_value = str(tmp_path / "run_job-1")

    runner._start_run(
        job_id=job.job_id,
        job=job,
        client_sites=client_sites,
        fl_ctx=fl_ctx,
        expected_client_names=["site-1", "site-2"],
    )
    summary = runner.resource_stats.finalize_job(job.job_id)

    site2 = next(entry for entry in summary["participants"] if entry["participant_name"] == "site-2")
    assert site2["status"] == "missing"
    assert job.meta[JobMetaKey.RESOURCE_PARTICIPANTS] == ["site-1", "site-2"]
    engine.get_component.return_value.update_meta.assert_called_once_with(
        "job-1",
        {JobMetaKey.RESOURCE_PARTICIPANTS.value: ["site-1", "site-2"]},
        fl_ctx,
    )


@patch("nvflare.private.fed.server.job_runner.check_client_replies", return_value=[])
@patch("nvflare.private.fed.server.job_runner.ConfigService.get_bool_var", return_value=True)
def test_deploy_failed_client_renders_disabled_not_missing(mock_get_bool, mock_check_replies, tmp_path):
    # Mirrors the call sequence run() performs once a site fails to deploy: start
    # the run with that site already excluded from client_sites but still in
    # expected_client_names, then disable it so it is distinguishable from a site
    # that was expected to report and simply never did.
    runner, fl_ctx, engine, job, client_sites = _make_runner_inputs()
    runner.resource_stats = type(runner.resource_stats)()
    fl_ctx.get_workspace.return_value.get_run_dir.return_value = str(tmp_path / "run_job-1")
    failed_clients = ["site-2"]

    runner._start_run(
        job_id=job.job_id,
        job=job,
        client_sites=client_sites,
        fl_ctx=fl_ctx,
        expected_client_names=["site-1", "site-2"],
    )
    runner.resource_stats.disable_clients(job.job_id, failed_clients)
    summary = runner.resource_stats.finalize_job(job.job_id)

    site2 = next(entry for entry in summary["participants"] if entry["participant_name"] == "site-2")
    assert site2["status"] == "disabled"


def test_job_runner_does_not_create_shared_parent_traffic_counters():
    runner = JobRunner(workspace_root="/tmp")

    assert not hasattr(runner, "f3_counters")


@pytest.mark.parametrize(
    "traffic",
    [None, {"sent_bytes": "0", "received_bytes": "0"}, {"sent_bytes": "1024", "received_bytes": "512"}],
)
def test_server_handoff_is_bound_and_accepted_before_finalization(tmp_path, traffic):
    runner = JobRunner(workspace_root=str(tmp_path))
    run_dir = tmp_path / "run_job-1"
    runner.resource_stats.start_job("job-1", ["site-1"], run_dir)
    workspace = MagicMock()
    workspace.get_run_dir.return_value = str(run_dir)
    fl_ctx = MagicMock()
    fl_ctx.get_workspace.return_value = workspace
    if traffic is not None:
        write_terminal_handoff(
            run_dir,
            {
                "duration_seconds": "10",
                "internal_version": INTERNAL_HANDOFF_VERSION,
                "kind": INTERNAL_HANDOFF_KIND,
                "resource_capacity": {
                    "cpu": {"groups": [{"units": "2"}]},
                    "memory": {"bytes": "4096"},
                    "gpu": {"groups": []},
                },
                "resource_time": capacity_to_resource_time(
                    {
                        "cpu": {"groups": [{"units": "2"}]},
                        "memory": {"bytes": "4096"},
                        "gpu": {"groups": []},
                    },
                    "10",
                ),
                "workspace_filesystem": {"capacity_bytes": "1024"},
                "resource_utilization": {
                    "cpu_consumed": {"cpu_seconds": "0"},
                    "retained_content": {"bytes": "0"},
                    "message_traffic": traffic,
                },
            },
        )

    runner._accept_server_resource_report("job-1", fl_ctx)
    summary = runner.resource_stats.finalize_job("job-1")

    server = next(entry for entry in summary["participants"] if entry["participant_name"] == "server")
    assert "status" not in server
    if traffic is None:
        assert server["resource_capacity"] == {"status": "unavailable", "issues": ["observation_incomplete"]}
        assert server["resource_utilization"]["message_traffic"] == {
            "status": "unavailable",
            "issues": ["observation_incomplete"],
        }
    else:
        assert server["duration_seconds"] == "10"
        assert server["resource_utilization"]["message_traffic"] == traffic
    assert not terminal_handoff_path(run_dir).exists()
    assert not hasattr(runner, "f3_counters")


def test_restore_running_job_registers_resource_participants_before_server_start(tmp_path):
    runner = JobRunner(workspace_root=str(tmp_path))
    runner.scheduler = MagicMock()
    workspace = MagicMock()
    workspace.get_run_dir.return_value = str(tmp_path / "run_job-1")
    fl_ctx = MagicMock()
    fl_ctx.get_workspace.return_value = workspace
    engine = fl_ctx.get_engine.return_value
    job = MagicMock()
    job.meta = {
        JobMetaKey.JOB_NAME.value: "hello-pt",
        JobMetaKey.RESOURCE_PARTICIPANTS.value: ["site-1", "site-2"],
    }
    engine.get_component.return_value.get_job.return_value = job
    client = MagicMock()
    client.name = "site-1"

    def assert_registered_before_start(*args, **kwargs):
        assert runner.resource_stats.has_job("job-1")
        return ""

    engine.start_app_on_server.side_effect = assert_registered_before_start

    runner.restore_running_job("job-1", {"token-1": client}, None, fl_ctx)

    assert runner.resource_stats.accept_resource_report("job-1", "site-1", {"participant_summary": b"{}"}) == "invalid"
    summary = runner.resource_stats.finalize_job("job-1")
    assert summary["job_id"] == "job-1"
    site2 = next(entry for entry in summary["participants"] if entry["participant_name"] == "site-2")
    assert site2["status"] == "missing"
    assert engine.start_app_on_server.call_args.kwargs["restored_attempt"] is True
    assert engine.start_app_on_server.call_args.kwargs["snapshot"] is None
    runner.scheduler.restore_scheduled_job.assert_called_once_with("job-1")


def _make_restore_inputs(tmp_path):
    runner, fl_ctx, engine, job, _client_sites = _make_runner_inputs()
    runner.scheduler = MagicMock()
    runner.log_error = MagicMock()
    fl_ctx.get_workspace.return_value.get_run_dir.return_value = str(tmp_path / "run_job-1")
    engine.get_component.return_value.get_job.return_value = job
    client = MagicMock()
    client.name = "site-1"
    return runner, fl_ctx, engine, job, {"token-1": client}


@pytest.mark.parametrize("raise_on_start", (False, True))
def test_failed_restore_does_not_claim_running_ownership(tmp_path, raise_on_start):
    runner, fl_ctx, engine, _job, clients = _make_restore_inputs(tmp_path)
    if raise_on_start:
        engine.start_app_on_server.side_effect = RuntimeError("launch failed")
    else:
        engine.start_app_on_server.return_value = "launch failed"

    runner.restore_running_job("job-1", clients, None, fl_ctx)

    assert "job-1" not in runner.running_jobs
    runner.scheduler.restore_scheduled_job.assert_not_called()
    runner.log_error.assert_called_once()


@pytest.mark.parametrize("already_running", (False, True))
def test_failed_restore_preserves_existing_running_job(tmp_path, already_running):
    runner, fl_ctx, engine, job, clients = _make_restore_inputs(tmp_path)
    if already_running:
        runner.running_jobs["job-1"] = job
    engine.start_app_on_server.return_value = "launch failed"

    runner.restore_running_job("job-1", clients, None, fl_ctx)

    assert ("job-1" in runner.running_jobs) is already_running
    if already_running:
        assert runner.running_jobs["job-1"] is job
    runner.scheduler.restore_scheduled_job.assert_not_called()
    runner.log_error.assert_called_once()


def test_scheduler_restore_failure_keeps_running_job_owned(tmp_path):
    runner, fl_ctx, _engine, job, clients = _make_restore_inputs(tmp_path)
    runner.scheduler.restore_scheduled_job.side_effect = RuntimeError("scheduler restore failed")

    runner.restore_running_job("job-1", clients, None, fl_ctx)

    assert runner.running_jobs["job-1"] is job
    assert runner._pending_client_outcomes["job-1"] == {"site-1"}
    runner.log_error.assert_called_once()


def test_restore_continues_when_optional_resource_report_setup_fails(tmp_path):
    runner, fl_ctx, _engine, job, clients = _make_restore_inputs(tmp_path)
    runner.resource_stats.start_job.side_effect = RuntimeError("report setup failed")
    runner.resource_stats.forget_job = MagicMock()

    runner.restore_running_job("job-1", clients, None, fl_ctx)

    assert runner.running_jobs["job-1"] is job
    runner.resource_stats.start_job.assert_called_once_with(
        "job-1", ["site-1"], str(tmp_path / "run_job-1"), reset_existing=True
    )
    runner.resource_stats.forget_job.assert_called_once_with("job-1")
    runner.scheduler.restore_scheduled_job.assert_called_once_with("job-1")
    runner.log_warning.assert_called_once()
