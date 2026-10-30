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

"""Reporting samples counters without changing normal job lifecycle."""

import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.job_def import RunStatus
from nvflare.private.fed.app import job_process_cleanup
from nvflare.private.fed.app.client import worker_process
from nvflare.private.fed.app.server import runner_process
from nvflare.private.fed.server import job_runner


class _StartupObserved(Exception):
    pass


@pytest.mark.parametrize("module", [worker_process, runner_process])
def test_collection_keeps_existing_workspace_and_import_order(module, tmp_path, monkeypatch):
    events = []
    args = SimpleNamespace(set=[], workspace=str(tmp_path), client_name="site-1", job_id="job-1")
    monkeypatch.setenv("PYTHONPATH", "existing-custom-import-path")
    original_path = sys.path[:]
    monkeypatch.setattr(module, "parse_vars", lambda _: {})
    monkeypatch.setattr(module, "Workspace", MagicMock())
    monkeypatch.setattr(module, "JobResourceCollector", lambda *_, **__: events.append("capacity"))
    monkeypatch.setattr(module, "download_workspace", lambda *_: events.append("download"))
    monkeypatch.setattr(module, "refresh_custom_dir_import_path", lambda *_: events.append("refresh"))

    def normal_configuration(*_):
        events.append("configuration")
        raise _StartupObserved()

    monkeypatch.setattr(module, "set_stats_pool_config_for_job", normal_configuration)
    with pytest.raises(_StartupObserved):
        module.main(args)

    assert events == ["download", "refresh", "capacity", "configuration"]
    assert sys.path == original_path
    assert os.environ["PYTHONPATH"] == "existing-custom-import-path"


@pytest.mark.parametrize("module", [worker_process, runner_process])
@pytest.mark.parametrize("callbacks_drained", [False, True])
@pytest.mark.parametrize("restored", [False, True])
def test_job_samples_actual_cell_before_existing_archive(module, callbacks_drained, restored, tmp_path, monkeypatch):
    events = []
    args = SimpleNamespace(
        set=[],
        workspace=str(tmp_path),
        client_name="site-1",
        job_id="job-1",
        token="token",
        token_signature="signature",
        ssid="session",
        app_root=str(tmp_path),
        root_url="root",
        parent_url="parent",
        host="localhost",
        port="8002",
    )
    workspace = MagicMock()
    workspace.get_file_path_in_root.side_effect = lambda name: str(tmp_path / name)
    app_runner, server, client, config = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    collector = MagicMock()
    monkeypatch.setattr(module, "parse_vars", lambda _: {"secure_train": False, "restore_snapshot": restored})
    monkeypatch.setattr(module, "Workspace", lambda *_, **__: workspace)
    monkeypatch.setattr(module, "JobResourceCollector", lambda *_, **__: collector)
    monkeypatch.setattr(module.threading, "Thread", MagicMock())
    for name in (
        "download_workspace",
        "refresh_custom_dir_import_path",
        "set_stats_pool_config_for_job",
        "fobs_initialize",
        "security_init_for_job",
        "configure_logging",
        "register_ext_decomposers",
    ):
        monkeypatch.setattr(module, name, MagicMock())
    monkeypatch.setattr(module, "get_script_logger", MagicMock())
    monkeypatch.setattr(module, "write_terminal_handoff", MagicMock())
    monkeypatch.setattr(module, "create_stats_pool_files_for_job", lambda *_: events.append("stats"))
    monkeypatch.setattr(module, "upload_results_on_shutdown", lambda *_, **__: events.append("upload"))
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: events.append("streams"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: events.append("security"))

    def snapshot(cell, *, incomplete):
        assert cell is (client.cell if module is worker_process else server.cell)
        assert incomplete is (not callbacks_drained or (restored and module is runner_process))
        events.append("snapshot")
        return {"sent_bytes": "7", "received_bytes": "11"}

    monkeypatch.setattr(module, "snapshot_message_traffic", snapshot)

    callback_timeouts = []

    def drain(timeout):
        callback_timeouts.append(timeout)
        events.append("callbacks")
        return callbacks_drained

    if module is worker_process:
        monkeypatch.setattr(module, "remove_restart_file", MagicMock())
        monkeypatch.setattr(module, "FLClientStarterConfiger", lambda **_: config)
        config.base_deployer.create_fed_client.return_value = client
        monkeypatch.setattr(module, "ClientAppRunner", lambda **_: app_runner)
        app_runner.close.side_effect = lambda: events.append("stop_commands")
        app_runner.wait_for_command_callbacks.side_effect = drain
        client.stop_cell.side_effect = lambda: events.append("cell")
    else:
        monkeypatch.setattr(module.os, "chdir", lambda _: None)
        monkeypatch.setattr(module, "FLServerStarterConfiger", lambda **_: config)
        config.deployer.create_fl_server.return_value = ({}, server)
        monkeypatch.setattr(module, "ServerAppRunner", lambda _: app_runner)
        monkeypatch.setattr(module, "set_add_auth_headers_filters", MagicMock())
        server.command_agent.shutdown.side_effect = lambda: events.append("stop_commands")
        server.command_agent.wait_for_callbacks.side_effect = drain
        server.create_job_cell.return_value = server.cell
        server.cell.stop.side_effect = lambda: events.append("cell")

    module.main(args)

    expected = ["stop_commands", "callbacks", "snapshot", "stats", "upload", "streams", "cell"]
    if not callbacks_drained:
        collector.mark_cpu_consumption_incomplete.assert_called_once_with()
    expected += ["callbacks", "security"]
    assert events == expected
    assert callback_timeouts == [0.0, 5.0]
    collector.finish.assert_called_once_with(message_traffic={"sent_bytes": "7", "received_bytes": "11"})


@pytest.mark.parametrize("failure_stage", [None, "archive", "status"])
def test_parent_finalizes_before_archive_and_terminal_publication(failure_stage, tmp_path, monkeypatch):
    events = []
    runner = job_runner.JobRunner(str(tmp_path))
    runner._fire_job_lifecycle_event = MagicMock()
    ctx, engine, manager = MagicMock(), MagicMock(), MagicMock()
    ctx.get_engine.return_value = engine
    ctx.get_peer_context.return_value = None
    ctx.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    engine.new_context.side_effect = lambda: nullcontext(ctx)
    engine.get_component.return_value = manager
    engine.run_processes = {}
    engine.exception_run_processes = {}
    runner.running_jobs = {"job-1": SimpleNamespace(job_id="job-1", run_aborted=True)}
    runner.resource_stats.start_job("job-1", [], str(tmp_path))
    original_finalize = runner.resource_stats.finalize_job

    def finalize(job_id):
        events.append("finalize")
        return original_finalize(job_id)

    def archive(*_):
        events.append("archive")
        if failure_stage == "archive" and events.count("archive") == 1:
            raise OSError("archive not ready")

    def publish(*_):
        events.append("status")
        if failure_stage == "status" and events.count("status") == 1:
            raise OSError("status store not ready")

    passes = []

    def next_pass(_):
        passes.append(None)
        if len(passes) == (2 if failure_stage else 1):
            runner.stop()
        else:
            assert runner._finished_job_states["job-1"].resource_stats_finalized is True

    monkeypatch.setattr(runner.resource_stats, "finalize_job", finalize)
    monkeypatch.setattr(runner, "_save_workspace", archive)
    manager.set_status.side_effect = publish
    monkeypatch.setattr(job_runner, "time", SimpleNamespace(sleep=next_pass, monotonic=lambda: 0.0))
    runner._job_complete_process(engine)

    expected = {
        None: ["finalize", "archive", "status"],
        "archive": ["finalize", "archive", "archive", "status"],
        "status": ["finalize", "archive", "status", "status"],
    }
    assert events == expected[failure_stage]
    assert "job-1" not in runner.running_jobs
    assert manager.set_status.call_args.args[1] is RunStatus.FINISHED_ABORTED
