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
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.flare_api.api_spec import MonitorReturnCode
from nvflare.fuel.utils.secret_utils import PotentialSecretWarning
from nvflare.job_config.api import FedJob
from nvflare.recipe.session_mgr import SessionManager, _job_monitor_callback


def test_submit_job_scans_generated_config_before_submission():
    job = FedJob(name="secret-submit-job", min_clients=1)
    job.to_server({"auth_token": "abcd1234efgh"})

    session = MagicMock()
    session.submit_job.return_value = "job-id"
    manager = SessionManager({})
    manager._get_session = MagicMock(return_value=session)

    with pytest.warns(PotentialSecretWarning, match="generated job file"):
        assert manager.submit_job(job) == "job-id"

    session.submit_job.assert_called_once()
    session.close.assert_called_once()


def test_monitor_reports_changes_without_repeating_normal_waits(monkeypatch, capsys):
    now = [0]
    monkeypatch.setattr("nvflare.recipe.session_mgr.time.monotonic", lambda: now[0])
    state = {"count": 0}
    meta = {"status": "RUNNING", "resource_spec": {"gpu": 2}, "deploy_map": {"app": ["@ALL"]}}
    for tick in (0, 1, 14, 15):
        now[0] = tick
        assert _job_monitor_callback(None, "job-id", meta, cb_run_counter=state)
    meta["status"] = "FINISHED:COMPLETED"
    now[0] = 16
    _job_monitor_callback(None, "job-id", meta, cb_run_counter=state)
    output = capsys.readouterr().out
    assert output.count("Job ID:") == 1
    assert output.count("Job status: RUNNING") == 1
    assert "15s monitored" not in output
    assert "Job status: FINISHED:COMPLETED" in output
    assert "resource_spec" not in output
    assert "deploy_map" not in output


def test_progress_monitor_replays_new_records_and_retries_partial_lines(monkeypatch, capsys):
    now = [0]
    monkeypatch.setattr("nvflare.recipe.session_mgr.time.monotonic", lambda: now[0])
    session = MagicMock()
    start = json.dumps({"fullName": "nvflare.metrics.progress", "message": "Round 1/3 | training"})
    metric = json.dumps({"fullName": "nvflare.metrics.progress", "message": "site-1 | loss=0.25"})
    noise = json.dumps({"fullName": "custom.trainer", "message": "raw model weights"})
    state = {"count": 0, "progress": {"seen": set()}}
    meta = {"status": "RUNNING"}
    session.get_job_logs.return_value = {"logs": {"server": start + "\n" + noise + "\n" + metric[:20]}}
    _job_monitor_callback(session, "job-id", meta, cb_run_counter=state)
    now[0] = 1
    _job_monitor_callback(session, "job-id", meta, cb_run_counter=state)
    assert session.get_job_logs.call_count == 1
    now[0] = 5
    session.get_job_logs.return_value = {"logs": {"server": start + "\n" + metric}}
    _job_monitor_callback(session, "job-id", meta, cb_run_counter=state)
    # Completion forces a final retrieval even before the five-second interval.
    now[0] = 6
    meta["status"] = "FINISHED:COMPLETED"
    _job_monitor_callback(session, "job-id", meta, cb_run_counter=state)
    output = capsys.readouterr().out
    assert output.count("Round 1/3 | training") == 1
    assert output.count("site-1 | loss=0.25") == 1
    assert "raw model weights" not in output
    assert session.get_job_logs.call_count == 3


def test_progress_unavailable_does_not_stop_job_monitoring(monkeypatch, capsys):
    now = [0]
    monkeypatch.setattr("nvflare.recipe.session_mgr.time.monotonic", lambda: now[0])
    session = MagicMock()
    session.get_job_logs.side_effect = RuntimeError("server temporarily unreachable")
    state = {"count": 0, "progress": {"seen": set()}}
    for tick in (0, 5, 10):
        now[0] = tick
        assert _job_monitor_callback(session, "job-id", {"status": "RUNNING"}, cb_run_counter=state)
    assert capsys.readouterr().out.count("Live progress could not be retrieved") == 1


def test_monitor_bounds_replay_and_memory_across_many_refreshes(capsys):
    from nvflare.recipe.session_mgr import _show_job_progress

    session = MagicMock()
    state = {"seen": set()}
    for batch in range(10):
        lines = [
            json.dumps({"fullName": "nvflare.metrics.progress", "message": f"row {batch}-{n}"}) for n in range(1000)
        ]
        session.get_job_logs.return_value = {"logs": {"server": "\n".join(lines)}}
        _show_job_progress(session, "job-id", state)
        assert len(state["seen"]) <= 200
        assert all(isinstance(key, bytes) and len(key) == 32 for key in state["seen"])
        output = capsys.readouterr().out
        assert output.count("row ") == 200
        assert f"row {batch}-999" in output
        _show_job_progress(session, "job-id", state)
        assert capsys.readouterr().out == ""
    session.get_job_logs.assert_called_with("job-id", target="server", log_file_name="log.json", tail_lines=200)


@pytest.mark.parametrize("failed", [False, True])
def test_error_log_retrieval_failure_preserves_result_and_closes_session(tmp_path, failed):
    session = MagicMock()

    def monitor(*args, **kwargs):
        kwargs["cb_run_counter"]["status"] = "FINISHED:EXECUTION_EXCEPTION" if failed else "FINISHED:COMPLETED"
        return MonitorReturnCode.JOB_FINISHED

    session.monitor_job.side_effect = monitor
    session.download_job_result.return_value = str(tmp_path)
    session.get_job_logs.side_effect = RuntimeError("logs unavailable")
    manager = SessionManager({})
    manager._get_session = MagicMock(return_value=session)
    assert manager.get_job_result("job-id") == str(tmp_path)
    assert session.get_job_logs.call_count == int(failed)
    session.close.assert_called_once()
