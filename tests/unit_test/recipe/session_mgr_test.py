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


@pytest.mark.parametrize("log_mode", ["concise", "progress", "full", "verbose"])
def test_monitor_reports_changes_without_repeating_normal_waits(monkeypatch, capsys, log_mode):
    monkeypatch.setenv("FL_LOG_LEVEL", log_mode)
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
    for field in ("resource_spec", "deploy_map"):
        assert output.count(field) == (0 if log_mode in ("concise", "progress") else 2)


def test_progress_monitor_replays_new_records_and_retries_partial_lines(monkeypatch, capsys):
    now = [0]
    monkeypatch.setattr("nvflare.recipe.session_mgr.time.monotonic", lambda: now[0])
    session = MagicMock()
    start = json.dumps(
        {
            "fullName": "nvflare.app_common.widgets.metrics_artifact_writer",
            "message": "Round 1/3 | training",
        }
    )
    metric = json.dumps(
        {
            "fullName": "nvflare.app_common.widgets.metrics_artifact_writer",
            "message": "site-1 | loss=0.25",
        }
    )
    warning = json.dumps(
        {
            "fullName": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
            "levelname": "WARNING",
            "message": "connection interrupted",
        }
    )
    noise = json.dumps({"fullName": "custom.trainer", "message": "raw model weights"})
    app_noise = json.dumps(
        {
            "fullName": "nvflare.app_common.workflows.lr.fedavg.FedAvgLR",
            "message": "Newton-Raphson updates: [1, 2, 3]",
        }
    )
    unmarked_allowlisted = json.dumps(
        {
            "fullName": "nvflare.app_common.widgets.metrics_artifact_writer",
            "message": "unmarked model contents",
            "nvflare_progress": False,
        }
    )
    marked_other = json.dumps(
        {
            "fullName": "nvflare.app_common.workflows.cyclic",
            "message": "cycle progress",
            "nvflare_progress": True,
        }
    )
    state = {"count": 0, "progress": {"seen": set()}}
    meta = {"status": "RUNNING"}
    session.get_job_logs.return_value = {
        "logs": {"server": "\n".join((start, noise, app_noise, unmarked_allowlisted, marked_other, metric[:20]))}
    }
    _job_monitor_callback(session, "job-id", meta, cb_run_counter=state)
    now[0] = 1
    _job_monitor_callback(session, "job-id", meta, cb_run_counter=state)
    assert session.get_job_logs.call_count == 1
    now[0] = 5
    session.get_job_logs.return_value = {"logs": {"server": start + "\n" + metric + "\n" + warning}}
    _job_monitor_callback(session, "job-id", meta, cb_run_counter=state)
    # Completion forces a final retrieval even before the five-second interval.
    now[0] = 6
    meta["status"] = "FINISHED:COMPLETED"
    _job_monitor_callback(session, "job-id", meta, cb_run_counter=state)
    output = capsys.readouterr().out
    assert output.count("Round 1/3 | training") == 1
    assert output.count("site-1 | loss=0.25") == 1
    assert output.count("WARNING: connection interrupted") == 1
    assert "raw model weights" not in output
    assert "Newton-Raphson updates" not in output
    assert "unmarked model contents" not in output
    assert output.count("cycle progress") == 1
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


@pytest.mark.parametrize(
    "structured_response",
    [
        {"logs": {}, "unavailable": {"server": "server log not available for this job"}},
        {"logs": {}},
    ],
)
def test_progress_empty_structured_response_uses_bounded_text_fallback(capsys, structured_response):
    from nvflare.recipe.session_mgr import _show_job_progress

    session = MagicMock()
    session.get_job_logs.side_effect = [
        structured_response,
        {"logs": {"server": "2026-09-15 - INFO - round started\n2026-09-15 - ERROR - training failed"}},
    ]

    _show_job_progress(session, "job-id", {"seen": set()})

    output = capsys.readouterr().out
    assert "Structured live progress is unavailable; showing the bounded server log tail." in output
    assert "2026-09-15 - INFO - round started" in output
    assert "2026-09-15 - ERROR - training failed" in output
    assert session.get_job_logs.call_args_list[0].kwargs == {
        "target": "server",
        "log_file_name": "log.json",
        "tail_lines": 200,
        "max_bytes": 64 * 1024,
    }
    assert session.get_job_logs.call_args_list[1].kwargs == {
        "target": "server",
        "log_file_name": "log.txt",
        "tail_lines": 50,
        "max_bytes": 64 * 1024,
    }


def test_progress_text_fallback_drops_partial_first_line_before_dedup(capsys):
    from nvflare.recipe.session_mgr import _show_job_progress

    session = MagicMock()
    long_line = "x" * (70 * 1024)
    session.get_job_logs.side_effect = [
        {"logs": {}},
        {"logs": {"server": f"{long_line}\nround 1"}},
        {"logs": {}},
        {"logs": {"server": f"{long_line}\nround 1\nround 2"}},
    ]
    state = {"seen": set()}

    _show_job_progress(session, "job-id", state)
    _show_job_progress(session, "job-id", state)

    output = capsys.readouterr().out
    assert "x" * 100 not in output
    assert output.count("round 1") == 1
    assert output.count("round 2") == 1


def test_progress_unavailable_response_allows_startup_then_warns_once(capsys):
    from nvflare.recipe.session_mgr import _show_job_progress

    session = MagicMock()
    session.get_job_logs.return_value = {
        "logs": {},
        "unavailable": {"server": "server log not available for this job"},
    }
    state = {"seen": set()}

    _show_job_progress(session, "job-id", state)
    assert capsys.readouterr().out == ""
    _show_job_progress(session, "job-id", state)
    _show_job_progress(session, "job-id", state)

    output = capsys.readouterr().out
    assert output.count("Live progress logs are unavailable") == 1
    assert "nvflare job logs" in output


@pytest.mark.parametrize("log_mode,expected", [("concise", False), ("progress", True)])
def test_remote_progress_replay_is_opt_in(monkeypatch, capsys, log_mode, expected):
    monkeypatch.setenv("FL_LOG_LEVEL", log_mode)
    session = MagicMock()
    session.monitor_job.return_value = MonitorReturnCode.ENDED_BY_CB
    manager = SessionManager({})
    manager._get_session = MagicMock(return_value=session)

    assert manager.get_job_result("job-id") is None

    state = session.monitor_job.call_args.kwargs["cb_run_counter"]
    assert ("progress" in state) is expected
    session.close.assert_called_once()
    capsys.readouterr()


def test_monitor_bounds_replay_and_memory_across_many_refreshes(capsys):
    from nvflare.recipe.session_mgr import _show_job_progress

    session = MagicMock()
    state = {"seen": set()}
    for batch in range(10):
        lines = [
            json.dumps(
                {
                    "fullName": "nvflare.app_common.widgets.metrics_artifact_writer",
                    "message": f"row {batch}-{n}",
                }
            )
            for n in range(1000)
        ]
        session.get_job_logs.return_value = {"logs": {"server": "\n".join(lines)}}
        _show_job_progress(session, "job-id", state)
        assert len(state["seen"]) <= 200
        assert all(isinstance(line, str) for line in state["seen"])
        output = capsys.readouterr().out
        assert output.count("row ") == 200
        assert f"row {batch}-999" in output
        _show_job_progress(session, "job-id", state)
        assert capsys.readouterr().out == ""
    session.get_job_logs.assert_called_with(
        "job-id", target="server", log_file_name="log.json", tail_lines=200, max_bytes=64 * 1024
    )


@pytest.mark.parametrize(
    "status,failed",
    [
        ("FINISHED:COMPLETED", False),
        ("FINISHED_OK", False),
        ("FINISHED:CAN_NOT_SCHEDULE", False),
        ("FINISHED:ABANDONED", True),
        ("FINISHED:EXECUTION_EXCEPTION", True),
        ("FAILED", True),
        ("FINISHED_EXCEPTION", True),
        ("ABORTED", True),
        ("ABANDONED", True),
    ],
)
def test_error_log_retrieval_failure_preserves_result_and_closes_session(tmp_path, status, failed):
    session = MagicMock()

    def monitor(*args, **kwargs):
        kwargs["cb_run_counter"]["status"] = status
        return MonitorReturnCode.JOB_FINISHED

    session.monitor_job.side_effect = monitor
    session.download_job_result.return_value = str(tmp_path)
    session.get_job_logs.side_effect = RuntimeError("logs unavailable")
    session.list_job_components.return_value = ["ERRORLOG_site-1"]
    manager = SessionManager({})
    manager._get_session = MagicMock(return_value=session)
    assert manager.get_job_result("job-id") == str(tmp_path)
    assert session.get_job_logs.call_count == int(failed)
    session.close.assert_called_once()
