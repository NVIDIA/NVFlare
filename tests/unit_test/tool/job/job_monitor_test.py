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
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, MonitorReturnCode, NoConnection
from nvflare.tool import cli_output
from nvflare.tool.job.job_cli import _parse_monitor_start_ts


def _make_args(job_id="abc123", timeout=0, interval=2, stats_target="server", metrics=None):
    args = MagicMock()
    args.job_id = job_id
    args.timeout = timeout
    args.interval = interval
    args.stats_target = stats_target
    args.metrics = metrics or []
    return args


def _make_meta(status="FINISHED_OK", job_name="test-job", duration="0:01:30"):
    from nvflare.apis.job_def import JobMetaKey

    return {
        JobMetaKey.STATUS.value: status,
        JobMetaKey.JOB_NAME.value: job_name,
        JobMetaKey.DURATION.value: duration,
    }


def test_parse_monitor_start_ts_from_start_time():
    from nvflare.apis.job_def import JobMetaKey

    meta = {
        JobMetaKey.START_TIME.value: "2026-04-16 12:34:56.000000",
    }

    result = _parse_monitor_start_ts(meta, JobMetaKey.START_TIME.value, JobMetaKey.SUBMIT_TIME_ISO.value)

    assert result is not None


def test_parse_monitor_start_ts_from_submit_time_iso():
    from nvflare.apis.job_def import JobMetaKey

    meta = {
        JobMetaKey.SUBMIT_TIME_ISO.value: "2026-04-16T12:34:56",
    }

    result = _parse_monitor_start_ts(meta, JobMetaKey.START_TIME.value, JobMetaKey.SUBMIT_TIME_ISO.value)

    assert result is not None


def _mock_session(rc, meta):
    """Return a patch context for _session() that yields a mock session."""
    mock_sess = MagicMock()
    mock_sess.monitor_job_and_return_job_meta.return_value = (rc, meta)
    mock_sess.show_stats.return_value = {}

    @contextmanager
    def _fake_session():
        yield mock_sess

    return patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session), mock_sess


class TestJobMonitorOutput:
    """Tests for nvflare job monitor output format and exit codes."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    # ------------------------------------------------------------------
    # Terminal states — exit codes and envelope shape
    # ------------------------------------------------------------------

    def test_finished_ok_exits_0(self, capsys):
        meta = _make_meta("FINISHED_OK")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["job_id"] == "abc123"
        assert data["data"]["status"] == "FINISHED_OK"

    def test_failed_outputs_ok_envelope_exits_1(self, capsys):
        meta = _make_meta("FAILED")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 1
        assert data["data"]["status"] == "FAILED"

    def test_aborted_outputs_ok_envelope_exits_1(self, capsys):
        meta = _make_meta("ABORTED")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["status"] == "ABORTED"

    def test_abandoned_exits_1(self, capsys):
        meta = _make_meta("ABANDONED")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

    def test_finished_exception_exits_1(self, capsys):
        meta = _make_meta("FINISHED_EXCEPTION")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_timeout_exits_3_with_error_code(self, capsys):
        ctx, _ = _mock_session(MonitorReturnCode.TIMEOUT, None)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args(timeout=10))
        assert exc_info.value.code == 3

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "TIMEOUT"
        assert data["exit_code"] == 3

    def test_job_not_found_exits_1(self, capsys):
        @contextmanager
        def _fake_session():
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = JobNotFound("job does not exist")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "JOB_NOT_FOUND"

    def test_connection_error_propagates_to_top_level_handler(self):
        @contextmanager
        def _fake_session():
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = NoConnection("connection refused")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(NoConnection):
                cmd_job_monitor(_make_args())

    def test_authentication_error_propagates_to_top_level_handler(self):
        @contextmanager
        def _fake_session():
            sess = MagicMock()
            sess.monitor_job_and_return_job_meta.side_effect = AuthenticationError("bad cert")
            yield sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(AuthenticationError):
                cmd_job_monitor(_make_args())

    def test_missing_meta_exits_internal_error(self, capsys):
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, None)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(_make_args())
        assert exc_info.value.code == 5

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "INTERNAL_ERROR"

    # ------------------------------------------------------------------
    # Envelope contents
    # ------------------------------------------------------------------

    def test_envelope_contains_duration_s(self, capsys):
        meta = _make_meta("FINISHED_OK", duration="0:01:30")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        data = json.loads(capsys.readouterr().out)
        assert "duration_s" in data["data"]
        assert data["data"]["duration_s"] == 90.0

    def test_envelope_contains_job_meta_summary(self, capsys):
        meta = _make_meta("FINISHED_OK", job_name="hello-pt")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        data = json.loads(capsys.readouterr().out)
        assert data["data"]["job_meta"]["job_name"] == "hello-pt"
        assert data["data"]["job_meta"]["status"] == "FINISHED_OK"

    def test_stats_raw_included_in_json_mode(self, capsys):
        """In json mode, stats_raw is included in the envelope data."""
        meta = _make_meta("FINISHED_OK")
        mock_sess = MagicMock()
        mock_sess.monitor_job_and_return_job_meta.return_value = (MonitorReturnCode.JOB_FINISHED, meta)
        mock_sess.show_stats.return_value = {"server": {"round": 10, "loss": 0.05}}

        @contextmanager
        def _fake_session():
            yield mock_sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        data = json.loads(capsys.readouterr().out)
        assert "stats_raw" in data["data"]

    def test_no_human_text_on_stdout(self, capsys):
        """In json mode, stdout contains only one JSON line."""
        meta = _make_meta("FINISHED_OK")
        ctx, _ = _mock_session(MonitorReturnCode.JOB_FINISHED, meta)
        with ctx:
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        captured = capsys.readouterr()
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1
        json.loads(stdout_lines[0])  # must be valid JSON

    # ------------------------------------------------------------------
    # ENDED_BY_CB path (callback stopped monitoring)
    # ------------------------------------------------------------------

    def test_ended_by_cb_uses_last_meta(self, capsys):
        """When rc is ENDED_BY_CB, result comes from cb_state last_meta."""
        terminal_meta = _make_meta("FINISHED_OK")

        def _side_effect(job_id, timeout, poll_interval, cb, state):
            # Simulate callback receiving terminal status and stopping
            state["last_meta"] = terminal_meta
            cb(MagicMock(), job_id, terminal_meta, state)
            return MonitorReturnCode.ENDED_BY_CB, None

        mock_sess = MagicMock()
        mock_sess.monitor_job_and_return_job_meta.side_effect = _side_effect
        mock_sess.show_stats.return_value = {}

        @contextmanager
        def _fake_session():
            yield mock_sess

        with patch("nvflare.tool.job.job_cli._session", side_effect=_fake_session):
            from nvflare.tool.job.job_cli import cmd_job_monitor

            cmd_job_monitor(_make_args())

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["status"] == "FINISHED_OK"

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def test_parser_args(self):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        def_job_cli_parser(root.add_subparsers())

        parser = job_sub_cmd_parser["monitor"]
        args = parser.parse_args(["abc123", "--timeout", "300", "--interval", "5"])
        assert args.job_id == "abc123"
        assert args.timeout == 300
        assert args.interval == 5

    def test_parser_defaults(self):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        def_job_cli_parser(root.add_subparsers())

        parser = job_sub_cmd_parser["monitor"]
        args = parser.parse_args(["abc123"])
        assert args.timeout == 0
        assert args.interval == 2
        assert args.stats_target == "server"
        assert args.metrics is None

    def test_parser_stats_target_and_metric(self):
        """--stats-target and --metric are parsed correctly."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        def_job_cli_parser(root.add_subparsers())

        parser = job_sub_cmd_parser["monitor"]
        args = parser.parse_args(["abc123", "--stats-target", "client", "--metric", "loss", "--metric", "accuracy"])
        assert args.stats_target == "client"
        assert args.metrics == ["loss", "accuracy"]
