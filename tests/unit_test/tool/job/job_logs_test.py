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

from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, NoConnection
from nvflare.tool import cli_output


def _make_args(job_id="abc123", site="server"):
    args = MagicMock()
    args.job_id = job_id
    args.site = site
    return args


class TestJobLogs:
    """Tests for nvflare job logs command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _fake_session(self, mock_sess):
        @contextmanager
        def _ctx(*_args, **_kwargs):
            yield mock_sess

        return _ctx

    def test_logs_json_envelope_shape(self, capsys):
        """get_job_logs result is wrapped in the expected JSON envelope."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {"logs": {"server": "line1\nline2\n"}}

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args(job_id="abc123"))

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "ok"
        assert envelope["exit_code"] == 0
        data = envelope["data"]
        assert data["job_id"] == "abc123"
        assert data["target"] == "server"
        assert data["logs"] == {"server": "line1\nline2\n"}

    def test_logs_no_log_source_field(self, capsys):
        """log_source is not present in the output data."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {"logs": {}}

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args())

        data = json.loads(capsys.readouterr().out)["data"]
        assert "log_source" not in data

    def test_logs_keyed_by_site(self, capsys):
        """logs dict is keyed by site name."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {"logs": {"server": "log text"}}

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args())

        data = json.loads(capsys.readouterr().out)["data"]
        assert "server" in data["logs"]

    def test_logs_human_single_site_prints_raw_log_text(self, capsys, monkeypatch):
        """Human mode prints readable log text instead of a dict/table."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {"logs": {"server": "line1\nline2\n"}}

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args())

        captured = capsys.readouterr()
        assert captured.out == "line1\nline2\n"
        assert captured.err == ""

    def test_logs_human_all_sites_prints_headers_and_unavailable(self, capsys, monkeypatch):
        from nvflare.tool.job.job_cli import cmd_job_logs

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {
            "logs": {"server": "server log\n", "site-1": "client log\n"},
            "unavailable": {"site-2": "client log stream not available for this job"},
        }

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args(site="all"))

        captured = capsys.readouterr()
        assert "===== server =====\nserver log\n" in captured.out
        assert "===== site-1 =====\nclient log\n" in captured.out
        assert "Unavailable logs:" in captured.err
        assert "site-2: client log stream not available for this job" in captured.err

    def test_logs_job_not_found_exits_1(self, capsys):
        """JobNotFound maps to JOB_NOT_FOUND, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.side_effect = JobNotFound("no such job: abc123")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_logs(_make_args())
        assert exc_info.value.code == 1

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "JOB_NOT_FOUND"

    def test_logs_connection_failed_exits_2(self, capsys):
        """NoConnection maps to CONNECTION_FAILED, exit 2."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.side_effect = NoConnection("no connection to server")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_logs(_make_args())
        assert exc_info.value.code == 2

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "CONNECTION_FAILED"
        assert envelope["exit_code"] == 2

    def test_logs_authentication_error_propagates_to_top_level_handler(self):
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.side_effect = AuthenticationError("bad cert")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(AuthenticationError):
                cmd_job_logs(_make_args())

    def test_logs_request_does_not_apply_cli_filters(self):
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {"logs": {}}

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args())

        mock_sess.get_job_logs.assert_called_once_with("abc123", target="server")

    def test_logs_client_site_calls_session_with_target(self, capsys):
        """--site with a client name is passed through to the session API."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {"logs": {"site-1": "client log\n"}}

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args(site="site-1"))

        mock_sess.get_job_logs.assert_called_once_with("abc123", target="site-1")
        data = json.loads(capsys.readouterr().out)["data"]
        assert data["target"] == "site-1"
        assert data["logs"] == {"site-1": "client log\n"}

    def test_logs_all_site_keeps_unavailable_as_partial_success(self, capsys):
        """--site all returns available logs plus unavailable sites."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {
            "logs": {"server": "server log\n", "site-1": "client log\n"},
            "unavailable": {"site-2": "client log stream not available for this job"},
        }

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args(site="all"))

        data = json.loads(capsys.readouterr().out)["data"]
        assert data["target"] == "all"
        assert data["logs"] == {"server": "server log\n", "site-1": "client log\n"}
        assert data["unavailable"] == {"site-2": "client log stream not available for this job"}

    def test_logs_specific_missing_client_exits_with_log_not_found(self, capsys):
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {
            "logs": {},
            "unavailable": {"site-1": "client log stream not available for this job"},
        }

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_logs(_make_args(site="site-1"))

        assert exc_info.value.code == 1
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "LOG_NOT_FOUND"
        assert "site-1" in envelope["message"]

    def test_logs_parser(self):
        """'logs' subparser parses job_id and open-ended --site."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["logs"]
        assert parser is not None
        args = parser.parse_args(["abc123", "--site", "server"])
        assert args.job_id == "abc123"
        assert args.site == "server"

        args = parser.parse_args(["abc123", "--site", "site-1"])
        assert args.site == "site-1"
        args = parser.parse_args(["abc123", "--site", "all"])
        assert args.site == "all"
        args = parser.parse_args(["abc123", "--sites", "server"])
        assert args.site == "server"
        args = parser.parse_args(["--sites", "server", "abc123"])
        assert args.job_id == "abc123"
        assert args.site == "server"
        with pytest.raises(SystemExit):
            parser.parse_args(["abc123", "--tail", "100"])
        with pytest.raises(SystemExit):
            parser.parse_args(["abc123", "--grep", "OOM"])
