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


def _make_args(job_id="abc123", site="server", tail=None, grep=None):
    args = MagicMock()
    args.job_id = job_id
    args.site = site
    args.tail = tail
    args.grep = grep
    return args


class TestJobLogs:
    """Tests for nvflare job logs command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _fake_session(self, mock_sess):
        @contextmanager
        def _ctx():
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

    def test_logs_tail_passed_to_session(self):
        """--tail value is forwarded as tail_lines to get_job_logs."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {"logs": {}}

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args(tail=50))

        mock_sess.get_job_logs.assert_called_once_with(
            "abc123",
            target="server",
            tail_lines=50,
            grep_pattern=None,
        )

    def test_logs_grep_passed_to_session(self):
        """--grep value is forwarded as grep_pattern to get_job_logs."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {"logs": {}}

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args(grep="ERROR"))

        mock_sess.get_job_logs.assert_called_once_with(
            "abc123",
            target="server",
            tail_lines=None,
            grep_pattern="ERROR",
        )

    def test_logs_multi_word_grep_passed_without_shell_quotes(self):
        """Multi-word grep patterns should be passed through literally, not shell-quoted."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        mock_sess = MagicMock()
        mock_sess.get_job_logs.return_value = {"logs": {}}

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_logs(_make_args(grep="CUDA out of memory"))

        mock_sess.get_job_logs.assert_called_once_with(
            "abc123",
            target="server",
            tail_lines=None,
            grep_pattern="CUDA out of memory",
        )

    def test_logs_non_server_site_rejected(self, capsys):
        """--site with a non-server value → INVALID_ARGS, exits 4."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        with pytest.raises(SystemExit) as exc_info:
            cmd_job_logs(_make_args(site="site-1"))
        assert exc_info.value.code == 4

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "INVALID_ARGS"

    def test_logs_all_site_rejected(self, capsys):
        """--site all → INVALID_ARGS, exits 4 (client streaming not yet available)."""
        from nvflare.tool.job.job_cli import cmd_job_logs

        with pytest.raises(SystemExit) as exc_info:
            cmd_job_logs(_make_args(site="all"))
        assert exc_info.value.code == 4

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "INVALID_ARGS"

    def test_logs_parser(self):
        """'logs' subparser parses job_id, server-only --site, --tail, and --grep correctly."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["logs"]
        assert parser is not None
        args = parser.parse_args(["abc123", "--site", "server", "--tail", "100", "--grep", "OOM"])
        assert args.job_id == "abc123"
        assert args.site == "server"
        assert args.tail == 100
        assert args.grep == "OOM"

        with pytest.raises(SystemExit):
            parser.parse_args(["abc123", "--site", "all"])
