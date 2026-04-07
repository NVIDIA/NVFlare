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
from unittest.mock import MagicMock, patch

import pytest


class TestJobMonitor:
    """Tests for nvflare job monitor command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setenv("NVFLARE_CLI_MODE", "agent")

    def _make_args(self, job_id="abc123", timeout=0, interval=2):
        args = MagicMock()
        args.job_id = job_id
        args.timeout = timeout
        args.interval = interval
        return args

    def test_monitor_finished_ok_exits_0(self, capsys):
        """FINISHED_OK: stdout is one JSON envelope, exit 0."""
        from nvflare.tool.job.job_cli import cmd_job_monitor

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = {"status": "FINISHED_OK", "job_id": "abc123"}

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_monitor(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["job_id"] == "abc123"
        assert data["data"]["status"] == "FINISHED_OK"

    def test_monitor_failed_exits_1(self, capsys):
        """FAILED: stdout is ok envelope, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_monitor

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = {"status": "FAILED", "job_id": "abc123"}

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["status"] == "FAILED"

    def test_monitor_aborted_exits_1(self, capsys):
        """ABORTED: stdout is ok envelope, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_monitor

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = {"status": "ABORTED", "job_id": "abc123"}

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(args)
        assert exc_info.value.code == 1

    def test_monitor_timeout_exits_3(self):
        """Timeout: exits 3 with TIMEOUT error code."""
        from nvflare.tool.job.job_cli import cmd_job_monitor

        args = self._make_args(timeout=1)
        mock_sess = MagicMock()
        mock_sess.wait_for_job.side_effect = TimeoutError("timed out")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(args)
        assert exc_info.value.code == 3

    def test_monitor_timeout_error_code_in_json(self, capsys):
        """Timeout: JSON envelope has TIMEOUT error_code."""
        from nvflare.tool.job.job_cli import cmd_job_monitor

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.wait_for_job.side_effect = TimeoutError("timed out")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit):
                cmd_job_monitor(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "TIMEOUT"

    def test_monitor_connection_failed_exits_2(self, capsys):
        """Connection failure: exits 2 with CONNECTION_FAILED error code."""
        from nvflare.tool.job.job_cli import cmd_job_monitor

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.wait_for_job.side_effect = Exception("connection refused")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_monitor(args)
        assert exc_info.value.code == 2

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["error_code"] == "CONNECTION_FAILED"

    def test_monitor_no_human_text_on_stdout(self, capsys):
        """No human-readable text leaks to stdout."""
        from nvflare.tool.job.job_cli import cmd_job_monitor

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.wait_for_job.return_value = {"status": "FINISHED_OK", "job_id": "abc123"}

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            cmd_job_monitor(args)

        captured = capsys.readouterr()
        # stdout must parse as JSON
        json.loads(captured.out)
        # no prose on stdout
        assert "status" not in captured.out or json.loads(captured.out)["status"] in ("ok", "error")

    def test_monitor_parser_args(self):
        """monitor parser: positional job_id, --timeout, --interval."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["monitor"]
        assert parser is not None
        args = parser.parse_args(["abc123", "--timeout", "300", "--interval", "5"])
        assert args.job_id == "abc123"
        assert args.timeout == 300
        assert args.interval == 5
