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

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import NoConnection
from nvflare.tool import cli_output


class TestSystemShutdown:
    """Tests for nvflare system shutdown command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, target="server", client_names=None, force=False, output="json"):
        from nvflare.tool.system.system_cli import _DEFAULT_SYSTEM_STATE_CHANGE_TIMEOUT

        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        args.force = force
        args.output = output
        args.timeout = _DEFAULT_SYSTEM_STATE_CHANGE_TIMEOUT
        return args

    def test_shutdown_with_force_no_prompt(self, capsys):
        """--force skips confirmation prompt."""
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.shutdown.return_value = None

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch("sys.stdin") as mock_stdin:
                cmd_system_shutdown(args)
                mock_stdin.readline.assert_not_called()

        mock_sess.shutdown.assert_called_once_with("server", client_names=None, timeout=30.0)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["status"] == "stopped"
        assert data["data"]["result"] is None

    def test_shutdown_no_wait_reports_initiated(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=True)
        args.no_wait = True
        mock_sess = MagicMock()
        mock_sess.shutdown.return_value = None

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_shutdown(args)

        mock_sess.shutdown.assert_called_once_with("server", client_names=None, wait=False)
        data = json.loads(capsys.readouterr().out)
        assert data["data"]["status"] == "shutdown initiated"

    def test_shutdown_preserves_server_reply(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.shutdown.return_value = {"status": "partial", "client_status": {"site-1": "offline"}}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_shutdown(args)

        data = json.loads(capsys.readouterr().out)
        assert data["data"]["result"] == {"status": "partial", "client_status": {"site-1": "offline"}}

    def test_shutdown_non_interactive_without_force_exits_4(self):
        """Non-interactive mode without --force exits 4."""
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=False)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_shutdown(args)
        assert exc_info.value.code == 4

    def test_shutdown_interactive_user_cancels(self):
        """Interactive mode: user says N → shutdown cancelled."""
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=False)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "N\n"
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_shutdown(args)
        assert exc_info.value.code == 0

    def test_shutdown_interactive_user_confirms(self, capsys):
        """Interactive mode: user says y → shutdown proceeds."""
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=False)
        mock_sess = MagicMock()
        mock_sess.shutdown.return_value = None

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "Y\n"
            with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
                cmd_system_shutdown(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0

    def test_shutdown_connection_failed_exits_2(self):
        """Connection failure exits with code 2."""
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=True)
        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("conn error")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_shutdown(args)
        assert exc_info.value.code == 2

    def test_shutdown_no_connection_propagates_to_top_level_handler(self):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.shutdown.side_effect = NoConnection("conn error")

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with pytest.raises(NoConnection):
                cmd_system_shutdown(args)

    def test_shutdown_connection_failed_does_not_emit_success_when_error_output_mocked(self):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=True)
        mocked_output = MagicMock()
        mocked_ok = MagicMock()

        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("conn error")):
            with patch("nvflare.tool.system.system_cli.output_error", mocked_output):
                with patch("nvflare.tool.system.system_cli.output_ok", mocked_ok):
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_system_shutdown(args)

        assert exc_info.value.code == 2
        mocked_output.assert_called_once()
        mocked_ok.assert_not_called()

    def test_shutdown_parser_accepts_client_target(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)

        args = parser.parse_args(["shutdown", "client", "--force"])
        assert args.system_sub_cmd == "shutdown"
        assert args.target == "client"
        assert args.force is True

    def test_shutdown_parser_accepts_no_wait(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)

        args = parser.parse_args(["shutdown", "all", "--force", "--no-wait"])
        assert args.no_wait is True

    def test_shutdown_parser_accepts_timeout(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)

        args = parser.parse_args(["shutdown", "all", "--force", "--timeout", "120"])
        assert args.timeout == 120.0
        with pytest.raises(SystemExit):
            parser.parse_args(["shutdown", "all", "--force", "--timeout", "-1"])
        with pytest.raises(SystemExit):
            parser.parse_args(["shutdown", "all", "--force", "--timeout", "0"])

    def test_shutdown_timeout_exits_timeout(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.shutdown.side_effect = TimeoutError("server did not stop")

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_shutdown(args)

        assert exc_info.value.code == 3
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "TIMEOUT"
        assert "--timeout" in data["hint"]
        assert "--no-wait" in data["hint"]

    def test_shutdown_rejects_client_names_for_non_client_target(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(target="all", client_names=["site-1"], force=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_system_shutdown(args)
        assert exc_info.value.code == 4
