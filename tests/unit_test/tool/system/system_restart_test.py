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


class TestSystemRestart:
    """Tests for nvflare system restart command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, target="server", client_names=None, force=False):
        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        args.force = force
        return args

    def test_restart_with_force_no_prompt(self, capsys):
        """--force skips confirmation prompt; response JSON contains restart-related status."""
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.restart.return_value = "restart initiated"

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with patch("sys.stdin") as mock_stdin:
                cmd_system_restart(args)
                mock_stdin.readline.assert_not_called()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["result"] == "restart initiated"
        assert data["data"]["target"] == "server"

    def test_restart_non_interactive_without_force_exits_4(self):
        """Non-interactive mode without --force exits with code 4."""
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(force=False)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_restart(args)
        assert exc_info.value.code == 4

    def test_restart_interactive_user_cancels(self):
        """Interactive mode: user says N → restart cancelled with exit 0."""
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(force=False)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "N\n"
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_restart(args)
        assert exc_info.value.code == 0

    def test_restart_interactive_user_confirms(self, capsys):
        """Interactive mode: user says Y → restart proceeds and returns ok envelope."""
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(force=False)
        mock_sess = MagicMock()
        mock_sess.restart.return_value = "restart initiated"

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "Y\n"
            with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
                cmd_system_restart(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0

    def test_restart_connection_failed_exits_2(self):
        """Connection failure exits with code 2."""
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(force=True)
        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("conn error")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_restart(args)
        assert exc_info.value.code == 2

    def test_restart_no_connection_propagates_to_top_level_handler(self):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.restart.side_effect = NoConnection("conn error")

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with pytest.raises(NoConnection):
                cmd_system_restart(args)

    def test_restart_connection_failed_does_not_emit_success_when_error_output_mocked(self):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(force=True)
        mocked_output = MagicMock()
        mocked_ok = MagicMock()

        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("conn error")):
            with patch("nvflare.tool.system.system_cli.output_error", mocked_output):
                with patch("nvflare.tool.system.system_cli.output_ok", mocked_ok):
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_system_restart(args)

        assert exc_info.value.code == 2
        mocked_output.assert_called_once()
        mocked_ok.assert_not_called()

    def test_restart_parser_accepts_all_target(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)

        args = parser.parse_args(["restart", "all", "--force"])
        assert args.system_sub_cmd == "restart"
        assert args.target == "all"
        assert args.force is True

    def test_restart_rejects_client_names_for_non_client_target(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(target="server", client_names=["site-1"], force=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_system_restart(args)
        assert exc_info.value.code == 4
