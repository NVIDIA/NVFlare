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

from nvflare.fuel.flare_api.api_spec import AuthenticationError
from nvflare.tool import cli_output


class TestSystemLogConfig:
    """Tests for nvflare system log-config command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, level=None, site="all", schema=False):
        args = MagicMock()
        args.level = level
        args.site = site
        args.schema = schema
        return args

    def _make_session(self):
        mock_sess = MagicMock()
        mock_sess.configure_site_log.return_value = None
        return mock_sess

    def test_log_level_string(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="DEBUG")
        mock_sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_log(args)

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["log_config"] == "DEBUG"
        assert data["data"]["status"] == "applied"

    def test_log_authentication_error_propagates_to_top_level_handler(self):
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="DEBUG")
        mock_sess = self._make_session()
        mock_sess.configure_site_log.side_effect = AuthenticationError("bad cert")

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            with pytest.raises(AuthenticationError):
                cmd_system_log(args)

    def test_log_missing_args_exits_4(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level=None)

        with pytest.raises(SystemExit) as exc_info:
            cmd_system_log(args)
        assert exc_info.value.code == 4

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "LOG_CONFIG_INVALID"

    def test_log_connection_failed_exits_2(self):
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="INFO")

        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("timeout")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_log(args)
        assert exc_info.value.code == 2

    def test_log_connection_failed_does_not_emit_success_when_error_output_mocked(self):
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="INFO")
        mocked_output = MagicMock()
        mocked_ok = MagicMock()

        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("timeout")):
            with patch("nvflare.tool.system.system_cli.output_error", mocked_output):
                with patch("nvflare.tool.system.system_cli.output_ok", mocked_ok):
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_system_log(args)

        assert exc_info.value.code == 2
        mocked_output.assert_called_once()
        mocked_ok.assert_not_called()

    def test_log_site_passed_to_session(self):
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="WARNING", site="site-1")
        mock_sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_log(args)

        mock_sess.configure_site_log.assert_called_once_with("WARNING", target="site-1")

    def test_log_site_in_ok_response(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="ERROR", site="server")
        mock_sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_log(args)

        data = json.loads(capsys.readouterr().out)
        assert data["data"]["site"] == "server"

    def test_log_schema_uses_log_config_name(self, monkeypatch, capsys):
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args()
        monkeypatch.setattr("sys.argv", ["nvflare", "system", "log-config", "DEBUG", "--schema"])

        with pytest.raises(SystemExit) as exc_info:
            cmd_system_log(args)

        assert exc_info.value.code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["command"] == "nvflare system log-config"


class TestSystemLogConfigHuman:
    @pytest.fixture(autouse=True)
    def text_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")

    def _make_args(self, level=None, site="all", schema=False):
        args = MagicMock()
        args.level = level
        args.site = site
        args.schema = schema
        return args

    def test_log_missing_args_prints_structured_error(self, capsys):
        import argparse

        from nvflare.tool.system.system_cli import cmd_system_log, def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)

        args = self._make_args(level=None)

        with pytest.raises(SystemExit) as exc_info:
            cmd_system_log(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "specify a log level or mode" in captured.err
        assert "Code: LOG_CONFIG_INVALID (exit 4)" in captured.err
        assert "usage:" in captured.err
