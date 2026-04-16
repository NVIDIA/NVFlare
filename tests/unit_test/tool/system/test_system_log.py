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
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from nvflare.tool import cli_output


class TestSystemLog:
    """Tests for nvflare system log command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, level=None, config=None, site="all"):
        args = MagicMock()
        args.level = level
        args.config = config
        args.site = site
        args.schema = False
        return args

    def _make_session(self):
        mock_sess = MagicMock()
        mock_sess.configure_site_log.return_value = None
        return mock_sess

    def test_log_level_string(self, capsys):
        """level='DEBUG' with no --config → ok envelope with log_config=='DEBUG'."""
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="DEBUG")
        mock_sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_log(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["log_config"] == "DEBUG"
        assert data["data"]["status"] == "applied"

    def test_log_level_inline_json(self, capsys):
        """--config with inline JSON dict → ok envelope with parsed dict as log_config."""
        from nvflare.tool.system.system_cli import cmd_system_log

        config_json = '{"version": 1}'
        args = self._make_args(config=config_json)
        mock_sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_log(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["log_config"] == {"version": 1}

    def test_log_level_file_config(self, capsys):
        """--config pointing to a JSON file → ok envelope with parsed dict."""
        from nvflare.tool.system.system_cli import cmd_system_log

        config_dict = {"version": 1, "root": {"level": "WARNING"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_dict, f)
            tmp_path = f.name

        try:
            args = self._make_args(config=tmp_path)
            mock_sess = self._make_session()

            with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
                cmd_system_log(args)

            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["status"] == "ok"
            assert data["data"]["log_config"] == config_dict
        finally:
            os.unlink(tmp_path)

    def test_log_level_missing_args_exits_4(self):
        """Neither level nor --config → INVALID_ARGS, exits 4."""
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level=None, config=None)

        with pytest.raises(SystemExit) as exc_info:
            cmd_system_log(args)
        assert exc_info.value.code == 4

    def test_log_level_missing_args_error_code(self, capsys):
        """Neither level nor --config → error_code INVALID_ARGS in envelope."""
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level=None, config=None)

        with pytest.raises(SystemExit):
            cmd_system_log(args)

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "INVALID_ARGS"

    def test_log_level_missing_args_human_mode_prints_help_and_error(self, capsys, monkeypatch):
        """Neither level nor --config in human mode should print help plus a structured error."""
        import argparse

        from nvflare.tool.system.system_cli import cmd_system_log, def_system_cli_parser

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)
        args = self._make_args(level=None, config=None)

        with pytest.raises(SystemExit) as exc_info:
            cmd_system_log(args)
        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "INVALID_ARGS" in captured.err
        assert "specify a log level or --config JSON/file" in captured.err
        assert "usage:" in captured.err

    def test_log_level_bad_json_exits_with_error(self):
        """--config with invalid JSON string (not a file path) → LOG_CONFIG_INVALID."""
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(config="not-valid-json{{{")

        with pytest.raises(SystemExit):
            cmd_system_log(args)

    def test_log_level_bad_json_error_code(self, capsys):
        """--config with invalid JSON → LOG_CONFIG_INVALID error code."""
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(config="not-valid-json{{{")

        with pytest.raises(SystemExit):
            cmd_system_log(args)

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "LOG_CONFIG_INVALID"

    def test_log_level_bad_json_file_error_code(self, capsys):
        """Malformed JSON file should also map to LOG_CONFIG_INVALID instead of INTERNAL_ERROR."""
        from nvflare.tool.system.system_cli import cmd_system_log

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            tmp_path = f.name

        try:
            args = self._make_args(config=tmp_path)

            with pytest.raises(SystemExit) as exc_info:
                cmd_system_log(args)
            assert exc_info.value.code == 1

            captured = capsys.readouterr()
            envelope = json.loads(captured.out)
            assert envelope["error_code"] == "LOG_CONFIG_INVALID"
        finally:
            os.unlink(tmp_path)

    def test_log_level_bad_json_human_mode_prints_help_and_error(self, capsys, monkeypatch):
        import argparse

        from nvflare.tool.system.system_cli import cmd_system_log, def_system_cli_parser

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)
        args = self._make_args(config="not-valid-json{{{")

        with pytest.raises(SystemExit) as exc_info:
            cmd_system_log(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.err
        assert "Log config is not valid JSON or a recognised log mode." in captured.err
        assert "Hint: Supply a valid dictConfig JSON file or one of:" in captured.err
        assert "Code: LOG_CONFIG_INVALID (exit 1)" in captured.err

    def test_log_level_connection_failed_exits_2(self):
        """Session failure → CONNECTION_FAILED, exits 2."""
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="INFO")

        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=Exception("timeout")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_log(args)
        assert exc_info.value.code == 2

    def test_log_level_site_passed_to_session(self):
        """--site value is forwarded to sess.configure_site_log as target kwarg."""
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="WARNING", site="site-1")
        mock_sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_log(args)

        mock_sess.configure_site_log.assert_called_once_with("WARNING", target="site-1")

    def test_log_level_site_in_ok_response(self, capsys):
        """site value appears in the ok response data."""
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="ERROR", site="server")
        mock_sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_log(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["data"]["site"] == "server"

    def test_log_both_level_and_config_exits_4(self, capsys):
        """Providing both level and --config → INVALID_ARGS, exits 4."""
        from nvflare.tool.system.system_cli import cmd_system_log

        args = self._make_args(level="DEBUG", config='{"version": 1}')

        with pytest.raises(SystemExit) as exc_info:
            cmd_system_log(args)
        assert exc_info.value.code == 4

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "INVALID_ARGS"


class TestSystemLogHuman:
    @pytest.fixture(autouse=True)
    def text_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")

    def _make_args(self, level=None, config=None, site="all", schema=False):
        args = MagicMock()
        args.level = level
        args.config = config
        args.site = site
        args.schema = schema
        return args

    def test_log_missing_args_prints_structured_error(self, capsys):
        import argparse

        from nvflare.tool.system.system_cli import cmd_system_log, def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)

        args = self._make_args(level=None, config=None)

        with pytest.raises(SystemExit) as exc_info:
            cmd_system_log(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "specify a log level or --config JSON/file" in captured.err
        assert "Code: INVALID_ARGS (exit 4)" in captured.err
        assert "usage:" in captured.err
