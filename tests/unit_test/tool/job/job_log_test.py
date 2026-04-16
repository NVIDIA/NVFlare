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
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import AuthenticationError, InvalidTarget, JobNotFound, NoConnection
from nvflare.tool import cli_output


def _make_args(job_id="abc123", level=None, config=None, site="all"):
    args = MagicMock()
    args.job_id = job_id
    args.level = level
    args.config = config
    args.site = site
    return args


class TestJobLog:
    """Tests for nvflare job log-config command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _fake_session(self, mock_sess):
        @contextmanager
        def _ctx():
            yield mock_sess

        return _ctx

    def _make_session(self, status="RUNNING"):
        mock_sess = MagicMock()
        mock_sess.get_job_meta.return_value = {"status": status}
        mock_sess.configure_job_log.return_value = None
        return mock_sess

    def test_log_level_string_applied(self, capsys):
        """level='DEBUG' with no --config and running job → ok envelope with config=='DEBUG'."""
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="DEBUG")
        mock_sess = self._make_session(status="RUNNING")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_log(args)

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "ok"
        assert envelope["exit_code"] == 0
        assert envelope["data"]["config"] == "DEBUG"
        assert envelope["data"]["status"] == "applied"
        assert envelope["data"]["sites"] == ["all"]

    def test_log_level_inline_json(self, capsys):
        """--config with inline JSON dict → ok envelope with parsed dict as config."""
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(config='{"version": 1}')
        mock_sess = self._make_session(status="RUNNING")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_log(args)

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "ok"
        assert envelope["data"]["config"] == {"version": 1}

    def test_log_invalid_config_exits_4(self, capsys):
        """Neither level nor --config → LOG_CONFIG_INVALID error, SystemExit with code 4."""
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level=None, config=None)

        with pytest.raises(SystemExit) as exc_info:
            cmd_job_log(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "LOG_CONFIG_INVALID"

    def test_log_both_level_and_config_exits_4(self, capsys):
        """Providing both --level and --config → INVALID_ARGS, exits 4."""
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="DEBUG", config='{"version": 1}')

        with pytest.raises(SystemExit) as exc_info:
            cmd_job_log(args)

        assert exc_info.value.code == 4
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "INVALID_ARGS"

    def test_log_bad_json_file_exits_4(self, capsys):
        """Malformed JSON file should map to LOG_CONFIG_INVALID instead of INTERNAL_ERROR."""
        from nvflare.tool.job.job_cli import cmd_job_log

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            tmp_path = f.name

        try:
            args = _make_args(config=tmp_path)

            with pytest.raises(SystemExit) as exc_info:
                cmd_job_log(args)

            assert exc_info.value.code == 4
            envelope = json.loads(capsys.readouterr().out)
            assert envelope["error_code"] == "LOG_CONFIG_INVALID"
        finally:
            os.unlink(tmp_path)


class TestJobLogHuman:
    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _fake_session(self, mock_sess):
        @contextmanager
        def _ctx():
            yield mock_sess

        return _ctx

    def _make_session(self, status="RUNNING"):
        mock_sess = MagicMock()
        mock_sess.get_job_meta.return_value = {"status": status}
        mock_sess.configure_job_log.return_value = None
        return mock_sess

    def test_log_missing_level_prints_help_then_error(self, capsys, monkeypatch):
        import argparse

        from nvflare.tool.job.job_cli import cmd_job_log, define_job_log_parser

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        root = argparse.ArgumentParser(prog="nvflare")
        subs = root.add_subparsers()
        define_job_log_parser(subs)

        args = _make_args(level=None, config=None)

        with pytest.raises(SystemExit) as exc_info:
            cmd_job_log(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        assert "usage:" in captured.err
        assert "\n\nLog config is not valid JSON or a recognised log mode." in captured.err
        assert "Hint: Supply a valid dictConfig JSON file or one of:" in captured.err
        assert "Code: LOG_CONFIG_INVALID (exit 4)" in captured.err

    def test_log_job_not_running_finished_ok_exits_1(self, capsys):
        """meta status 'FINISHED_OK' → JOB_NOT_RUNNING error, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="INFO")
        mock_sess = self._make_session(status="FINISHED_OK")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_log(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "JOB_NOT_RUNNING"

    def test_log_job_not_running_aborted_exits_1(self, capsys):
        """meta status 'ABORTED' → JOB_NOT_RUNNING error, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="INFO")
        mock_sess = self._make_session(status="ABORTED")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_log(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "JOB_NOT_RUNNING"

    def test_log_job_not_found_exits_1(self, capsys):
        """JobNotFound → JOB_NOT_FOUND, exit 1."""
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="INFO")
        mock_sess = MagicMock()
        mock_sess.get_job_meta.side_effect = JobNotFound("job does not exist")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_log(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "JOB_NOT_FOUND"

    def test_log_connection_error_propagates_to_top_level_handler(self):
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="INFO")
        mock_sess = MagicMock()
        mock_sess.get_job_meta.side_effect = NoConnection("connection refused")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(NoConnection):
                cmd_job_log(args)

    def test_log_authentication_error_propagates_to_top_level_handler(self):
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="INFO")
        mock_sess = MagicMock()
        mock_sess.get_job_meta.side_effect = AuthenticationError("bad cert")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(AuthenticationError):
                cmd_job_log(args)

    def test_log_site_passed_to_session(self):
        """--site value is forwarded to sess.configure_job_log as target kwarg."""
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="WARNING", site="server")
        mock_sess = self._make_session(status="RUNNING")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_log(args)

        mock_sess.configure_job_log.assert_called_once_with("abc123", "WARNING", target="server")

    def test_log_explicit_site_reflected_in_response(self, capsys):
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="WARNING", site="server")
        mock_sess = self._make_session(status="RUNNING")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            cmd_job_log(args)

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["data"]["sites"] == ["server"]

    def test_log_unknown_site_returns_site_not_found(self, capsys):
        from nvflare.tool.job.job_cli import cmd_job_log

        args = _make_args(level="WARNING", site="site-1")
        mock_sess = MagicMock()
        mock_sess.get_job_meta.return_value = {"status": "RUNNING"}
        mock_sess.configure_job_log.side_effect = InvalidTarget("INVALID_CLIENT(s): site-1")

        with patch("nvflare.tool.job.job_cli._session", side_effect=self._fake_session(mock_sess)):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_log(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "SITE_NOT_FOUND"

    def test_log_parser(self):
        """Primary 'log-config' parser and deprecated 'log' alias both parse correctly."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers(dest="sub_command")
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["log-config"]
        assert parser is not None
        args = parser.parse_args(["abc123", "DEBUG"])
        assert args.job_id == "abc123"
        assert args.level == "DEBUG"

        alias_args = root.parse_args(["job", "log", "abc123", "DEBUG"])
        assert alias_args.sub_command == "job"
        assert alias_args.job_sub_cmd == "log"
        assert alias_args.job_id == "abc123"
        assert alias_args.level == "DEBUG"
