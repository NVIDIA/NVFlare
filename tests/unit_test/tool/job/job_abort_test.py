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

from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound, JobNotRunning
from nvflare.tool import cli_output


class TestJobAbort:
    """Tests for nvflare job abort command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, job_id="abc123", output="json", force=False):
        args = MagicMock()
        args.job_id = job_id
        args.output = output
        args.force = force
        return args

    def test_abort_with_force_no_prompt(self, capsys):
        """--force skips confirmation prompt; stdout is one JSON line."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.abort_job.return_value = None

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with patch("sys.stdin") as mock_stdin:
                cmd_job_abort(args)
                # readline must not be called when --force is set
                mock_stdin.readline.assert_not_called()

        captured = capsys.readouterr()
        # stdout: exactly one JSON line
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1
        data = json.loads(stdout_lines[0])
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["status"] == "ABORTED"
        # no JSON on stderr
        assert not captured.err.strip().startswith("{")

    def test_abort_non_interactive_without_force_exits_4(self):
        """Non-interactive without --force exits 4."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_abort(args)
        assert exc_info.value.code == 4

    def test_abort_non_interactive_still_exits_when_output_error_is_mocked(self):
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with patch("nvflare.tool.cli_output.output_error") as mocked_output_error:
                with patch("nvflare.tool.cli_output.prompt_yn") as mocked_prompt:
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_job_abort(args)

        assert exc_info.value.code == 4
        mocked_output_error.assert_called_once()
        mocked_prompt.assert_not_called()

    def test_abort_interactive_user_confirms(self, capsys):
        """Interactive mode: user says Y → abort proceeds; stdout is one JSON line."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=False)
        mock_sess = MagicMock()
        mock_sess.abort_job.return_value = None

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "Y\n"
            with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
                cmd_job_abort(args)

        captured = capsys.readouterr()
        # stdout: exactly one JSON line
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1
        data = json.loads(stdout_lines[0])
        assert data["status"] == "ok"
        assert data["exit_code"] == 0

    def test_abort_interactive_user_cancels(self, capsys):
        """Interactive mode: user says N → abort cancelled; nothing on stdout; prompt on stderr."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "N\n"
            cmd_job_abort(args)

        captured = capsys.readouterr()
        # stdout must be empty — no JSON envelope emitted when user cancels
        assert captured.out.strip() == ""
        # prompt and cancellation message go to stderr
        assert "Abort job" in captured.err
        assert "Aborted" in captured.err

    def test_abort_prompt_is_on_stderr_not_stdout(self, capsys):
        """Prompt text must appear on stderr, not stdout."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(job_id="job999", force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "N\n"
            cmd_job_abort(args)

        captured = capsys.readouterr()
        assert "job999" in captured.err
        assert "job999" not in captured.out

    def test_abort_job_not_found_exits_1(self, capsys):
        """JOB_NOT_FOUND exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.abort_job.side_effect = JobNotFound("job not found")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_abort(args)
        assert exc_info.value.code == 1
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "JOB_NOT_FOUND"
        assert "searched study 'default'" in envelope["message"]
        assert "nvflare job list --study <study_name>" in envelope["hint"]

    def test_abort_job_not_running_exits_1(self):
        """JOB_NOT_RUNNING exits with code 1 when job is not active."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.abort_job.side_effect = JobNotRunning("job is not running")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_abort(args)
        assert exc_info.value.code == 1

    def test_abort_authentication_error_propagates(self):
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.abort_job.side_effect = AuthenticationError("bad cert")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(AuthenticationError):
                cmd_job_abort(args)

    def test_abort_parser_force_flag(self):
        """abort parser should have --force flag."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["abort"]
        assert parser is not None
        args = parser.parse_args(["job123", "--force"])
        assert args.force is True
        assert args.job_id == "job123"
