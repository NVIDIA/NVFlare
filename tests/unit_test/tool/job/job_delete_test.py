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

from nvflare.fuel.flare_api.api_spec import AuthenticationError, JobNotFound
from nvflare.tool import cli_output


class TestJobDelete:
    """Tests for nvflare job delete command."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, job_id="abc123", output="json", force=False):
        args = MagicMock()
        args.job_id = job_id
        args.output = output
        args.force = force
        return args

    def test_delete_with_force_no_prompt(self, capsys):
        """--force skips confirmation prompt; stdout is one JSON line."""
        from nvflare.tool.job.job_cli import cmd_job_delete

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.delete_job.return_value = {"job_id": "abc123", "submit_records_marked_deleted": 1}

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with patch("sys.stdin") as mock_stdin:
                cmd_job_delete(args)
                # readline must not be called when --force is set
                mock_stdin.readline.assert_not_called()

        captured = capsys.readouterr()
        # stdout: exactly one JSON line
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1
        data = json.loads(stdout_lines[0])
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["job_id"] == "abc123"
        assert data["data"]["submit_records_marked_deleted"] == 1
        # no JSON on stderr
        assert not captured.err.strip().startswith("{")

    def test_delete_non_interactive_without_force_exits_4(self):
        """Non-interactive without --force exits 4."""
        from nvflare.tool.job.job_cli import cmd_job_delete

        args = self._make_args(force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_delete(args)
        assert exc_info.value.code == 4

    def test_delete_non_interactive_still_exits_when_output_error_is_mocked(self):
        from nvflare.tool.job.job_cli import cmd_job_delete

        args = self._make_args(force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with patch("nvflare.tool.cli_output.output_error") as mocked_output_error:
                with patch("nvflare.tool.cli_output.prompt_yn") as mocked_prompt:
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_job_delete(args)

        assert exc_info.value.code == 4
        mocked_output_error.assert_called_once()
        mocked_prompt.assert_not_called()

    def test_delete_interactive_user_confirms(self, capsys):
        """Interactive mode: user says Y → delete proceeds; stdout is one JSON line."""
        from nvflare.tool.job.job_cli import cmd_job_delete

        args = self._make_args(force=False)
        mock_sess = MagicMock()
        mock_sess.delete_job.return_value = None

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "Y\n"
            with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
                cmd_job_delete(args)

        captured = capsys.readouterr()
        # stdout: exactly one JSON line
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1
        data = json.loads(stdout_lines[0])
        assert data["status"] == "ok"
        assert data["exit_code"] == 0

    def test_delete_interactive_user_cancels(self, capsys):
        """Interactive mode: user says N → delete cancelled; nothing on stdout; prompt on stderr."""
        from nvflare.tool.job.job_cli import cmd_job_delete

        args = self._make_args(force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "N\n"
            cmd_job_delete(args)

        captured = capsys.readouterr()
        # stdout must be empty — no JSON envelope emitted when user cancels
        assert captured.out.strip() == ""
        # prompt and cancellation message go to stderr
        assert "Delete job" in captured.err
        assert "Cancelled" in captured.err

    def test_delete_prompt_is_on_stderr_not_stdout(self, capsys):
        """Prompt text must appear on stderr, not stdout."""
        from nvflare.tool.job.job_cli import cmd_job_delete

        args = self._make_args(job_id="job999", force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "N\n"
            cmd_job_delete(args)

        captured = capsys.readouterr()
        assert "job999" in captured.err
        assert "job999" not in captured.out

    def test_delete_not_found_exits_1(self, capsys):
        """JOB_NOT_FOUND exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_delete

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.delete_job.side_effect = JobNotFound("job not found")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_delete(args)
        assert exc_info.value.code == 1
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "JOB_NOT_FOUND"
        assert "searched study 'default'" in envelope["message"]
        assert "nvflare job list --study <study_name>" in envelope["hint"]

    def test_delete_authentication_error_propagates(self):
        from nvflare.tool.job.job_cli import cmd_job_delete

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.delete_job.side_effect = AuthenticationError("bad cert")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(AuthenticationError):
                cmd_job_delete(args)

    def test_delete_parser_force_flag(self):
        """delete parser should have --force flag."""
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        parser = job_sub_cmd_parser["delete"]
        assert parser is not None
        args = parser.parse_args(["job123", "--force"])
        assert args.force is True
        assert args.job_id == "job123"
