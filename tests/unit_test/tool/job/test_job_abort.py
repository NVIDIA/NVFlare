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


class TestJobAbort:
    """Tests for nvflare job abort command."""

    def _make_args(self, job_id="abc123", output="json", force=False):
        args = MagicMock()
        args.job_id = job_id
        args.output = output
        args.force = force
        return args

    def test_abort_with_force_no_prompt(self, capsys):
        """--force skips confirmation prompt."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.abort_job.return_value = None

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with patch("builtins.input") as mock_input:
                cmd_job_abort(args)
                mock_input.assert_not_called()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["status"] == "ABORTED"

    def test_abort_non_interactive_without_force_exits_4(self):
        """Non-interactive without --force exits 4."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_abort(args)
        assert exc_info.value.code == 4

    def test_abort_interactive_user_confirms(self, capsys):
        """Interactive mode: user says Y → abort proceeds."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=False)
        mock_sess = MagicMock()
        mock_sess.abort_job.return_value = None

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", return_value="Y"):
                with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
                    cmd_job_abort(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"

    def test_abort_interactive_user_cancels(self, capsys):
        """Interactive mode: user says N → abort cancelled."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=False)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with patch("builtins.input", return_value="N"):
                cmd_job_abort(args)

        captured = capsys.readouterr()
        # Nothing to stdout (no JSON output)
        assert captured.out.strip() == "" or "Aborted" in captured.out

    def test_abort_job_not_found_exits_1(self):
        """JOB_NOT_FOUND exits with code 1."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.abort_job.side_effect = Exception("job not found")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_abort(args)
        assert exc_info.value.code == 1

    def test_abort_job_not_running_exits_1(self):
        """JOB_NOT_RUNNING exits with code 1 when job is not active."""
        from nvflare.tool.job.job_cli import cmd_job_abort

        args = self._make_args(force=True)
        mock_sess = MagicMock()
        mock_sess.abort_job.side_effect = Exception("job is not running")

        with patch("nvflare.tool.job.job_cli._get_session", return_value=mock_sess):
            with pytest.raises(SystemExit) as exc_info:
                cmd_job_abort(args)
        assert exc_info.value.code == 1

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
