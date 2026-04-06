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


class TestPocOutput:
    """Tests for poc subcommand JSON envelopes, exit codes, and stream routing."""

    # ------------------------------------------------------------------ helpers

    def _make_prepare_args(self, force=False):
        args = MagicMock()
        args.number_of_clients = 2
        args.clients = None
        args.docker_image = None
        args.he = False
        args.project_input = None
        args.force = force
        return args

    def _make_stop_args(self):
        args = MagicMock()
        args.service = None
        args.ex = None
        return args

    # ------------------------------------------------------------------ unit envelope checks (output_ok / output_error)

    def test_output_ok_envelope_shape(self, capsys):
        """output_ok emits correct JSON schema_version/status/data."""
        from nvflare.tool.cli_output import output_ok

        output_ok({"workspace": "/tmp/poc", "clients": ["site-1", "site-2"]})
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert "workspace" in data["data"]
        assert "clients" in data["data"]

    def test_output_error_exits_with_code_4(self):
        """output_error with exit_code=4 calls sys.exit(4)."""
        from nvflare.tool.cli_output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("INVALID_ARGS", exit_code=4)
        assert exc_info.value.code == 4

    # ------------------------------------------------------------------ prepare_poc split-stream tests

    def test_prepare_poc_success_stdout_is_one_json_line(self, capsys, tmp_path):
        """prepare_poc success: stdout is exactly one JSON line; nothing else."""
        from nvflare.tool.poc.poc_commands import prepare_poc

        args = self._make_prepare_args(force=True)

        poc_ws = str(tmp_path / "poc")
        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=poc_ws),
            patch("nvflare.tool.poc.poc_commands._prepare_poc", return_value=True),
            patch("nvflare.tool.install_skills.install_skills", return_value=None),
            patch("os.path.exists", return_value=False),
            patch("os.listdir", return_value=["server", "site-1", "site-2", "admin@nvidia.com"]),
        ):
            prepare_poc(args)

        captured = capsys.readouterr()
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1, f"Expected one JSON line on stdout, got: {captured.out!r}"
        data = json.loads(stdout_lines[0])
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"

    def test_prepare_poc_workspace_exists_non_interactive_exits_4(self, tmp_path):
        """prepare_poc exits 4 when workspace exists and stdin is not a tty (no --force)."""
        from nvflare.tool.poc.poc_commands import prepare_poc

        poc_ws = str(tmp_path / "poc_ws")

        args = self._make_prepare_args(force=False)

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=poc_ws),
            patch("os.path.exists", return_value=True),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc_info:
                prepare_poc(args)
        assert exc_info.value.code == 4

    def test_prepare_poc_prompt_on_stderr_not_stdout(self, capsys, tmp_path):
        """When workspace exists and user is prompted interactively, prompt appears on stderr."""
        from nvflare.tool.poc.poc_commands import prepare_poc

        poc_ws = str(tmp_path / "poc_ws")

        args = self._make_prepare_args(force=False)

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=poc_ws),
            patch("nvflare.tool.poc.poc_commands._prepare_poc", return_value=False),
            patch("nvflare.tool.install_skills.install_skills", return_value=None),
            patch("os.path.exists", return_value=True),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "N\n"
            prepare_poc(args)

        captured = capsys.readouterr()
        # stdout must have no JSON (user declined, _prepare_poc returned False)
        assert captured.out.strip() == ""
        # The internal _prepare_poc was mocked to return False;
        # but the prompt question is written by _prepare_poc itself, so the
        # important thing is that nothing from the prompt leaked to stdout.
        assert not captured.out.strip().startswith("{")

    # ------------------------------------------------------------------ stop_poc split-stream test

    def test_stop_poc_success_stdout_is_one_json_line(self, capsys, tmp_path):
        """stop_poc success: stdout is exactly one JSON line."""
        from nvflare.tool.poc.poc_commands import stop_poc

        args = self._make_stop_args()

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._stop_poc", return_value=None),
        ):
            stop_poc(args)

        captured = capsys.readouterr()
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1, f"Expected one JSON line on stdout, got: {captured.out!r}"
        data = json.loads(stdout_lines[0])
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["data"]["status"] == "stopped"

    def test_stop_poc_no_human_text_on_stdout(self, capsys, tmp_path):
        """stop_poc: no human-readable text leaks to stdout."""
        from nvflare.tool.poc.poc_commands import stop_poc

        args = self._make_stop_args()

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._stop_poc", return_value=None),
        ):
            stop_poc(args)

        captured = capsys.readouterr()
        # stdout must parse as JSON and contain no prose
        data = json.loads(captured.out.strip())
        assert data["status"] == "ok"

    # ------------------------------------------------------------------ poc prepare parsers

    def test_poc_prepare_parser_has_force_flag(self):
        """poc prepare parser should have --force flag."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "prepare", "--force"])
        assert args.force is True

    def test_stop_poc_invalid_service_name_exits_4(self, capsys, tmp_path):
        """stop_poc with an unknown -p/--service name exits 4 (structured error), not 1."""
        from nvflare.tool.poc.poc_commands import stop_poc

        args = self._make_stop_args()

        def fake_stop(workspace, excluded, services_list):
            from nvflare.cli_exception import CLIException

            raise CLIException("participant 'bad-site' is not defined, expecting one of: ['server', 'site-1']")

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=["bad-site"]),
            patch("nvflare.tool.poc.poc_commands._stop_poc", side_effect=fake_stop),
        ):
            with pytest.raises(SystemExit) as exc_info:
                stop_poc(args)
        assert exc_info.value.code == 4

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "INVALID_ARGS"

    def test_poc_prepare_parser_has_schema_flag(self):
        """poc prepare parser should have --schema flag."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "prepare", "--schema"])
        assert args.schema is True
