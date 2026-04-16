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

"""Tests for deprecated nvflare job create command."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestJobCreateDeprecated:
    """nvflare job create is deprecated; tests verify warning, hint, and --schema behavior."""

    def _make_args(self, job_folder="/tmp/myjob", template=None, force=False):
        args = MagicMock()
        args.job_folder = job_folder
        args.template = template
        args.force = force
        return args

    def test_create_prints_deprecation_warning(self, capsys):
        """create_job prints a deprecation warning via print_human before doing any work."""
        from nvflare.tool.job.job_cli import create_job

        args = self._make_args()

        with patch("sys.argv", ["nvflare", "job", "create", "--job-folder", "/tmp/myjob"]):
            with patch("nvflare.tool.job.job_cli.get_src_template", side_effect=Exception("stop early")):
                with pytest.raises(Exception):
                    create_job(args)

        captured = capsys.readouterr()
        assert "deprecated" in captured.out.lower() or "deprecated" in captured.err.lower()

    def test_create_deprecation_hint_mentions_export(self, capsys):
        """Deprecation warning mentions 'python job.py --export --export-dir' as replacement."""
        from nvflare.tool.job.job_cli import create_job

        args = self._make_args()

        with patch("sys.argv", ["nvflare", "job", "create", "--job-folder", "/tmp/myjob"]):
            with patch("nvflare.tool.job.job_cli.get_src_template", side_effect=Exception("stop early")):
                with pytest.raises(Exception):
                    create_job(args)

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "--export" in combined

    def test_create_deprecation_hint_mentions_submit(self, capsys):
        """Deprecation warning mentions 'nvflare job submit' as the follow-up step."""
        from nvflare.tool.job.job_cli import create_job

        args = self._make_args()

        with patch("sys.argv", ["nvflare", "job", "create", "--job-folder", "/tmp/myjob"]):
            with patch("nvflare.tool.job.job_cli.get_src_template", side_effect=Exception("stop early")):
                with pytest.raises(Exception):
                    create_job(args)

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "job submit" in combined

    def test_create_schema_exits_0_with_deprecated_flag(self, capsys):
        """--schema on job create exits 0 and marks the command as deprecated."""
        import argparse

        from nvflare.tool.cli_schema import handle_schema_flag
        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        with patch("sys.argv", ["nvflare", "job", "create", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                handle_schema_flag(
                    job_sub_cmd_parser.get("create"),
                    "nvflare job create",
                    [],
                    ["job", "create", "--schema"],
                    deprecated=True,
                    deprecated_message="Use 'python job.py --export --export-dir <job_folder>' + 'nvflare job submit -j <job_folder>' instead.",
                )
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema.get("deprecated") is True
        assert "deprecated_message" in schema
        assert "--export" in schema["deprecated_message"]

    def test_create_schema_command_name(self, capsys):
        """--schema on job create reports correct command name."""
        import argparse

        from nvflare.tool.cli_schema import handle_schema_flag
        from nvflare.tool.job.job_cli import def_job_cli_parser, job_sub_cmd_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_job_cli_parser(subs)

        with patch("sys.argv", ["nvflare", "job", "create", "--schema"]):
            with pytest.raises(SystemExit):
                handle_schema_flag(
                    job_sub_cmd_parser.get("create"),
                    "nvflare job create",
                    [],
                    ["job", "create", "--schema"],
                    deprecated=True,
                    deprecated_message="Use 'python job.py --export --export-dir <job_folder>' instead.",
                )

        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema["command"] == "nvflare job create"

    def test_job_help_lists_deprecated_subcommands(self):
        import argparse

        from nvflare.tool.job.job_cli import def_job_cli_parser

        root = argparse.ArgumentParser(prog="nvflare")
        subs = root.add_subparsers()
        parser = def_job_cli_parser(subs)["job"]

        help_text = parser.format_help()
        assert "list_templates" in help_text
        assert "show_variables" in help_text
        assert "[DEPRECATED]" in help_text

    @pytest.mark.parametrize(
        ("handler_name", "parser_key", "expected_detail"),
        [
            ("create_job", "create", "bad create"),
            ("show_variables", "show_variables", "required job folder is not specified."),
            ("list_templates", "list_templates", "bad list"),
        ],
    )
    def test_deprecated_handlers_use_their_own_parser_for_usage_errors(
        self, handler_name, parser_key, expected_detail, monkeypatch
    ):
        import argparse

        from nvflare.tool.job import job_cli

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        job_cli.def_job_cli_parser(subs)

        args = MagicMock()
        args.debug = False
        args.job_folder = "/tmp/myjob"
        args.job_templates_dir = None

        output_usage_error = MagicMock(side_effect=SystemExit(4))
        monkeypatch.setattr("nvflare.tool.cli_output.output_usage_error", output_usage_error)
        monkeypatch.setattr("sys.argv", ["nvflare", "job", parser_key])

        if handler_name == "create_job":
            args.template = "missing"
            monkeypatch.setattr(job_cli, "get_src_template", lambda _args: None)
            monkeypatch.setattr(
                job_cli, "get_src_template_by_name", lambda _args: (_ for _ in ()).throw(ValueError(expected_detail))
            )
        elif handler_name == "show_variables":
            args.job_folder = "/not/a/dir"
        else:
            monkeypatch.setattr(
                job_cli,
                "find_job_templates_location",
                lambda _path=None: (_ for _ in ()).throw(ValueError(expected_detail)),
            )

        handler = getattr(job_cli, handler_name)
        with pytest.raises(SystemExit) as exc_info:
            handler(args)

        assert exc_info.value.code == 4
        output_usage_error.assert_called_once()
        assert output_usage_error.call_args.args[0] is job_cli.job_sub_cmd_parser[parser_key]
        assert output_usage_error.call_args.kwargs["detail"] == expected_detail
        assert output_usage_error.call_args.kwargs["exit_code"] == 4
