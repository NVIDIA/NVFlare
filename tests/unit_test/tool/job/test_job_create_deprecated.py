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
