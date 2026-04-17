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

"""Tests that --schema works even when required positional args are absent.

cli.py uses lenient parsing (parse_known_args) when --schema is in sys.argv,
so handlers are always reached and handle_schema_flag() exits with 0 before
argparse can complain about missing required args.
"""

import json
from unittest.mock import patch

import pytest


class TestSchemaWithMissingArgs:
    """--schema exits 0 with JSON schema even when required positional args are absent."""

    def _run_schema(self, argv):
        """Patch sys.argv and invoke the CLI main entry point, capturing output."""
        with patch("sys.argv", argv):
            from nvflare import cli

            with patch("nvflare.cli.ensure_hidden_config_migrated"):
                with pytest.raises(SystemExit) as exc_info:
                    cli.main()
        return exc_info.value.code

    def test_job_abort_schema_no_job_id(self, capsys):
        """nvflare job abort --schema works without required job_id."""
        from nvflare.tool.cli_schema import handle_schema_flag
        from nvflare.tool.job.job_cli import job_sub_cmd_parser

        with patch("sys.argv", ["nvflare", "job", "abort", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                handle_schema_flag(
                    job_sub_cmd_parser.get("abort"),
                    "nvflare job abort",
                    ["nvflare job abort abc123 --force"],
                    ["job", "abort", "--schema"],
                )
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema["command"] == "nvflare job abort"
        assert "args" in schema

    def test_job_meta_schema_no_job_id(self, capsys):
        """nvflare job meta --schema works without required job_id."""
        from nvflare.tool.cli_schema import handle_schema_flag
        from nvflare.tool.job.job_cli import job_sub_cmd_parser

        with patch("sys.argv", ["nvflare", "job", "meta", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                handle_schema_flag(
                    job_sub_cmd_parser.get("meta"),
                    "nvflare job meta",
                    ["nvflare job meta abc123"],
                    ["job", "meta", "--schema"],
                )
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema["command"] == "nvflare job meta"

    def test_system_shutdown_schema_no_target(self, capsys):
        """nvflare system shutdown --schema works without required target."""
        from nvflare.tool.cli_schema import handle_schema_flag
        from nvflare.tool.system.system_cli import _system_sub_cmd_parsers

        with patch("sys.argv", ["nvflare", "system", "shutdown", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                handle_schema_flag(
                    _system_sub_cmd_parsers.get("shutdown"),
                    "nvflare system shutdown",
                    ["nvflare system shutdown all --force"],
                    ["system", "shutdown", "--schema"],
                )
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema["command"] == "nvflare system shutdown"

    def test_recipe_schema_defaults_to_list_subcommand(self, capsys):
        """nvflare recipe --schema should use the default list subcommand in fast-path parsing."""
        exit_code = self._run_schema(["nvflare", "recipe", "--schema"])
        assert exit_code == 0

        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema["command"] == "nvflare recipe list"

    def test_recipe_schema_bypasses_version_check(self, capsys):
        """--schema should work even if the runtime version gate would reject normal CLI execution."""
        with patch("sys.argv", ["nvflare", "recipe", "--schema"]):
            from nvflare import cli

            with patch("nvflare.cli.version_check", side_effect=RuntimeError("unsupported")):
                with pytest.raises(SystemExit) as exc_info:
                    cli.main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema["command"] == "nvflare recipe list"

    def test_schema_output_is_valid_json(self, capsys):
        """handle_schema_flag output is valid JSON with required top-level fields."""
        import argparse

        from nvflare.tool.cli_schema import handle_schema_flag

        parser = argparse.ArgumentParser()
        parser.add_argument("job_id")
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--schema", action="store_true")

        with patch("sys.argv", ["nvflare", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                handle_schema_flag(parser, "nvflare test cmd", ["nvflare test cmd job1"], ["--schema"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema["schema_version"] == "1"
        assert schema["command"] == "nvflare test cmd"
        assert isinstance(schema["args"], list)
        assert schema["examples"] == ["nvflare test cmd job1"]

    def test_schema_required_positional_marked_required(self, capsys):
        """Positional arg without nargs='?' is marked required=True in schema."""
        import argparse

        from nvflare.tool.cli_schema import handle_schema_flag

        parser = argparse.ArgumentParser()
        parser.add_argument("job_id", help="the job ID")
        parser.add_argument("--schema", action="store_true")

        with patch("sys.argv", ["--schema"]):
            with pytest.raises(SystemExit):
                handle_schema_flag(parser, "nvflare test", [], ["--schema"])

        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        job_id_arg = next(a for a in schema["args"] if a["name"] == "job_id")
        assert job_id_arg["required"] is True

    def test_schema_optional_positional_marked_not_required(self, capsys):
        """Positional arg with nargs='?' is marked required=False in schema."""
        import argparse

        from nvflare.tool.cli_schema import handle_schema_flag

        parser = argparse.ArgumentParser()
        parser.add_argument("target", nargs="?", help="optional target")
        parser.add_argument("--schema", action="store_true")

        with patch("sys.argv", ["--schema"]):
            with pytest.raises(SystemExit):
                handle_schema_flag(parser, "nvflare test", [], ["--schema"])

        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        target_arg = next(a for a in schema["args"] if a["name"] == "target")
        assert target_arg["required"] is False

    def test_schema_deprecated_flag_included(self, capsys):
        """deprecated=True adds deprecated and deprecated_message fields to schema."""
        import argparse

        from nvflare.tool.cli_schema import handle_schema_flag

        parser = argparse.ArgumentParser()
        parser.add_argument("--schema", action="store_true")

        with patch("sys.argv", ["--schema"]):
            with pytest.raises(SystemExit):
                handle_schema_flag(
                    parser,
                    "nvflare old cmd",
                    [],
                    ["--schema"],
                    deprecated=True,
                    deprecated_message="Use new cmd instead.",
                )

        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema.get("deprecated") is True
        assert schema.get("deprecated_message") == "Use new cmd instead."
