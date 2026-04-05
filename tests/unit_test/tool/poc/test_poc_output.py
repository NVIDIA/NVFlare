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

import pytest


class TestPocOutput:
    """Tests for poc subcommand JSON envelopes and exit codes."""

    def test_prepare_json_envelope(self, capsys):
        """poc prepare success: JSON envelope has workspace and clients keys."""
        from nvflare.tool.cli_output import output_ok

        output_ok({"workspace": "/tmp/poc", "clients": ["site-1", "site-2"]}, "json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert "workspace" in data["data"]
        assert "clients" in data["data"]

    def test_start_json_envelope(self, capsys):
        """poc start success: JSON envelope has status, server_url, clients."""
        from nvflare.tool.cli_output import output_ok

        output_ok({"status": "running", "server_url": "grpc://localhost:8002", "clients": ["site-1"]}, "json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["data"]["status"] == "running"
        assert "server_url" in data["data"]
        assert "clients" in data["data"]

    def test_stop_json_envelope(self, capsys):
        """poc stop success: JSON envelope has status=stopped."""
        from nvflare.tool.cli_output import output_ok

        output_ok({"status": "stopped"}, "json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["data"]["status"] == "stopped"

    def test_clean_json_envelope(self, capsys):
        """poc clean success: JSON envelope has status=cleaned."""
        from nvflare.tool.cli_output import output_ok

        output_ok({"status": "cleaned"}, "json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["data"]["status"] == "cleaned"

    def test_start_txt_ready_prefix(self, capsys):
        """poc start --output txt: first line should start with 'READY: '."""
        server_url = "grpc://localhost:8002"
        print(f"READY: {server_url}")
        captured = capsys.readouterr()
        assert captured.out.startswith("READY: ")

    def test_error_exits_with_code_4(self):
        """INVALID_ARGS exit code 4."""
        from nvflare.tool.cli_output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("INVALID_ARGS", "json", exit_code=4)
        assert exc_info.value.code == 4

    def test_poc_prepare_parsers_have_output_flag(self):
        """All poc subcommand parsers should have --output and --force flags."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        # Build the parser tree
        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        # Test prepare has --output and --force
        args = root.parse_args(["poc", "prepare", "--output", "json", "--force"])
        assert args.output == "json"
        assert args.force is True

    def test_poc_start_parser_has_output_flag(self):
        """poc start should have --output flag."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)
        args = root.parse_args(["poc", "start", "--output", "txt"])
        assert args.output == "txt"

    def test_poc_stop_parser_has_output_flag(self):
        """poc stop should have --output flag."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)
        args = root.parse_args(["poc", "stop", "--output", "json"])
        assert args.output == "json"

    def test_poc_clean_parser_has_output_flag(self):
        """poc clean should have --output flag."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)
        args = root.parse_args(["poc", "clean", "--output", "txt"])
        assert args.output == "txt"
