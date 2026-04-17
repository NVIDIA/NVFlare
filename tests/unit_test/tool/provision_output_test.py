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

from nvflare.lighter.constants import CtxKey
from nvflare.tool import cli_output


class TestProvisionOutput:
    """Tests for nvflare provision output format."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.output = kwargs.get("output", "json")
        args.force = kwargs.get("force", False)
        args.generate = kwargs.get("generate", False)
        args.gen_edge = kwargs.get("gen_edge", False)
        args.project_file = kwargs.get("project_file", None)
        args.workspace = kwargs.get("workspace", "workspace")
        args.custom_folder = kwargs.get("custom_folder", ".")
        args.add_user = kwargs.get("add_user", "")
        args.add_client = kwargs.get("add_client", "")
        args.gen_scripts = kwargs.get("gen_scripts", False)
        return args

    def test_json_envelope_on_generate(self, capsys, tmp_path):
        """No-arg default: stdout is exactly one JSON line with no human text."""
        from nvflare.lighter.provision import handle_provision

        args = self._make_args()  # no project_file -> generate mode

        with patch("nvflare.lighter.provision.copy_project"):
            with patch("nvflare.lighter.provision.os.getcwd", return_value=str(tmp_path)):
                with patch("nvflare.tool.install_skills.install_skills"):
                    handle_provision(args)

        captured = capsys.readouterr()

        # stdout: exactly one JSON line, nothing else
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1, f"Expected one JSON line on stdout, got: {captured.out!r}"
        data = json.loads(stdout_lines[0])
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert "workspace" in data["data"]
        assert "project_yml" in data["data"]
        assert data["data"]["message"] == "Sample project file generated."
        assert data["data"]["next_step"] == "Edit the project file, then run provisioning."
        assert data["data"]["suggested_command"] == "nvflare provision -p project.yml"

        assert captured.err == ""

    def test_json_envelope_on_generate_includes_guidance_for_edge_project(self, capsys, tmp_path):
        """Edge project generation includes next-step guidance in the JSON payload."""
        from nvflare.lighter.provision import handle_provision

        args = self._make_args(gen_edge=True)

        with patch("nvflare.lighter.provision.copy_project"):
            with patch("nvflare.lighter.provision.os.getcwd", return_value=str(tmp_path)):
                with patch("nvflare.tool.install_skills.install_skills"):
                    handle_provision(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["data"]["message"] == "Sample project file generated."
        assert data["data"]["next_step"] == "Edit the project file, then run provisioning."
        assert data["data"]["suggested_command"] == "nvflare provision -p project.yml"
        assert captured.err == ""

    def test_provision_parser_has_no_required_group(self):
        """provision parser should not require -g/-p/-e — default is generate."""
        import argparse

        from nvflare.lighter.provision import define_provision_parser

        parser = argparse.ArgumentParser()
        define_provision_parser(parser)
        # Should parse with NO required flags (old behavior had required=True)
        args = parser.parse_args([])
        assert args.project_file is None
        assert args.generate is False
        assert args.gen_edge is False

    def test_provision_parser_force_flag(self):
        """provision parser should have --force flag."""
        import argparse

        from nvflare.lighter.provision import define_provision_parser

        parser = argparse.ArgumentParser()
        define_provision_parser(parser)
        args = parser.parse_args(["--force"])
        assert args.force is True

    def test_install_skills_called_on_success(self, capsys, tmp_path):
        """install_skills should be called (and its failure ignored) on success."""
        from nvflare.lighter.provision import handle_provision

        args = self._make_args()
        install_called = []

        with patch("nvflare.lighter.provision.copy_project"):
            with patch("nvflare.lighter.provision.os.getcwd", return_value=str(tmp_path)):
                with patch("nvflare.tool.install_skills.install_skills", side_effect=lambda: install_called.append(1)):
                    handle_provision(args)

        assert len(install_called) == 1

    def test_project_file_runs_provisioning(self, capsys, tmp_path):
        """When -p project.yml is given in JSON mode, no human progress text is emitted."""
        from nvflare.lighter.provision import handle_provision

        args = self._make_args(project_file="project.yml")

        with patch("nvflare.lighter.provision.os.path.join", side_effect=lambda *a: "/".join(a)):
            with patch("nvflare.lighter.provision.provision") as mock_prov:
                with patch("nvflare.lighter.provision.os.getcwd", return_value=str(tmp_path)):
                    with patch("nvflare.tool.install_skills.install_skills"):
                        with patch("nvflare.lighter.provision.os.path.isdir", return_value=False):
                            handle_provision(args)

        mock_prov.assert_called_once()

        captured = capsys.readouterr()
        assert captured.err == ""
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1
        data = json.loads(stdout_lines[0])
        assert data["status"] == "ok"
        assert data["exit_code"] == 0

    def test_edge_mode_failure_returns_structured_error(self, capsys, tmp_path):
        from nvflare.lighter.provision import handle_provision

        args = self._make_args(project_file="project.yml")
        project_dict = {"edge": {"enabled": True}, "gen_scripts": False}

        with patch("nvflare.lighter.provision.load_yaml", return_value=project_dict):
            with patch("nvflare.lighter.provision.os.getcwd", return_value=str(tmp_path)):
                with patch("nvflare.lighter.provision.provision_for_edge", side_effect=RuntimeError("boom")):
                    with pytest.raises(SystemExit) as exc_info:
                        handle_provision(args)

        assert exc_info.value.code == 5
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["error_code"] == "INTERNAL_ERROR"
        assert "Provisioning failed in edge mode: boom" in data["message"]

    def test_build_error_ctx_returns_structured_error_with_ctx_diagnostics(self, capsys, tmp_path):
        from nvflare.lighter.provision import handle_provision

        args = self._make_args(project_file="project.yml")
        fake_ctx = {
            CtxKey.BUILD_ERROR: True,
            CtxKey.ERRORS: ["Exception boom raised during provision.  Incomplete prod_n folder removed."],
            CtxKey.WARNINGS: ["the connect_to.host 'bad-host' may be invalid: bad name"],
        }

        with patch("nvflare.lighter.provision.os.getcwd", return_value=str(tmp_path)):
            with patch("nvflare.lighter.provision.provision", return_value=fake_ctx):
                with pytest.raises(SystemExit) as exc_info:
                    handle_provision(args)

        assert exc_info.value.code == 5
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["error_code"] == "INTERNAL_ERROR"
        assert "Errors:" in data["message"]
        assert "Warnings:" in data["message"]

    def test_copy_project_suppresses_human_text_in_json_mode(self, capsys, tmp_path):
        """Generating a sample project in JSON mode should not emit human guidance."""
        from nvflare.lighter.provision import copy_project

        dest = tmp_path / "project.yml"
        copy_project("dummy_project.yml", str(dest))

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
