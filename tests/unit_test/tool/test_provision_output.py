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


class TestProvisionOutput:
    """Tests for nvflare provision output format."""

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
        """No-arg default: stdout is exactly one JSON line; progress text goes to stderr."""
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
        assert "workspace" in data["data"]
        assert "project_yml" in data["data"]

        # copy_project is mocked so its print_human never fires;
        # the contract is enforced by stdout being clean (asserted above)

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
        """When -p project.yml is given, provisioning runs; progress goes to stderr only."""
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
        # progress ("Project yaml file: ...") must be on stderr, not stdout
        assert "Project yaml file" in captured.err
        # stdout must contain only the JSON envelope
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1
        data = json.loads(stdout_lines[0])
        assert data["status"] == "ok"
