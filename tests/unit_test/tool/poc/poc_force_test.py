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

import os
from unittest.mock import MagicMock, patch

import pytest
from pyhocon import ConfigFactory as CF


class TestPocForce:
    """Tests for poc --force flag behavior."""

    def test_force_flag_skips_prompt_prepare(self, tmp_path):
        """With --force, prepare_poc should not prompt even if workspace exists."""
        from nvflare.tool.poc.poc_commands import _prepare_poc

        workspace = str(tmp_path / "poc_ws")
        os.makedirs(workspace)

        # With force=True: should NOT call input(), should delete and proceed
        with patch("nvflare.tool.poc.poc_commands.prepare_poc_provision") as mock_prov:
            mock_prov.return_value = {"name": "example_project", "participants": []}
            with patch("nvflare.tool.poc.poc_commands.save_startup_kit_dir_config"):
                result = _prepare_poc([], 2, workspace, force=True)
        # force=True means no prompt; result should be True (not False)
        assert result is True

    def test_no_force_non_interactive_exits_4(self, tmp_path):
        """Non-interactive mode without --force should output INVALID_ARGS exit 4."""
        from nvflare.tool.poc.poc_commands import prepare_poc

        workspace = str(tmp_path / "poc_ws")
        os.makedirs(workspace)

        cmd_args = MagicMock()
        cmd_args.output = "json"
        cmd_args.force = False
        cmd_args.clients = []
        cmd_args.number_of_clients = 2
        cmd_args.docker_image = None
        cmd_args.he = False
        cmd_args.project_input = ""

        with patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=workspace):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = False
                with pytest.raises(SystemExit) as exc_info:
                    prepare_poc(cmd_args)
        assert exc_info.value.code == 4

    def test_force_flag_in_parser(self):
        """poc prepare parser should accept --force flag."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)
        args = root.parse_args(["poc", "prepare", "--force"])
        assert args.force is True

    def test_prepare_jobs_dir_force_flag(self):
        """poc prepare-jobs-dir parser should accept --force flag."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)
        args = root.parse_args(["poc", "prepare-jobs-dir", "--force"])
        assert args.force is True

    def test_interactive_without_force_prompts(self, tmp_path):
        """In interactive mode without --force, should ask for confirmation."""
        from nvflare.tool.poc.poc_commands import _prepare_poc

        workspace = str(tmp_path / "poc_ws2")
        os.makedirs(workspace)

        # Simulate user saying "N" — prompt_yn uses sys.stdin.readline, not input()
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "N\n"
            result = _prepare_poc([], 2, workspace, force=False)
        assert result is False

    def test_save_startup_kit_dir_config_uses_poc_admin_dir(self, tmp_path):
        from nvflare.tool.poc.poc_commands import save_startup_kit_dir_config

        workspace = str(tmp_path / "poc_ws")
        os.makedirs(workspace, exist_ok=True)
        with open(os.path.join(workspace, "project.yml"), "w") as f:
            f.write(
                """
name: example_project
participants:
  - name: server
    type: server
  - name: admin@nvidia.com
    type: admin
    role: project_admin
"""
            )

        dst = str(tmp_path / "config.conf")
        with patch("nvflare.tool.poc.poc_commands.get_or_create_hidden_nvflare_config_path", return_value=dst):
            save_startup_kit_dir_config(workspace, "example_project")

        config = CF.parse_file(dst)
        assert config.get("poc.startup_kit").endswith("/example_project/prod_00/admin@nvidia.com")
        assert config.get("poc.workspace") == workspace

    def test_force_does_not_delete_workspace_before_rejecting_project_file_inside_workspace(self, tmp_path):
        from nvflare.tool.poc.poc_commands import prepare_poc

        workspace = tmp_path / "poc_ws"
        workspace.mkdir()
        project_file = workspace / "project.yml"
        project_file.write_text("name: example_project\nparticipants: []\n")
        sentinel = workspace / "keep.txt"
        sentinel.write_text("keep me")

        cmd_args = MagicMock()
        cmd_args.output = "json"
        cmd_args.force = True
        cmd_args.clients = []
        cmd_args.number_of_clients = 2
        cmd_args.docker_image = None
        cmd_args.he = False
        cmd_args.project_input = str(project_file)

        with patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(workspace)):
            with pytest.raises(SystemExit) as exc_info:
                prepare_poc(cmd_args)

        assert exc_info.value.code == 4
        assert workspace.exists()
        assert sentinel.exists()

    def test_prepare_jobs_dir_raises_when_output_error_is_mocked(self, tmp_path):
        from nvflare.tool.poc.poc_commands import prepare_jobs_dir

        args = MagicMock()
        args.jobs_dir = str(tmp_path / "jobs")
        args.force = False

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands._prepare_jobs_dir", side_effect=Exception("boom")),
            patch("nvflare.tool.cli_output.output_error"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                prepare_jobs_dir(args)

        assert exc_info.value.code == 5

    def test_prepare_poc_raises_when_output_error_is_mocked_for_noninteractive_conflict(self, tmp_path):
        from nvflare.tool.poc.poc_commands import prepare_poc

        workspace = str(tmp_path / "poc_ws")
        os.makedirs(workspace)

        cmd_args = MagicMock()
        cmd_args.output = "json"
        cmd_args.force = False
        cmd_args.clients = []
        cmd_args.number_of_clients = 2
        cmd_args.docker_image = None
        cmd_args.he = False
        cmd_args.project_input = ""

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=workspace),
            patch("sys.stdin") as mock_stdin,
            patch("nvflare.tool.cli_output.output_error"),
        ):
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc_info:
                prepare_poc(cmd_args)

        assert exc_info.value.code == 4

    def test_start_poc_raises_when_output_error_is_mocked(self, tmp_path):
        from nvflare.tool.poc.poc_commands import start_poc

        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._start_poc", side_effect=Exception("boom")),
            patch("nvflare.tool.cli_output.output_error"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                start_poc(args)

        assert exc_info.value.code == 5

    def test_stop_poc_raises_when_output_error_is_mocked(self, tmp_path):
        from nvflare.tool.poc.poc_commands import stop_poc

        args = MagicMock()
        args.service = None
        args.ex = None

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._stop_poc", side_effect=Exception("boom")),
            patch("nvflare.tool.cli_output.output_error"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                stop_poc(args)

        assert exc_info.value.code == 5

    def test_clean_poc_raises_when_output_error_is_mocked(self, tmp_path):
        from nvflare.tool.poc.poc_commands import clean_poc

        args = MagicMock()
        args.force = True

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands._clean_poc", side_effect=Exception("boom")),
            patch("nvflare.tool.cli_output.output_error"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                clean_poc(args)

        assert exc_info.value.code == 5
