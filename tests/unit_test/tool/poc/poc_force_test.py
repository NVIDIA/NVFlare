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

from nvflare.cli_exception import CLIException
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC


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

    def test_force_prepare_stops_running_poc_before_delete(self, tmp_path):
        from nvflare.tool.poc.poc_commands import _prepare_poc

        workspace = str(tmp_path / "poc_ws_running")
        os.makedirs(workspace)

        calls = []

        def fake_stop(_workspace, *_args, **_kwargs):
            calls.append("stop")

        def fake_rmtree(path, ignore_errors=False):
            assert path == workspace
            assert ignore_errors is True
            calls.append("rmtree")

        with (
            patch(
                "nvflare.tool.poc.poc_commands.setup_service_config",
                return_value=(
                    {"name": "example_project"},
                    {
                        SC.FLARE_SERVER: "server",
                        SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
                    },
                ),
            ),
            patch("nvflare.tool.poc.poc_commands.is_poc_ready", return_value=True),
            patch(
                "nvflare.tool.poc.poc_commands.is_poc_running",
                side_effect=[True, False],
            ),
            patch("nvflare.tool.poc.poc_commands._stop_poc", side_effect=fake_stop) as mock_stop,
            patch("nvflare.tool.poc.poc_commands.shutil.rmtree", side_effect=fake_rmtree),
            patch(
                "nvflare.tool.poc.poc_commands.prepare_poc_provision",
                return_value={"name": "example_project"},
            ),
            patch("nvflare.tool.poc.poc_commands.save_startup_kit_dir_config"),
            patch("nvflare.tool.cli_output.print_human"),
        ):
            result = _prepare_poc([], 2, workspace, force=True)

        assert result is True
        assert mock_stop.call_count == 1
        assert calls == ["stop", "rmtree"]

    def test_force_prepare_preserves_workspace_when_running_system_does_not_stop(self, tmp_path):
        from nvflare.tool.poc.poc_commands import _prepare_poc

        workspace = tmp_path / "poc_ws_stuck"
        workspace.mkdir()
        sentinel = workspace / "keep.txt"
        sentinel.write_text("keep me")

        with (
            patch(
                "nvflare.tool.poc.poc_commands.setup_service_config",
                return_value=(
                    {"name": "example_project"},
                    {
                        SC.FLARE_SERVER: "server",
                        SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
                    },
                ),
            ),
            patch("nvflare.tool.poc.poc_commands.is_poc_ready", return_value=True),
            patch("nvflare.tool.poc.poc_commands.is_poc_running", return_value=True),
            patch("nvflare.tool.poc.poc_commands._stop_poc"),
            patch("nvflare.tool.poc.poc_commands.time.time", side_effect=[0, 31]),
            patch("nvflare.tool.poc.poc_commands.prepare_poc_provision") as mock_prov,
        ):
            with pytest.raises(
                CLIException,
                match="system is still running after shutdown was requested",
            ):
                _prepare_poc([], 2, str(workspace), force=True)

        mock_prov.assert_not_called()
        assert workspace.exists()
        assert sentinel.exists()

    def test_prepare_without_force_raises_for_running_poc_before_prompt(self, tmp_path):
        from nvflare.tool.poc.poc_commands import _prepare_poc

        workspace = str(tmp_path / "poc_ws_running_no_force")
        os.makedirs(workspace)

        with (
            patch(
                "nvflare.tool.poc.poc_commands.setup_service_config",
                return_value=(
                    {"name": "example_project"},
                    {
                        SC.FLARE_SERVER: "server",
                        SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
                    },
                ),
            ),
            patch("nvflare.tool.poc.poc_commands.is_poc_ready", return_value=True),
            patch("nvflare.tool.poc.poc_commands.is_poc_running", return_value=True),
            patch("nvflare.tool.cli_output.prompt_yn") as mock_prompt,
            patch("nvflare.tool.poc.poc_commands._stop_poc") as mock_stop,
            patch("nvflare.tool.poc.poc_commands.prepare_poc_provision") as mock_prov,
        ):
            with pytest.raises(
                CLIException,
                match="system is still running, please stop the system first.",
            ):
                _prepare_poc([], 2, workspace, force=False)

        mock_prompt.assert_not_called()
        mock_stop.assert_not_called()
        mock_prov.assert_not_called()

    def test_force_prepare_ignores_unreadable_workspace_config(self, tmp_path):
        from nvflare.tool.poc.poc_commands import _prepare_poc

        workspace = str(tmp_path / "poc_ws_bad_yaml")
        os.makedirs(workspace)

        with (
            patch(
                "nvflare.tool.poc.poc_commands.setup_service_config",
                side_effect=Exception("bad yaml"),
            ),
            patch("nvflare.tool.poc.poc_commands.shutil.rmtree") as mock_rmtree,
            patch(
                "nvflare.tool.poc.poc_commands.prepare_poc_provision",
                return_value={"name": "example_project"},
            ) as mock_prov,
            patch("nvflare.tool.poc.poc_commands.save_startup_kit_dir_config"),
        ):
            result = _prepare_poc([], 2, workspace, force=True)

        assert result is True
        mock_rmtree.assert_called_once_with(workspace, ignore_errors=True)
        mock_prov.assert_called_once()

    def test_is_poc_running_true_when_only_daemon_pid_is_alive(self, tmp_path):
        from nvflare.tool.poc.poc_commands import is_poc_running

        workspace = tmp_path / "poc_ws_daemon_only"
        server_dir = workspace / "example_project" / "prod_00" / "server"
        server_dir.mkdir(parents=True)
        (server_dir / "daemon_pid.fl").write_text("12345")

        with patch("os.kill") as mock_kill:
            assert is_poc_running(
                str(workspace),
                {SC.FLARE_SERVER: "server"},
                {"name": "example_project"},
            )

        mock_kill.assert_called_once_with(12345, 0)

    def test_is_poc_running_false_for_stale_daemon_pid(self, tmp_path):
        from nvflare.tool.poc.poc_commands import is_poc_running

        workspace = tmp_path / "poc_ws_stale_daemon"
        server_dir = workspace / "example_project" / "prod_00" / "server"
        server_dir.mkdir(parents=True)
        (server_dir / "daemon_pid.fl").write_text("12345")

        with patch("os.kill", side_effect=OSError):
            assert not is_poc_running(
                str(workspace),
                {SC.FLARE_SERVER: "server"},
                {"name": "example_project"},
            )

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
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "N\n"
            result = _prepare_poc([], 2, workspace, force=False)
        assert result is False

    def test_save_startup_kit_dir_config_uses_poc_admin_dir(self, tmp_path):
        from nvflare.tool.poc.poc_commands import get_poc_workspace, save_startup_kit_dir_config

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
        with patch(
            "nvflare.tool.poc.poc_commands.get_or_create_hidden_nvflare_dir",
            return_value=str(tmp_path),
        ):
            with patch(
                "nvflare.tool.poc.poc_commands.get_hidden_nvflare_config_path",
                return_value=dst,
            ):
                save_startup_kit_dir_config(workspace, "example_project")

        config = CF.parse_file(dst)
        assert config.get("poc.startup_kit").endswith("/example_project/prod_00/admin@nvidia.com")
        assert config.get("poc.workspace") == workspace

        with patch(
            "nvflare.tool.poc.poc_commands.get_or_create_hidden_nvflare_dir",
            return_value=str(tmp_path),
        ):
            with patch(
                "nvflare.tool.poc.poc_commands.get_hidden_nvflare_config_path",
                return_value=dst,
            ):
                with patch.dict(os.environ, {}, clear=True):
                    assert get_poc_workspace() == workspace

    def test_save_startup_kit_dir_config_rejects_invalid_project_yaml(self, tmp_path):
        from nvflare.tool.poc.poc_commands import save_startup_kit_dir_config

        workspace = str(tmp_path / "poc_ws")
        os.makedirs(workspace, exist_ok=True)
        with open(os.path.join(workspace, "project.yml"), "w") as f:
            f.write("[]\n")

        with pytest.raises(CLIException, match="invalid or unreadable project config"):
            save_startup_kit_dir_config(workspace, "example_project")

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

        with patch(
            "nvflare.tool.poc.poc_commands.get_poc_workspace",
            return_value=str(workspace),
        ):
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
            patch(
                "nvflare.tool.poc.poc_commands.get_poc_workspace",
                return_value=str(tmp_path),
            ),
            patch(
                "nvflare.tool.poc.poc_commands._prepare_jobs_dir",
                side_effect=Exception("boom"),
            ),
            patch("nvflare.tool.cli_output.output_error"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                prepare_jobs_dir(args)

        assert exc_info.value.code == 5

    def test_prepare_jobs_dir_replaces_existing_empty_symlink(self, tmp_path):
        from nvflare.tool.poc.poc_commands import _prepare_jobs_dir

        workspace = tmp_path / "workspace"
        jobs_src = tmp_path / "jobs_src"
        old_jobs = tmp_path / "old_jobs"
        jobs_src.mkdir()
        old_jobs.mkdir()

        admin_name = "admin@nvidia.com"
        transfer_name = "transfer"
        console_dir = workspace / "proj" / "prod_00" / admin_name
        startup_dir = console_dir / SC.STARTUP
        dst = console_dir / transfer_name
        startup_dir.mkdir(parents=True)
        os.symlink(old_jobs, dst)

        project_config = {"name": "proj"}
        service_config = {SC.FLARE_PROJ_ADMIN: admin_name}

        with patch("nvflare.tool.poc.poc_commands.get_upload_dir", return_value=transfer_name):
            result = _prepare_jobs_dir(
                str(jobs_src),
                str(workspace),
                config_packages=(project_config, service_config),
                force=True,
            )

        assert result is True
        assert os.path.islink(dst)
        assert os.readlink(dst) == str(jobs_src)

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
            patch(
                "nvflare.tool.poc.poc_commands.get_poc_workspace",
                return_value=workspace,
            ),
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
            patch(
                "nvflare.tool.poc.poc_commands.get_poc_workspace",
                return_value=str(tmp_path),
            ),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch(
                "nvflare.tool.poc.poc_commands._start_poc",
                side_effect=Exception("boom"),
            ),
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
            patch(
                "nvflare.tool.poc.poc_commands.get_poc_workspace",
                return_value=str(tmp_path),
            ),
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
            patch(
                "nvflare.tool.poc.poc_commands.get_poc_workspace",
                return_value=str(tmp_path),
            ),
            patch(
                "nvflare.tool.poc.poc_commands._clean_poc",
                side_effect=Exception("boom"),
            ),
            patch("nvflare.tool.cli_output.output_error"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                clean_poc(args)

        assert exc_info.value.code == 5
