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

import errno
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pyhocon import ConfigFactory as CF

from nvflare.cli_exception import CLIException
from nvflare.fuel_opt.utils.pyhocon_loader import PyhoconConfig
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC


def _load_config_dict(path):
    return PyhoconConfig(CF.parse_file(str(path))).to_dict()


def _make_admin_startup_kit(prod_dir, identity):
    startup_dir = prod_dir / identity / SC.STARTUP
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_admin.json").write_text("{}")
    (startup_dir / "client.crt").write_text("")
    (startup_dir / "rootCA.pem").write_text("")
    return prod_dir / identity


def _make_site_startup_kit(prod_dir, identity):
    startup_dir = prod_dir / identity / SC.STARTUP
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_client.json").write_text("{}")
    return prod_dir / identity


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

    def test_is_poc_running_true_for_eperm_pid(self, tmp_path):
        from nvflare.tool.poc.poc_commands import is_poc_running

        workspace = tmp_path / "poc_ws_eperm_pid"
        server_dir = workspace / "example_project" / "prod_00" / "server"
        server_dir.mkdir(parents=True)
        (server_dir / "daemon_pid.fl").write_text("12345")

        with patch("os.kill", side_effect=OSError(errno.EPERM, "permission denied")) as mock_kill:
            assert is_poc_running(
                str(workspace),
                {SC.FLARE_SERVER: "server"},
                {"name": "example_project"},
            )

        mock_kill.assert_called_once_with(12345, 0)

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

        args = root.parse_args(["poc", "clean", "--force"])
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

    def test_poc_add_user_and_site_parser(self):
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        user_args = root.parse_args(["poc", "add", "user", "lead", "bob@nvidia.com", "--org", "nvidia"])
        assert user_args.poc_sub_cmd == "add"
        assert user_args.poc_add_sub_cmd == "user"
        assert user_args.cert_role == "lead"
        assert user_args.email == "bob@nvidia.com"
        assert user_args.org == "nvidia"

        site_args = root.parse_args(["poc", "add", "site", "site-3", "--org", "nvidia"])
        assert site_args.poc_sub_cmd == "add"
        assert site_args.poc_add_sub_cmd == "site"
        assert site_args.name == "site-3"
        assert site_args.org == "nvidia"

    def test_poc_add_project_admin_guard_allows_project_admin(self):
        from nvflare.tool.poc.poc_commands import _require_poc_project_admin

        with (
            patch("nvflare.tool.poc.poc_commands.resolve_startup_kit_dir", return_value="/startup-kit"),
            patch(
                "nvflare.tool.poc.poc_commands.inspect_startup_kit_metadata",
                return_value={"identity": "admin@nvidia.com", "cert_role": "project_admin"},
            ),
        ):
            _require_poc_project_admin()

    def test_poc_add_project_admin_guard_rejects_non_project_admin(self):
        from nvflare.tool.poc.poc_commands import AuthorizationError, _require_poc_project_admin

        with (
            patch("nvflare.tool.poc.poc_commands.resolve_startup_kit_dir", return_value="/startup-kit"),
            patch(
                "nvflare.tool.poc.poc_commands.inspect_startup_kit_metadata",
                return_value={"identity": "lead@nvidia.com", "cert_role": "lead"},
            ),
        ):
            with pytest.raises(AuthorizationError, match="project_admin"):
                _require_poc_project_admin()

    def test_poc_add_project_admin_guard_rejects_missing_cert_role(self):
        from nvflare.tool.poc.poc_commands import AuthorizationError, _require_poc_project_admin

        with (
            patch("nvflare.tool.poc.poc_commands.resolve_startup_kit_dir", return_value="/startup-kit"),
            patch(
                "nvflare.tool.poc.poc_commands.inspect_startup_kit_metadata",
                return_value={"identity": "admin@nvidia.com", "cert_role": None},
            ),
        ):
            with pytest.raises(AuthorizationError, match="could not determine the certificate role"):
                _require_poc_project_admin()

    def test_poc_add_user_rejects_non_project_admin_before_mutation(self, capsys):
        from nvflare.tool.poc.poc_commands import AuthorizationError, add_poc_user

        args = MagicMock()
        args.cert_role = "lead"
        args.email = "bob@nvidia.com"
        args.org = "nvidia"
        args.force = False
        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value="/tmp/poc"),
            patch(
                "nvflare.tool.poc.poc_commands._require_poc_project_admin",
                side_effect=AuthorizationError("active identity 'lead@nvidia.com' has role 'lead'"),
            ),
            patch("nvflare.tool.poc.poc_commands._add_poc_user") as add_user,
        ):
            with pytest.raises(SystemExit) as exc_info:
                add_poc_user(args)

        assert exc_info.value.code == 1
        add_user.assert_not_called()
        captured = capsys.readouterr()
        assert "Not authorized" in captured.err
        assert "lead@nvidia.com" in captured.err

    def test_poc_add_site_rejects_non_project_admin_before_mutation(self, capsys):
        from nvflare.tool.poc.poc_commands import AuthorizationError, add_poc_site

        args = MagicMock()
        args.name = "site-3"
        args.org = "nvidia"
        args.force = False
        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value="/tmp/poc"),
            patch(
                "nvflare.tool.poc.poc_commands._require_poc_project_admin",
                side_effect=AuthorizationError("active identity 'lead@nvidia.com' has role 'lead'"),
            ),
            patch("nvflare.tool.poc.poc_commands._add_poc_site") as add_site,
        ):
            with pytest.raises(SystemExit) as exc_info:
                add_poc_site(args)

        assert exc_info.value.code == 1
        add_site.assert_not_called()
        captured = capsys.readouterr()
        assert "Not authorized" in captured.err
        assert "lead@nvidia.com" in captured.err

    def test_poc_add_user_preserves_startup_kit_resolution_hint(self, capsys):
        from nvflare.tool.kit.kit_config import StartupKitConfigError
        from nvflare.tool.poc.poc_commands import add_poc_user

        args = MagicMock()
        args.cert_role = "lead"
        args.email = "bob@nvidia.com"
        args.org = "nvidia"
        args.force = False
        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value="/tmp/poc"),
            patch(
                "nvflare.tool.poc.poc_commands._require_poc_project_admin",
                side_effect=StartupKitConfigError(
                    "active startup kit is stale", hint="Run 'nvflare config use admin@nvidia.com'"
                ),
            ),
            patch("nvflare.tool.poc.poc_commands._add_poc_user") as add_user,
        ):
            with pytest.raises(SystemExit) as exc_info:
                add_poc_user(args)

        assert exc_info.value.code == 4
        add_user.assert_not_called()
        captured = capsys.readouterr()
        assert "active startup kit is stale" in captured.err
        assert "Run 'nvflare config use admin@nvidia.com'" in captured.err

    def test_poc_add_site_preserves_startup_kit_resolution_hint(self, capsys):
        from nvflare.tool.kit.kit_config import StartupKitConfigError
        from nvflare.tool.poc.poc_commands import add_poc_site

        args = MagicMock()
        args.name = "site-3"
        args.org = "nvidia"
        args.force = False
        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value="/tmp/poc"),
            patch(
                "nvflare.tool.poc.poc_commands._require_poc_project_admin",
                side_effect=StartupKitConfigError(
                    "active startup kit is stale", hint="Run 'nvflare config use admin@nvidia.com'"
                ),
            ),
            patch("nvflare.tool.poc.poc_commands._add_poc_site") as add_site,
        ):
            with pytest.raises(SystemExit) as exc_info:
                add_poc_site(args)

        assert exc_info.value.code == 4
        add_site.assert_not_called()
        captured = capsys.readouterr()
        assert "active startup kit is stale" in captured.err
        assert "Run 'nvflare config use admin@nvidia.com'" in captured.err

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

    def test_poc_prepare_registers_admin_kits_and_active_project_admin(self, tmp_path, monkeypatch):
        from nvflare.tool.poc.poc_commands import _prepare_poc

        monkeypatch.setenv("HOME", str(tmp_path))
        workspace = tmp_path / "poc_ws"
        project_name = "example_project"
        prod_dir = workspace / project_name / "prod_00"
        project_config = {
            "name": project_name,
            "participants": [
                {"name": "server", "type": "server"},
                {"name": "site-1", "type": "client"},
                {"name": "lead@nvidia.com", "type": "admin", "role": "lead"},
                {"name": "org-admin@nvidia.com", "type": "admin", "role": "org_admin"},
                {"name": "admin@nvidia.com", "type": "admin", "role": "project_admin"},
            ],
        }

        def fake_prepare_poc_provision(*_args, **_kwargs):
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "project.yml").write_text(
                """
name: example_project
participants:
  - name: server
    type: server
  - name: site-1
    type: client
  - name: lead@nvidia.com
    type: admin
    role: lead
  - name: org-admin@nvidia.com
    type: admin
    role: org_admin
  - name: admin@nvidia.com
    type: admin
    role: project_admin
"""
            )
            for identity in ("admin@nvidia.com", "org-admin@nvidia.com", "lead@nvidia.com"):
                _make_admin_startup_kit(prod_dir, identity)
            site_startup = prod_dir / "site-1" / SC.STARTUP
            site_startup.mkdir(parents=True)
            (site_startup / "fed_client.json").write_text("{}")
            return project_config

        dst = tmp_path / ".nvflare" / "config.conf"
        with (
            patch("nvflare.tool.poc.poc_commands.prepare_poc_provision", side_effect=fake_prepare_poc_provision),
            patch(
                "nvflare.tool.poc.poc_commands.get_or_create_hidden_nvflare_dir",
                return_value=str(tmp_path),
            ),
            patch(
                "nvflare.tool.poc.poc_commands.get_hidden_nvflare_config_path",
                return_value=str(dst),
            ),
        ):
            assert _prepare_poc([], 2, str(workspace), force=True) is True

        config = _load_config_dict(dst)
        entries = config["startup_kits"]["entries"]
        assert config["startup_kits"]["active"] == "admin@nvidia.com"
        assert entries == {
            "admin@nvidia.com": str(prod_dir / "admin@nvidia.com"),
            "org-admin@nvidia.com": str(prod_dir / "org-admin@nvidia.com"),
            "lead@nvidia.com": str(prod_dir / "lead@nvidia.com"),
        }
        assert config["poc"]["workspace"] == str(workspace)
        assert "startup_kit" not in config["poc"]
        assert "prod" not in config or "startup_kit" not in config["prod"]

    def test_dynamic_poc_project_config_contains_only_server_and_new_participant(self):
        from nvflare.tool.poc.poc_commands import _dynamic_poc_project_config

        project_config = {
            "api_version": 4,
            "name": "example_project",
            "participants": [
                {"name": "server", "type": "server", "org": "nvidia"},
                {"name": "site-1", "type": "client", "org": "nvidia"},
                {"name": "admin@nvidia.com", "type": "admin", "role": "project_admin", "org": "nvidia"},
            ],
            "studies": {"study-a": {"site_orgs": {"nvidia": ["site-1"]}}},
        }
        participant = {"name": "site-3", "type": "client", "org": "nvidia"}

        dynamic_config = _dynamic_poc_project_config(project_config, participant)

        assert dynamic_config["participants"] == [
            {"name": "server", "type": "server", "org": "nvidia"},
            {"name": "site-3", "type": "client", "org": "nvidia"},
        ]
        assert "studies" not in dynamic_config
        assert project_config["participants"][1]["name"] == "site-1"

    def test_provision_poc_participant_only_moves_new_kit_into_existing_prod_dir(self, tmp_path, monkeypatch):
        from nvflare.lighter.constants import CtxKey
        from nvflare.tool.poc import poc_commands

        workspace = tmp_path / "poc_ws"
        project_name = "example_project"
        prod_00 = workspace / project_name / "prod_00"
        state_dir = workspace / project_name / "state"
        state_dir.mkdir(parents=True)
        (state_dir / "cert.json").write_text("{}")
        server_startup = prod_00 / "server" / SC.STARTUP
        server_startup.mkdir(parents=True)
        (server_startup / "rootCA.pem").write_text("root ca")
        _make_admin_startup_kit(prod_00, "admin@nvidia.com")
        _make_site_startup_kit(prod_00, "site-1")
        existing_admin_marker = prod_00 / "admin@nvidia.com" / "marker.txt"
        existing_admin_marker.write_text("unchanged")

        project_config = {
            "api_version": 3,
            "name": project_name,
            "participants": [
                {"name": "server", "type": "server", "org": "nvidia"},
                {"name": "site-1", "type": "client", "org": "nvidia"},
                {"name": "admin@nvidia.com", "type": "admin", "role": "project_admin", "org": "nvidia"},
                {"name": "site-3", "type": "client", "org": "nvidia"},
            ],
        }
        participant = {"name": "site-3", "type": "client", "org": "nvidia"}
        captured_dynamic_config = {}

        def fake_prepare_project(dynamic_config):
            captured_dynamic_config.update(dynamic_config)
            return object()

        class FakeProvisioner:
            def __init__(self, root_dir, builders, packager):
                assert root_dir == str(workspace)
                assert builders == ["builder"]
                assert packager == "packager"

            def provision(self, project, mode=None, logger=None):
                prod_01 = workspace / project_name / "prod_01"
                _make_site_startup_kit(prod_01, "site-3")
                (prod_01 / "server").mkdir(parents=True)
                return {CtxKey.CURRENT_PROD_DIR: str(prod_01)}

        monkeypatch.setattr(poc_commands, "prepare_project", fake_prepare_project)
        monkeypatch.setattr(poc_commands, "prepare_builders", lambda dynamic_config: ["builder"])
        monkeypatch.setattr(poc_commands, "prepare_packager", lambda dynamic_config: "packager")
        monkeypatch.setattr(poc_commands, "Provisioner", FakeProvisioner)

        result = poc_commands._provision_poc_participant_only(str(workspace), project_config, participant, str(prod_00))

        assert result == str(prod_00 / "site-3")
        assert (prod_00 / "site-3" / SC.STARTUP / "fed_client.json").is_file()
        assert not (workspace / project_name / "prod_01").exists()
        assert existing_admin_marker.read_text() == "unchanged"
        assert captured_dynamic_config["participants"] == [
            {"name": "server", "type": "server", "org": "nvidia"},
            {"name": "site-3", "type": "client", "org": "nvidia"},
        ]

    def test_poc_add_user_persists_project_and_registers_new_admin_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.poc.poc_commands import _add_poc_user

        monkeypatch.setenv("HOME", str(tmp_path))
        workspace = tmp_path / "poc_ws"
        project_name = "example_project"
        prod_00 = workspace / project_name / "prod_00"
        _make_admin_startup_kit(prod_00, "admin@nvidia.com")
        _make_site_startup_kit(prod_00, "site-1")
        workspace.mkdir(parents=True, exist_ok=True)
        project_file = workspace / "project.yml"
        project_file.write_text(
            """
name: example_project
participants:
  - name: server
    type: server
  - name: site-1
    type: client
    org: nvidia
  - name: admin@nvidia.com
    type: admin
    role: project_admin
    org: nvidia
"""
        )
        config_path = tmp_path / ".nvflare" / "config.conf"
        config_path.parent.mkdir()
        config_path.write_text(
            f"""
version = 2
startup_kits {{
  active = "admin@nvidia.com"
  entries {{
    "admin@nvidia.com" = "{prod_00 / 'admin@nvidia.com'}"
  }}
}}
poc {{
  workspace = "{workspace}"
}}
"""
        )

        def fake_provision_participant_only(workspace_arg, project_config_arg, participant, target_prod_dir, force):
            assert workspace_arg == str(workspace)
            assert target_prod_dir == str(prod_00)
            assert participant == {"name": "bob@nvidia.com", "type": "admin", "org": "nvidia", "role": "lead"}
            assert {"name": "site-1", "type": "client", "org": "nvidia"} in project_config_arg["participants"]
            assert force is False
            _make_admin_startup_kit(Path(target_prod_dir), participant["name"])
            return os.path.join(target_prod_dir, participant["name"])

        with patch(
            "nvflare.tool.poc.poc_commands._provision_poc_participant_only",
            side_effect=fake_provision_participant_only,
        ):
            result = _add_poc_user(str(workspace), "lead", "bob@nvidia.com", "nvidia")

        persisted = yaml.safe_load(project_file.read_text())
        assert {"name": "bob@nvidia.com", "type": "admin", "role": "lead", "org": "nvidia"} in persisted["participants"]
        assert result["status"] == "added"
        assert result["id"] == "bob@nvidia.com"
        assert result["startup_kit"] == str(workspace / project_name / "prod_00" / "bob@nvidia.com")
        assert result["next_step"] == "nvflare config use bob@nvidia.com"
        assert "active" not in result
        assert not (workspace / project_name / "prod_01").exists()

        config = _load_config_dict(config_path)
        entries = config["startup_kits"]["entries"]
        assert config["startup_kits"]["active"] == "admin@nvidia.com"
        assert entries["admin@nvidia.com"] == str(workspace / project_name / "prod_00" / "admin@nvidia.com")
        assert entries["bob@nvidia.com"] == str(workspace / project_name / "prod_00" / "bob@nvidia.com")
        assert "site-1" not in entries

    def test_poc_add_user_auto_activates_when_no_active_kit_exists(self, tmp_path, monkeypatch):
        from nvflare.tool.poc.poc_commands import _add_poc_user

        monkeypatch.setenv("HOME", str(tmp_path))
        workspace = tmp_path / "poc_ws"
        project_name = "example_project"
        prod_00 = workspace / project_name / "prod_00"
        _make_admin_startup_kit(prod_00, "admin@nvidia.com")
        workspace.mkdir(parents=True, exist_ok=True)
        project_file = workspace / "project.yml"
        project_file.write_text(
            """
name: example_project
participants:
  - name: server
    type: server
  - name: admin@nvidia.com
    type: admin
    role: project_admin
    org: nvidia
"""
        )

        def fake_provision_participant_only(workspace_arg, project_config_arg, participant, target_prod_dir, force):
            assert workspace_arg == str(workspace)
            assert target_prod_dir == str(prod_00)
            assert participant == {"name": "bob@nvidia.com", "type": "admin", "org": "nvidia", "role": "lead"}
            assert force is False
            _make_admin_startup_kit(Path(target_prod_dir), participant["name"])
            return os.path.join(target_prod_dir, participant["name"])

        with patch(
            "nvflare.tool.poc.poc_commands._provision_poc_participant_only",
            side_effect=fake_provision_participant_only,
        ):
            result = _add_poc_user(str(workspace), "lead", "bob@nvidia.com", "nvidia")

        config_path = tmp_path / ".nvflare" / "config.conf"
        config = _load_config_dict(config_path)
        entries = config["startup_kits"]["entries"]
        assert config["startup_kits"]["active"] == "bob@nvidia.com"
        assert entries["admin@nvidia.com"] == str(workspace / project_name / "prod_00" / "admin@nvidia.com")
        assert entries["bob@nvidia.com"] == str(workspace / project_name / "prod_00" / "bob@nvidia.com")
        assert result["id"] == "bob@nvidia.com"
        assert result["active"] == "bob@nvidia.com"
        assert "next_step" not in result

    def test_poc_add_site_persists_project_without_registering_site_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.poc.poc_commands import _add_poc_site

        monkeypatch.setenv("HOME", str(tmp_path))
        workspace = tmp_path / "poc_ws"
        project_name = "example_project"
        prod_00 = workspace / project_name / "prod_00"
        _make_admin_startup_kit(prod_00, "admin@nvidia.com")
        _make_site_startup_kit(prod_00, "site-1")
        workspace.mkdir(parents=True, exist_ok=True)
        project_file = workspace / "project.yml"
        project_file.write_text(
            """
name: example_project
participants:
  - name: server
    type: server
  - name: site-1
    type: client
    org: nvidia
  - name: admin@nvidia.com
    type: admin
    role: project_admin
    org: nvidia
"""
        )
        config_path = tmp_path / ".nvflare" / "config.conf"
        config_path.parent.mkdir()
        config_path.write_text(
            f"""
version = 2
startup_kits {{
  active = "admin@nvidia.com"
  entries {{
    "admin@nvidia.com" = "{prod_00 / 'admin@nvidia.com'}"
  }}
}}
poc {{
  workspace = "{workspace}"
}}
"""
        )

        def fake_provision_participant_only(workspace_arg, project_config_arg, participant, target_prod_dir, force):
            assert workspace_arg == str(workspace)
            assert target_prod_dir == str(prod_00)
            assert participant == {"name": "site-3", "type": "client", "org": "nvidia"}
            assert {
                "name": "admin@nvidia.com",
                "type": "admin",
                "role": "project_admin",
                "org": "nvidia",
            } in project_config_arg["participants"]
            assert force is False
            _make_site_startup_kit(Path(target_prod_dir), participant["name"])
            return os.path.join(target_prod_dir, participant["name"])

        with patch(
            "nvflare.tool.poc.poc_commands._provision_poc_participant_only",
            side_effect=fake_provision_participant_only,
        ):
            result = _add_poc_site(str(workspace), "site-3", "nvidia")

        persisted = yaml.safe_load(project_file.read_text())
        assert {"name": "site-3", "type": "client", "org": "nvidia"} in persisted["participants"]
        assert result["status"] == "added"
        assert result["id"] == "site-3"
        assert result["startup_kit"] == str(workspace / project_name / "prod_00" / "site-3")
        assert not (workspace / project_name / "prod_01").exists()

        config = _load_config_dict(config_path)
        entries = config["startup_kits"]["entries"]
        assert config["startup_kits"]["active"] == "admin@nvidia.com"
        assert entries == {
            "admin@nvidia.com": str(workspace / project_name / "prod_00" / "admin@nvidia.com"),
        }

    def test_poc_add_rejects_duplicate_without_force(self, tmp_path, monkeypatch):
        from nvflare.tool.poc.poc_commands import _add_poc_site

        monkeypatch.setenv("HOME", str(tmp_path))
        workspace = tmp_path / "poc_ws"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "project.yml").write_text(
            """
name: example_project
participants:
  - name: site-1
    type: client
    org: nvidia
"""
        )

        with patch("nvflare.tool.poc.poc_commands._provision_poc_participant_only") as mock_prepare:
            with pytest.raises(CLIException, match="already exists"):
                _add_poc_site(str(workspace), "site-1", "nvidia")

        mock_prepare.assert_not_called()

    def test_poc_add_user_rejects_role_change_before_persisting(self, tmp_path, monkeypatch):
        from nvflare.tool.poc.poc_commands import _add_poc_user

        monkeypatch.setenv("HOME", str(tmp_path))
        workspace = tmp_path / "poc_ws"
        workspace.mkdir(parents=True, exist_ok=True)
        project_file = workspace / "project.yml"
        project_file.write_text(
            """
name: example_project
participants:
  - name: bob@nvidia.com
    type: admin
    role: lead
    org: nvidia
"""
        )

        with patch("nvflare.tool.poc.poc_commands._provision_poc_participant_only") as mock_prepare:
            with pytest.raises(CLIException, match="changing a POC user's certificate role"):
                _add_poc_user(str(workspace), "org_admin", "bob@nvidia.com", "nvidia", force=True)

        mock_prepare.assert_not_called()
        persisted = yaml.safe_load(project_file.read_text())
        assert persisted["participants"] == [
            {
                "name": "bob@nvidia.com",
                "type": "admin",
                "role": "lead",
                "org": "nvidia",
            }
        ]

    def test_save_startup_kit_dir_config_writes_registry_and_poc_workspace(self, tmp_path, monkeypatch):
        from nvflare.tool.poc.poc_commands import get_poc_workspace, save_startup_kit_dir_config

        monkeypatch.setenv("HOME", str(tmp_path))
        workspace = tmp_path / "poc_ws"
        project_name = "example_project"
        prod_dir = workspace / project_name / "prod_00"
        os.makedirs(workspace, exist_ok=True)
        _make_admin_startup_kit(prod_dir, "admin@nvidia.com")
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

        dst = tmp_path / ".nvflare" / "config.conf"
        with patch(
            "nvflare.tool.poc.poc_commands.get_or_create_hidden_nvflare_dir",
            return_value=str(tmp_path),
        ):
            with patch(
                "nvflare.tool.poc.poc_commands.get_hidden_nvflare_config_path",
                return_value=str(dst),
            ):
                save_startup_kit_dir_config(str(workspace), project_name)

        config = _load_config_dict(dst)
        assert config["startup_kits"]["active"] == "admin@nvidia.com"
        assert config["startup_kits"]["entries"] == {
            "admin@nvidia.com": str(prod_dir / "admin@nvidia.com"),
        }
        assert config["poc"]["workspace"] == str(workspace)
        assert "startup_kit" not in config["poc"]
        assert "prod" not in config or "startup_kit" not in config["prod"]

        with patch(
            "nvflare.tool.poc.poc_commands.get_or_create_hidden_nvflare_dir",
            return_value=str(tmp_path),
        ):
            with patch(
                "nvflare.tool.poc.poc_commands.get_hidden_nvflare_config_path",
                return_value=str(dst),
            ):
                with patch.dict(os.environ, {}, clear=True):
                    assert get_poc_workspace() == str(workspace)

    def test_register_poc_startup_kits_replaces_previous_poc_workspace_entries(self, tmp_path):
        from nvflare.tool.kit.kit_config import get_startup_kit_entries
        from nvflare.tool.poc.poc_commands import _register_poc_startup_kits

        old_workspace = tmp_path / "old_poc"
        new_workspace = tmp_path / "new_poc"
        old_admin = old_workspace / "example_project" / "prod_00" / "admin@nvidia.com"
        new_prod_dir = new_workspace / "example_project" / "prod_00"
        new_admin = _make_admin_startup_kit(new_prod_dir, "admin@nvidia.com")
        manual_admin = tmp_path / "prod" / "cancer" / "lead@nvidia.com"

        config = CF.parse_string(
            f"""
version = 2
startup_kits {{
  active = "admin@nvidia.com"
  entries {{
    "admin@nvidia.com" = "{old_admin}"
    cancer_lead = "{manual_admin}"
  }}
}}
poc {{
  workspace = "{old_workspace}"
}}
"""
        )

        updated, removed_ids = _register_poc_startup_kits(
            config,
            str(new_workspace),
            {"admin@nvidia.com": str(new_admin)},
        )

        entries = get_startup_kit_entries(updated)
        assert removed_ids == {"admin@nvidia.com"}
        assert entries["admin@nvidia.com"] == str(new_admin.resolve())
        assert entries["cancer_lead"] == str(manual_admin)

    def test_register_poc_startup_kits_rejects_manual_id_conflict(self, tmp_path):
        from nvflare.tool.poc.poc_commands import _register_poc_startup_kits

        workspace = tmp_path / "poc"
        new_admin = _make_admin_startup_kit(workspace / "example_project" / "prod_00", "admin@nvidia.com")
        manual_admin = _make_admin_startup_kit(tmp_path / "prod", "admin@nvidia.com")

        config = CF.parse_string(
            f"""
version = 2
startup_kits {{
  entries {{
    "admin@nvidia.com" = "{manual_admin}"
  }}
}}
"""
        )

        with pytest.raises(CLIException, match="already exists outside POC workspace"):
            _register_poc_startup_kits(config, str(workspace), {"admin@nvidia.com": str(new_admin)})

    def test_register_poc_startup_kits_replaces_stale_outside_entry(self, tmp_path):
        from nvflare.tool.kit.kit_config import get_startup_kit_entries
        from nvflare.tool.poc.poc_commands import _register_poc_startup_kits

        workspace = tmp_path / "poc"
        new_admin = _make_admin_startup_kit(workspace / "example_project" / "prod_00", "admin@nvidia.com")
        stale_admin = tmp_path / "deleted-poc" / "example_project" / "prod_00" / "admin@nvidia.com"

        config = CF.parse_string(
            f"""
version = 2
startup_kits {{
  active = "admin@nvidia.com"
  entries {{
    "admin@nvidia.com" = "{stale_admin}"
  }}
}}
"""
        )

        updated, removed_ids = _register_poc_startup_kits(config, str(workspace), {"admin@nvidia.com": str(new_admin)})

        entries = get_startup_kit_entries(updated)
        assert removed_ids == {"admin@nvidia.com"}
        assert entries["admin@nvidia.com"] == str(new_admin.resolve())

    def test_clean_poc_removes_workspace_and_only_canonical_workspace_entries(self, tmp_path, monkeypatch):
        from nvflare.tool.poc.poc_commands import _clean_poc

        monkeypatch.setenv("HOME", str(tmp_path))
        workspace = tmp_path / "poc"
        workspace.mkdir()
        poc_admin = workspace / "example_project" / "prod_00" / "admin@nvidia.com"
        poc_lead = workspace / "example_project" / "prod_00" / ".." / "prod_00" / "lead@nvidia.com"
        poc_backup_admin = tmp_path / "poc-backup" / "example_project" / "prod_00" / "admin@nvidia.com"
        prod_admin = tmp_path / "prod" / "cancer" / "lead@nvidia.com"
        dst = tmp_path / ".nvflare" / "config.conf"
        dst.parent.mkdir()
        dst.write_text(
            f"""
version = 2
startup_kits {{
  active = "admin@nvidia.com"
  entries {{
    "admin@nvidia.com" = "{poc_admin}"
    "lead@nvidia.com" = "{poc_lead}"
    "poc-backup-admin" = "{poc_backup_admin}"
    cancer_lead = "{prod_admin}"
  }}
}}
poc {{
  workspace = "{workspace}"
  startup_kit = "{poc_admin}"
  mode = "keep"
}}
prod {{
  startup_kit = "{prod_admin}"
  mode = "keep"
}}
"""
        )

        with (
            patch(
                "nvflare.tool.poc.poc_commands.get_or_create_hidden_nvflare_dir",
                return_value=str(tmp_path),
            ),
            patch(
                "nvflare.tool.poc.poc_commands.get_hidden_nvflare_config_path",
                return_value=str(dst),
            ),
            patch(
                "nvflare.tool.poc.poc_commands.setup_service_config",
                return_value=({"name": "example_project"}, {SC.FLARE_SERVER: "server"}),
            ),
            patch("nvflare.tool.poc.poc_commands.is_poc_ready", return_value=True),
            patch("nvflare.tool.poc.poc_commands.is_poc_running", return_value=False),
            patch("nvflare.tool.cli_output.print_human"),
        ):
            assert _clean_poc(str(workspace)) is True

        config = _load_config_dict(dst)
        entries = config["startup_kits"]["entries"]
        assert not workspace.exists()
        assert "active" not in config["startup_kits"]
        assert "admin@nvidia.com" not in entries
        assert "lead@nvidia.com" not in entries
        assert entries["poc-backup-admin"] == str(poc_backup_admin)
        assert entries["cancer_lead"] == str(prod_admin)
        assert config["poc"] == {"mode": "keep"}
        assert config["prod"] == {"mode": "keep"}

    def test_clean_poc_keeps_config_when_workspace_remove_fails(self, tmp_path, monkeypatch):
        from nvflare.tool.poc.poc_commands import _clean_poc

        monkeypatch.setenv("HOME", str(tmp_path))
        workspace = tmp_path / "poc"
        workspace.mkdir()
        poc_admin = workspace / "example_project" / "prod_00" / "admin@nvidia.com"
        dst = tmp_path / ".nvflare" / "config.conf"
        dst.parent.mkdir()
        dst.write_text(
            f"""
version = 2
startup_kits {{
  active = "admin@nvidia.com"
  entries {{
    "admin@nvidia.com" = "{poc_admin}"
  }}
}}
poc {{
  workspace = "{workspace}"
}}
"""
        )

        with (
            patch(
                "nvflare.tool.poc.poc_commands.get_or_create_hidden_nvflare_dir",
                return_value=str(tmp_path),
            ),
            patch(
                "nvflare.tool.poc.poc_commands.get_hidden_nvflare_config_path",
                return_value=str(dst),
            ),
            patch(
                "nvflare.tool.poc.poc_commands.setup_service_config",
                return_value=({"name": "example_project"}, {SC.FLARE_SERVER: "server"}),
            ),
            patch("nvflare.tool.poc.poc_commands.is_poc_ready", return_value=True),
            patch("nvflare.tool.poc.poc_commands.is_poc_running", return_value=False),
            patch("nvflare.tool.poc.poc_commands.shutil.rmtree", side_effect=OSError("boom")),
            patch("nvflare.tool.cli_output.print_human") as print_human,
        ):
            with pytest.raises(OSError, match="boom"):
                _clean_poc(str(workspace))

        config = _load_config_dict(dst)
        assert config["startup_kits"]["active"] == "admin@nvidia.com"
        assert config["startup_kits"]["entries"] == {"admin@nvidia.com": str(poc_admin)}
        assert config["poc"]["workspace"] == str(workspace)
        print_human.assert_not_called()

    def test_clean_poc_force_stops_running_system_before_removal(self, tmp_path):
        from nvflare.tool.poc.poc_commands import _clean_poc

        workspace = tmp_path / "poc"
        workspace.mkdir()
        project_config = {"name": "example_project"}
        service_config = {SC.FLARE_SERVER: "server"}

        with (
            patch(
                "nvflare.tool.poc.poc_commands.setup_service_config",
                return_value=(project_config, service_config),
            ),
            patch("nvflare.tool.poc.poc_commands.is_poc_ready", return_value=True),
            patch("nvflare.tool.poc.poc_commands.is_poc_running", side_effect=[True, False]),
            patch("nvflare.tool.poc.poc_commands._ensure_poc_stopped") as ensure_stopped,
            patch("nvflare.tool.poc.poc_commands.shutil.rmtree") as rmtree,
            patch("nvflare.tool.poc.poc_commands._clean_poc_config") as clean_config,
            patch("nvflare.tool.cli_output.print_human"),
        ):
            assert _clean_poc(str(workspace), force=True) is True

        ensure_stopped.assert_called_once_with(
            str(workspace),
            project_config=project_config,
            service_config=service_config,
            reason="cleaning the workspace",
        )
        rmtree.assert_called_once_with(str(workspace))
        clean_config.assert_called_once_with(str(workspace))

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
