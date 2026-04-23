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

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from nvflare.fuel.flare_api.api_spec import AuthenticationError, InvalidTarget, NoConnection
from nvflare.tool import cli_output


class TestSystemStatus:
    """Tests for nvflare system status command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, target=None, client_names=None, output="json"):
        args = MagicMock()
        args.target = target
        args.startup_target = None
        args.client_names = client_names or []
        args.output = output
        args.startup_kit = None
        return args

    def test_status_json_output_shape(self, capsys):
        """JSON output has schema_version, status, and data keys."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {"server_status": "running", "clients": []}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert "data" in data

    def test_status_server_only(self, capsys):
        """Target=server calls check_status with 'server'."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args(target="server")
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {"server_status": "running"}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        mock_sess.check_status.assert_called_once_with("server", None)

    def test_status_with_client_names(self, capsys):
        """Client names are passed to check_status."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args(target="client", client_names=["site-1", "site-2"])
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {"client_status": {}}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        mock_sess.check_status.assert_called_once_with("client", ["site-1", "site-2"])

    def test_status_connection_failed_exits_2(self):
        """Connection failure exits with code 2."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=NoConnection("connection error")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_status(args)
        assert exc_info.value.code == 2

    def test_status_connection_failed_uses_custom_hint(self, capsys):
        """system status uses a non-recursive hint on connection failure."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=NoConnection("connection error")):
            with pytest.raises(SystemExit):
                cmd_system_status(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["hint"] == "Start the server or verify the admin startup kit endpoint."

    def test_status_propagates_authentication_error(self):
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        with patch(
            "nvflare.tool.system.system_cli._get_system_session",
            side_effect=AuthenticationError("certificate issue"),
        ):
            with pytest.raises(AuthenticationError, match="certificate issue"):
                cmd_system_status(args)

    def test_status_connection_failed_does_not_fall_through_when_error_output_mocked(self):
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        mocked_output = MagicMock()
        mocked_render = MagicMock()

        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=NoConnection("connection error")):
            with patch("nvflare.tool.system.system_cli.output_error_message", mocked_output):
                with patch("nvflare.tool.system.system_cli._output_system_status", mocked_render):
                    with pytest.raises(SystemExit) as exc_info:
                        cmd_system_status(args)

        assert exc_info.value.code == 2
        mocked_output.assert_called_once()
        mocked_render.assert_not_called()

    def test_status_unexpected_exception_maps_to_internal_error(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        with patch("nvflare.tool.system.system_cli._get_system_session", side_effect=RuntimeError("boom")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_status(args)

        assert exc_info.value.code == 5
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "INTERNAL_ERROR"

    def test_status_default_target_is_all(self, capsys):
        """When target is None, defaults to 'all'."""
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args(target=None)
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        mock_sess.check_status.assert_called_once_with("all", None)

    def test_get_system_session_missing_username_exits_startup_kit_missing(self, capsys):
        from nvflare.tool.system.system_cli import _get_system_session

        args = MagicMock()
        args.target = None
        args.startup_target = None
        args.startup_kit = None

        with patch(
            "nvflare.utils.cli_utils.get_startup_kit_dir_for_target",
            return_value="/tmp/poc-startup",
        ):
            with patch(
                "nvflare.tool.job.job_cli._resolve_admin_user_and_dir_from_startup_kit",
                side_effect=Exception("bad startup"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    _get_system_session(args)

        assert exc_info.value.code == 4
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "STARTUP_KIT_MISSING"
        assert "admin username could not be resolved" in envelope["message"]

    def test_get_system_session_still_exits_if_output_error_is_mocked(self):
        from nvflare.tool.system.system_cli import _get_system_session

        args = MagicMock()
        args.target = None
        args.startup_target = None
        args.startup_kit = None

        with patch(
            "nvflare.utils.cli_utils.get_startup_kit_dir_for_target",
            side_effect=ValueError("no startup kit configured"),
        ):
            with patch("nvflare.tool.system.system_cli.output_error") as mocked_output_error:
                with pytest.raises(SystemExit) as exc_info:
                    _get_system_session(args)

        assert exc_info.value.code == 4
        mocked_output_error.assert_called_once()

    def test_get_system_session_startup_kit_fallback_error_uses_startup_kit_missing(self, capsys):
        from nvflare.tool.system.system_cli import _get_system_session

        args = MagicMock()
        args.target = None
        args.startup_target = None
        args.startup_kit = None

        with patch(
            "nvflare.utils.cli_utils.get_startup_kit_dir_for_target",
            side_effect=ValueError("no startup kit configured"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _get_system_session(args)

        assert exc_info.value.code == 4
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "STARTUP_KIT_MISSING"
        assert "no startup kit configured" in envelope["message"]

    def test_get_system_session_uses_explicit_target(self):
        from nvflare.tool.system.system_cli import _get_system_session

        args = MagicMock()
        args.target = "prod"
        args.startup_target = "prod"
        args.startup_kit = None

        with patch(
            "nvflare.utils.cli_utils.get_startup_kit_dir_for_target",
            return_value="/tmp/prod-startup",
        ) as get_startup:
            with patch(
                "nvflare.tool.job.job_cli._resolve_admin_user_and_dir_from_startup_kit",
                return_value=("admin@nvidia.com", "/tmp/prod-startup"),
            ):
                with patch("nvflare.tool.cli_session.new_cli_session", return_value=MagicMock()):
                    with patch("nvflare.tool.cli_output.get_connect_timeout", return_value=10.0):
                        _get_system_session(args)

        get_startup.assert_called_once_with(startup_kit_dir=None, target="prod")

    def test_get_system_session_uses_explicit_startup_kit_override(self):
        from nvflare.tool.system.system_cli import _get_system_session

        args = MagicMock()
        args.target = "prod"
        args.startup_target = "prod"
        args.startup_kit = "/tmp/custom-startup"

        with patch(
            "nvflare.utils.cli_utils.get_startup_kit_dir_for_target",
            return_value="/tmp/custom-startup",
        ) as get_startup:
            with patch(
                "nvflare.tool.job.job_cli._resolve_admin_user_and_dir_from_startup_kit",
                return_value=("admin@nvidia.com", "/tmp/custom-startup"),
            ):
                with patch("nvflare.tool.cli_session.new_cli_session", return_value=MagicMock()):
                    with patch("nvflare.tool.cli_output.get_connect_timeout", return_value=10.0):
                        _get_system_session(args)

        get_startup.assert_called_once_with(startup_kit_dir="/tmp/custom-startup", target="prod")

    def test_confirm_or_force_does_not_prompt_when_output_error_is_mocked(self):
        from nvflare.tool.system.system_cli import _confirm_or_force

        args = MagicMock()
        args.force = False

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with patch("nvflare.tool.system.system_cli.output_error") as mocked_output_error:
                with patch("nvflare.tool.cli_output.prompt_yn") as prompt_yn:
                    with pytest.raises(SystemExit) as exc_info:
                        _confirm_or_force("confirm?", args)

        assert exc_info.value.code == 4
        mocked_output_error.assert_called_once()
        prompt_yn.assert_not_called()

    def test_get_system_session_succeeds_when_defaulting_to_poc_target(self, monkeypatch):
        from nvflare.tool.system.system_cli import _get_system_session

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        args = MagicMock()
        args.target = None
        args.startup_target = None
        args.startup_kit = None

        with patch(
            "nvflare.utils.cli_utils.get_startup_kit_dir_for_target",
            return_value="/tmp/poc-startup",
        ):
            with patch(
                "nvflare.tool.job.job_cli._resolve_admin_user_and_dir_from_startup_kit",
                return_value=("admin@nvidia.com", "/tmp/poc-startup"),
            ):
                with patch("nvflare.tool.cli_session.new_cli_session", return_value=MagicMock()):
                    with patch("nvflare.tool.cli_output.get_connect_timeout", return_value=10.0):
                        _get_system_session(args)  # should not raise

    def test_get_system_session_resolves_startup_kit_once_before_username_lookup(self):
        from nvflare.tool.system.system_cli import _get_system_session

        args = MagicMock()
        args.target = "prod"
        args.startup_target = "prod"
        args.startup_kit = None

        with patch(
            "nvflare.utils.cli_utils.get_startup_kit_dir_for_target",
            return_value="/tmp/prod-startup",
        ) as get_startup:
            with patch(
                "nvflare.tool.job.job_cli._resolve_admin_user_and_dir_from_startup_kit",
                return_value=("admin@nvidia.com", "/tmp/prod-startup"),
            ) as resolve_admin:
                with patch("nvflare.tool.cli_session.new_cli_session", return_value=MagicMock()):
                    with patch("nvflare.tool.cli_output.get_connect_timeout", return_value=10.0):
                        _get_system_session(args)

        get_startup.assert_called_once_with(startup_kit_dir=None, target="prod")
        resolve_admin.assert_called_once_with("/tmp/prod-startup")

    def test_system_session_none_guard_when_get_system_session_returns_none(self):
        from nvflare.tool.system.system_cli import _system_session

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=None):
            with _system_session():
                pass

    def test_system_parser_accepts_startup_kit_after_subcommand(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)

        args = parser.parse_args(["status", "--startup-kit", "/tmp/startup", "--startup-target", "prod"])

        assert args.system_sub_cmd == "status"
        assert args.startup_kit == "/tmp/startup"
        assert args.startup_target == "prod"


class TestSystemResources:
    """Tests for nvflare system resources command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, target=None, client_names=None, output="json"):
        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        args.output = output
        return args

    def test_resources_json_empty_output(self, capsys):
        """Empty resources still return a JSON envelope."""
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.report_resources.return_value = {}

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_resources(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"] == {}


class TestSystemStatusHuman:
    @pytest.fixture(autouse=True)
    def text_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")

    def _make_args(self, target=None, client_names=None):
        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        return args

    def test_status_human_output_is_formatted(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args()
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {
            "server_status": "started",
            "server_start_time": 1775860407.993352,
            "jobs": [],
            "clients": [
                {"client_name": "site-1", "client_last_conn_time": 1775860421.002409, "fqcn": "site-1"},
                {"client_name": "site-2", "client_last_conn_time": 1775860421.75365, "fqcn": "site-2"},
            ],
            "client_status": [
                {"client_name": "site-1", "status": "no_jobs"},
                {"client_name": "site-2", "status": "no_jobs"},
            ],
        }

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        captured = capsys.readouterr()
        assert "Engine status: started" in captured.out
        assert "Registered clients: 2" in captured.out
        assert "site-1" in captured.out
        assert "site-2" in captured.out
        assert "clients: [{'client_name'" not in captured.out

    def test_status_human_output_expands_server_job_table_for_long_job_ids(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args(target="server")
        long_job_id = "123e4567-e89b-12d3-a456-426614174000"
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {
            "server_status": "started",
            "server_start_time": 1775860407.993352,
            "jobs": [{"job_id": long_job_id, "app_name": "hello-pt"}],
            "clients": [],
            "client_status": [],
        }

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        captured = capsys.readouterr()
        assert f"| {long_job_id} | hello-pt |" in captured.out

    def test_status_client_human_uses_client_status_count_when_inventory_missing(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_status

        args = self._make_args(target="client")
        mock_sess = MagicMock()
        mock_sess.check_status.return_value = {
            "client_status": [
                {"client_name": "site-1", "status": "no_jobs"},
                {"client_name": "site-2", "status": "no_jobs"},
            ]
        }

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=mock_sess):
            cmd_system_status(args)

        captured = capsys.readouterr()
        assert "Clients: 2" in captured.out
        assert "site-1" in captured.out
        assert "site-2" in captured.out
        assert "Connected: 0" not in captured.out


class TestSystemShutdown:
    """Tests for nvflare system shutdown command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, target="server", client_names=None, force=True):
        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        args.force = force
        args.startup_target = None
        args.startup_kit = None
        return args

    def _make_session(self, result=None):
        sess = MagicMock()
        sess.shutdown.return_value = result or {"status": "ok"}
        return sess

    def test_shutdown_server_calls_session_shutdown(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(target="server")
        sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_shutdown(args)

        sess.shutdown.assert_called_once_with("server", client_names=None)

    def test_shutdown_client_all_calls_session_shutdown(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(target="client")
        sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_shutdown(args)

        sess.shutdown.assert_called_once_with("client", client_names=None)

    def test_shutdown_client_named_passes_client_names(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(target="client", client_names=["site-1", "site-2"])
        sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_shutdown(args)

        sess.shutdown.assert_called_once_with("client", client_names=["site-1", "site-2"])

    def test_shutdown_all_calls_session_shutdown(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(target="all")
        sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_shutdown(args)

        sess.shutdown.assert_called_once_with("all", client_names=None)

    def test_shutdown_invalid_target_exits_4(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(target="client", client_names=["unknown-site"])
        with patch(
            "nvflare.tool.system.system_cli._get_system_session",
            side_effect=InvalidTarget("invalid client(s): unknown-site"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_shutdown(args)

        assert exc_info.value.code == 4
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "INVALID_ARGS"
        assert "unknown-site" in data["message"]

    def test_shutdown_no_connection_reraises(self):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args()
        with patch(
            "nvflare.tool.system.system_cli._get_system_session",
            side_effect=NoConnection("no server"),
        ):
            with pytest.raises(NoConnection):
                cmd_system_shutdown(args)

    def test_shutdown_auth_error_reraises(self):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args()
        with patch(
            "nvflare.tool.system.system_cli._get_system_session",
            side_effect=AuthenticationError("bad cert"),
        ):
            with pytest.raises(AuthenticationError):
                cmd_system_shutdown(args)

    def test_shutdown_ok_output_shape(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = self._make_args(target="server")
        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=self._make_session()):
            cmd_system_shutdown(args)

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["target"] == "server"
        assert data["data"]["status"] == "shutdown initiated"

    def test_shutdown_parser_accepts_all_targets(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)
        for target in ("server", "client", "all"):
            args = parser.parse_args(["shutdown", target, "--force"])
            assert args.target == target
            assert args.force is True

    def test_shutdown_parser_accepts_client_names(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)
        args = parser.parse_args(["shutdown", "client", "site-1", "site-2"])
        assert args.target == "client"
        assert args.client_names == ["site-1", "site-2"]


class TestSystemRestart:
    """Tests for nvflare system restart command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, target="server", client_names=None, force=True):
        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        args.force = force
        args.startup_target = None
        args.startup_kit = None
        return args

    def _make_session(self, result=None):
        sess = MagicMock()
        sess.restart.return_value = result or {"status": "ok"}
        return sess

    def test_restart_server_calls_session_restart(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(target="server")
        sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_restart(args)

        sess.restart.assert_called_once_with("server", client_names=None)

    def test_restart_client_named_passes_client_names(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(target="client", client_names=["site-1"])
        sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_restart(args)

        sess.restart.assert_called_once_with("client", client_names=["site-1"])

    def test_restart_all_calls_session_restart(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(target="all")
        sess = self._make_session()

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_restart(args)

        sess.restart.assert_called_once_with("all", client_names=None)

    def test_restart_invalid_target_exits_4(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(target="client", client_names=["ghost"])
        with patch(
            "nvflare.tool.system.system_cli._get_system_session",
            side_effect=InvalidTarget("invalid client(s): ghost"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_restart(args)

        assert exc_info.value.code == 4
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "INVALID_ARGS"

    def test_restart_no_connection_reraises(self):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args()
        with patch(
            "nvflare.tool.system.system_cli._get_system_session",
            side_effect=NoConnection("no server"),
        ):
            with pytest.raises(NoConnection):
                cmd_system_restart(args)

    def test_restart_ok_output_shape(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = self._make_args(target="server")
        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=self._make_session()):
            cmd_system_restart(args)

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["target"] == "server"
        assert data["data"]["status"] == "restart initiated"

    def test_restart_parser_accepts_all_targets(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)
        for target in ("server", "client", "all"):
            args = parser.parse_args(["restart", target, "--force"])
            assert args.target == target

    def test_restart_parser_accepts_client_names(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)
        args = parser.parse_args(["restart", "client", "site-1", "site-2"])
        assert args.target == "client"
        assert args.client_names == ["site-1", "site-2"]


class TestSystemRemoveClient:
    """Tests for nvflare system remove-client command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, client_name="site-1", force=True):
        args = MagicMock()
        args.client_name = client_name
        args.force = force
        args.startup_target = None
        args.startup_kit = None
        return args

    def test_remove_client_calls_session_remove_client(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_remove_client

        args = self._make_args(client_name="site-1")
        sess = MagicMock()
        sess.remove_client.return_value = None

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_remove_client(args)

        sess.remove_client.assert_called_once_with("site-1")

    def test_remove_client_ok_output_shape(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_remove_client

        args = self._make_args(client_name="site-1")
        sess = MagicMock()
        sess.remove_client.return_value = None

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_remove_client(args)

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["client_name"] == "site-1"
        assert data["data"]["status"] == "removed"

    def test_remove_client_invalid_target_exits_4(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_remove_client

        args = self._make_args(client_name="ghost")
        with patch(
            "nvflare.tool.system.system_cli._get_system_session",
            side_effect=InvalidTarget("invalid client(s): ghost"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_system_remove_client(args)

        assert exc_info.value.code == 4
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "INVALID_ARGS"

    def test_remove_client_no_connection_reraises(self):
        from nvflare.tool.system.system_cli import cmd_system_remove_client

        args = self._make_args()
        with patch(
            "nvflare.tool.system.system_cli._get_system_session",
            side_effect=NoConnection("no server"),
        ):
            with pytest.raises(NoConnection):
                cmd_system_remove_client(args)

    def test_remove_client_auth_error_reraises(self):
        from nvflare.tool.system.system_cli import cmd_system_remove_client

        args = self._make_args()
        with patch(
            "nvflare.tool.system.system_cli._get_system_session",
            side_effect=AuthenticationError("bad cert"),
        ):
            with pytest.raises(AuthenticationError):
                cmd_system_remove_client(args)

    def test_remove_client_parser_accepts_client_name(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)
        args = parser.parse_args(["remove-client", "site-1", "--force"])
        assert args.client_name == "site-1"
        assert args.force is True
