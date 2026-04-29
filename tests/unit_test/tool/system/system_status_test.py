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


def _configure_active_startup_kit(tmp_path, monkeypatch):
    home = tmp_path / "home"
    admin_dir = tmp_path / "active-admin"
    startup_dir = admin_dir / "startup"
    startup_dir.mkdir(parents=True)
    (startup_dir / "fed_admin.json").write_text('{"admin": {"username": "admin@nvidia.com"}}', encoding="utf-8")
    (startup_dir / "client.crt").write_text("cert", encoding="utf-8")
    (startup_dir / "rootCA.pem").write_text("root", encoding="utf-8")

    config_dir = home / ".nvflare"
    config_dir.mkdir(parents=True)
    (config_dir / "config.conf").write_text(
        f"""
        version = 2
        startup_kits {{
          active = "admin@nvidia.com"
          entries {{
            "admin@nvidia.com" = "{admin_dir}"
          }}
        }}
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)
    return admin_dir


class TestSystemStatus:
    """Tests for nvflare system status command."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, target=None, client_names=None, output="json"):
        args = MagicMock()
        args.target = target
        args.client_names = client_names or []
        args.output = output
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

    def test_get_system_session_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.system.system_cli import _get_system_session

        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)
        fake_session = MagicMock()

        with patch("nvflare.tool.cli_session.new_secure_session", return_value=fake_session) as new_secure:
            sess = _get_system_session(argparse.Namespace())

        assert sess is fake_session
        assert new_secure.call_args.kwargs["username"] == "admin@nvidia.com"
        assert new_secure.call_args.kwargs["startup_kit_location"] == str(active_admin_dir)

    def test_get_system_session_still_exits_if_output_error_is_mocked(self, monkeypatch, tmp_path):
        from nvflare.tool.system.system_cli import _get_system_session

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

        with patch("nvflare.tool.system.system_cli.output_error") as mocked_output_error:
            with pytest.raises(SystemExit) as exc_info:
                _get_system_session(argparse.Namespace())

        assert exc_info.value.code == 4
        mocked_output_error.assert_called_once()

    def test_get_system_session_active_resolver_error_uses_startup_kit_missing(self, capsys, monkeypatch, tmp_path):
        from nvflare.tool.system.system_cli import _get_system_session

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            _get_system_session(argparse.Namespace())

        assert exc_info.value.code == 4
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["error_code"] == "STARTUP_KIT_MISSING"

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

    def test_system_session_none_guard_when_get_system_session_returns_none(self):
        from nvflare.tool.system.system_cli import _system_session

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=None):
            with _system_session():
                pass

    @pytest.mark.parametrize(
        ("argv_prefix"),
        [
            ["status"],
            ["resources"],
            ["shutdown", "server", "--force"],
            ["restart", "server", "--force"],
            ["remove-client", "site-1", "--force"],
            ["disable-client", "site-1", "--force"],
            ["enable-client", "site-1", "--force"],
            ["version"],
            ["log-config", "INFO"],
        ],
    )
    @pytest.mark.parametrize(
        ("selector", "value"),
        [
            ("--startup-target", "prod"),
            ("--startup_target", "prod"),
            ("--startup-kit", "/tmp/startup"),
            ("--startup_kit", "/tmp/startup"),
        ],
    )
    def test_system_parser_rejects_old_startup_selectors_after_subcommand(self, argv_prefix, selector, value):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)

        with pytest.raises(SystemExit):
            parser.parse_args([*argv_prefix, selector, value])

    @pytest.mark.parametrize(
        ("selector", "value"),
        [
            ("--startup-target", "prod"),
            ("--startup_target", "prod"),
            ("--startup-kit", "/tmp/startup"),
            ("--startup_kit", "/tmp/startup"),
        ],
    )
    def test_system_parser_rejects_old_startup_selectors_before_subcommand(self, selector, value):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)

        with pytest.raises(SystemExit):
            parser.parse_args([selector, value, "status"])

    def test_system_help_and_schema_omit_old_startup_selectors(self, capsys):
        from nvflare.tool.system import system_cli

        parser = argparse.ArgumentParser(prog="nvflare system")
        system_cli.def_system_cli_parser(parser)

        all_help = [parser.format_help()] + [p.format_help() for p in system_cli._system_sub_cmd_parsers.values()]
        for help_text in all_help:
            for token in ("--startup-target", "--startup_target", "--startup-kit", "--startup_kit"):
                assert token not in help_text

        schema_cases = [
            ("status", system_cli.cmd_system_status),
            ("resources", system_cli.cmd_system_resources),
            ("shutdown", system_cli.cmd_system_shutdown),
            ("restart", system_cli.cmd_system_restart),
            ("remove-client", system_cli.cmd_system_remove_client),
            ("disable-client", system_cli.cmd_system_disable_client),
            ("enable-client", system_cli.cmd_system_enable_client),
            ("version", system_cli.cmd_system_version),
            ("log-config", system_cli.cmd_system_log),
        ]
        for cmd_name, handler in schema_cases:
            with patch("sys.argv", ["nvflare", "system", cmd_name, "--schema"]):
                with pytest.raises(SystemExit) as exc_info:
                    handler(MagicMock())
            assert exc_info.value.code == 0
            schema_text = capsys.readouterr().out
            for token in ("--startup-target", "--startup_target", "--startup-kit", "--startup_kit"):
                assert token not in schema_text


class TestSystemActiveStartupKit:
    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _run_and_assert_active_session(self, tmp_path, monkeypatch, command_fn, args, configure_session):
        active_admin_dir = _configure_active_startup_kit(tmp_path, monkeypatch)
        sess = MagicMock()
        configure_session(sess)

        with patch("nvflare.tool.cli_session.new_secure_session", return_value=sess) as new_secure:
            command_fn(args)

        assert new_secure.call_args.kwargs["username"] == "admin@nvidia.com"
        assert new_secure.call_args.kwargs["startup_kit_location"] == str(active_admin_dir)
        return sess

    def test_status_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.system.system_cli import cmd_system_status

        args = argparse.Namespace(target=None, client_names=[], output="json")
        sess = self._run_and_assert_active_session(
            tmp_path,
            monkeypatch,
            cmd_system_status,
            args,
            lambda s: setattr(s.check_status, "return_value", {"server_status": "running"}),
        )

        sess.check_status.assert_called_once_with("all", None)

    def test_resources_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.system.system_cli import cmd_system_resources

        args = argparse.Namespace(target=None, client_names=[], output="json")
        sess = self._run_and_assert_active_session(
            tmp_path,
            monkeypatch,
            cmd_system_resources,
            args,
            lambda s: setattr(s.report_resources, "return_value", {"server": {"cpu": 1}}),
        )

        sess.report_resources.assert_called_once_with("all", None)

    def test_shutdown_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.system.system_cli import cmd_system_shutdown

        args = argparse.Namespace(target="server", client_names=[], force=True)
        sess = self._run_and_assert_active_session(
            tmp_path,
            monkeypatch,
            cmd_system_shutdown,
            args,
            lambda s: setattr(s.shutdown, "return_value", {"status": "ok"}),
        )

        sess.shutdown.assert_called_once_with("server", client_names=None)

    def test_restart_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.system.system_cli import cmd_system_restart

        args = argparse.Namespace(target="server", client_names=[], force=True)
        sess = self._run_and_assert_active_session(
            tmp_path,
            monkeypatch,
            cmd_system_restart,
            args,
            lambda s: setattr(s.restart, "return_value", {"status": "ok"}),
        )

        sess.restart.assert_called_once_with("server", client_names=None)

    def test_version_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.system.system_cli import cmd_system_version

        args = argparse.Namespace(site="server")
        sess = self._run_and_assert_active_session(
            tmp_path,
            monkeypatch,
            cmd_system_version,
            args,
            lambda s: setattr(s.report_version, "return_value", {"server": {"version": "1.0.0"}}),
        )

        sess.report_version.assert_called_once_with("server", None)

    def test_log_config_uses_active_startup_kit(self, tmp_path, monkeypatch):
        from nvflare.tool.system.system_cli import cmd_system_log

        args = argparse.Namespace(level="INFO", site="server", schema=False)
        sess = self._run_and_assert_active_session(
            tmp_path,
            monkeypatch,
            cmd_system_log,
            args,
            lambda s: setattr(s.configure_site_log, "return_value", None),
        )

        sess.configure_site_log.assert_called_once_with("INFO", target="server")


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
        assert data["data"]["status"] == "deregistered_from_server_registry"
        assert data["data"]["reconnect_prevented"] is False
        assert data["data"]["credential_revoked"] is False

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


class TestSystemDisableEnableClient:
    """Tests for nvflare system disable-client and enable-client commands."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, client_name="site-1", force=True):
        args = MagicMock()
        args.client_name = client_name
        args.force = force
        return args

    def test_disable_client_calls_session_disable_client(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_disable_client

        args = self._make_args(client_name="site-1")
        sess = MagicMock()
        sess.disable_client.return_value = {
            "clients": [
                {
                    "client_name": "site-1",
                    "state": "disabled",
                    "active_session_removed": True,
                    "credential_revoked": False,
                    "rejoin_allowed": False,
                }
            ]
        }

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_disable_client(args)

        sess.disable_client.assert_called_once_with("site-1")
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["client_name"] == "site-1"
        assert data["data"]["state"] == "disabled"
        assert data["data"]["active_session_removed"] is True
        assert data["data"]["credential_revoked"] is False
        assert data["data"]["rejoin_allowed"] is False

    def test_enable_client_calls_session_enable_client(self, capsys):
        from nvflare.tool.system.system_cli import cmd_system_enable_client

        args = self._make_args(client_name="site-1")
        sess = MagicMock()
        sess.enable_client.return_value = {
            "clients": [
                {
                    "client_name": "site-1",
                    "state": "enabled",
                    "was_disabled": True,
                    "credential_revoked": False,
                    "rejoin_allowed": True,
                }
            ]
        }

        with patch("nvflare.tool.system.system_cli._get_system_session", return_value=sess):
            cmd_system_enable_client(args)

        sess.enable_client.assert_called_once_with("site-1")
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["client_name"] == "site-1"
        assert data["data"]["state"] == "enabled"
        assert data["data"]["was_disabled"] is True
        assert data["data"]["credential_revoked"] is False
        assert data["data"]["rejoin_allowed"] is True

    def test_disable_enable_parser_accepts_client_name(self):
        from nvflare.tool.system.system_cli import def_system_cli_parser

        parser = argparse.ArgumentParser(prog="nvflare system")
        def_system_cli_parser(parser)
        disable_args = parser.parse_args(["disable-client", "site-1", "--force"])
        enable_args = parser.parse_args(["enable-client", "site-1", "--force"])
        assert disable_args.client_name == "site-1"
        assert enable_args.client_name == "site-1"
        assert disable_args.force is True
        assert enable_args.force is True
