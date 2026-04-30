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
import yaml
from pyhocon import ConfigFactory as CF

from nvflare.cli_exception import CLIException
from nvflare.tool import cli_output


class TestPocOutput:
    """Tests for poc subcommand JSON envelopes, exit codes, and stream routing."""

    @pytest.fixture(autouse=True)
    def json_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    # ------------------------------------------------------------------ helpers

    def _make_prepare_args(self, force=False):
        args = MagicMock()
        args.number_of_clients = 2
        args.clients = None
        args.docker_image = None
        args.he = False
        args.project_input = None
        args.force = force
        return args

    def _make_stop_args(self):
        args = MagicMock()
        args.service = None
        args.ex = None
        return args

    def test_client_gpu_assignments_uses_enumerated_gpu_id(self):
        from nvflare.tool.poc.poc_commands import client_gpu_assignments

        assignments = client_gpu_assignments(["site-1", "site-2"], [3, 7, 9])

        assert assignments == {"site-1": [3, 9], "site-2": [7]}

    # ------------------------------------------------------------------ unit envelope checks (output_ok / output_error)

    def test_output_ok_envelope_shape(self, capsys):
        """output_ok emits correct JSON schema_version/status/data."""
        from nvflare.tool.cli_output import output_ok

        output_ok({"workspace": "/tmp/poc", "clients": ["site-1", "site-2"]})
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert "workspace" in data["data"]
        assert "clients" in data["data"]

    def test_output_error_exits_with_code_4(self):
        """output_error with exit_code=4 calls sys.exit(4)."""
        from nvflare.tool.cli_output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("INVALID_ARGS", exit_code=4)
        assert exc_info.value.code == 4

    def test_poc_config_parser_accepts_workspace_flag(self):
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "config", "--pw", "/path/to/poc"])

        assert args.poc_sub_cmd == "config"
        assert args.poc_workspace_dir == "/path/to/poc"

    def test_poc_config_writes_workspace(self, capsys):
        from nvflare.tool.poc.poc_commands import config_poc

        args = MagicMock()
        args.poc_workspace_dir = "/path/to/poc"
        config = CF.parse_string("{}")

        with (
            patch(
                "nvflare.tool.poc.poc_commands.load_hidden_config_state",
                return_value=("/fake/config.conf", config, False),
            ),
            patch("nvflare.tool.poc.poc_commands.save_config") as save_config,
            patch("nvflare.tool.cli_schema.handle_schema_flag"),
        ):
            config_poc(args)

        saved_config = save_config.call_args.args[0]
        assert saved_config.get("poc.workspace") == "/path/to/poc"
        assert save_config.call_args.args[1] == "/fake/config.conf"
        captured = capsys.readouterr()
        assert "POC workspace configured: /path/to/poc" in captured.err
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["config_file"] == "/fake/config.conf"
        assert data["data"]["poc_workspace_dir"] == "/path/to/poc"

    def test_poc_config_human_output_omits_json_payload(self, capsys, monkeypatch):
        from nvflare.tool.poc.poc_commands import config_poc

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        args = MagicMock()
        args.poc_workspace_dir = "/path/to/poc"
        config = CF.parse_string("{}")

        with (
            patch(
                "nvflare.tool.poc.poc_commands.load_hidden_config_state",
                return_value=("/fake/config.conf", config, False),
            ),
            patch("nvflare.tool.poc.poc_commands.save_config"),
            patch("nvflare.tool.cli_schema.handle_schema_flag"),
        ):
            config_poc(args)

        captured = capsys.readouterr()
        assert captured.out.strip() == "POC workspace configured: /path/to/poc"
        assert "config_file:" not in captured.out
        assert "poc_workspace_dir:" not in captured.out
        assert captured.err == ""

    # ------------------------------------------------------------------ prepare_poc split-stream tests

    def test_add_poc_user_human_output_omits_json_payload(self, capsys, monkeypatch, tmp_path):
        from nvflare.tool.poc.poc_commands import add_poc_user

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        args = MagicMock()
        args.cert_role = "lead"
        args.email = "bob@nvidia.com"
        args.org = "nvidia"
        args.force = False
        result = {
            "status": "created",
            "identity": "bob@nvidia.com",
            "startup_kit": str(tmp_path / "bob@nvidia.com"),
            "next_step": "nvflare config use bob@nvidia.com",
        }

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands._require_poc_project_admin"),
            patch("nvflare.tool.poc.poc_commands._add_poc_user", return_value=result),
            patch("nvflare.tool.cli_schema.handle_schema_flag"),
        ):
            add_poc_user(args)

        captured = capsys.readouterr()
        assert "POC user created: bob@nvidia.com" in captured.out
        assert "Startup kit:" in captured.out
        assert "status:" not in captured.out
        assert "identity:" not in captured.out
        assert "next_step:" not in captured.out
        assert captured.err == ""

    def test_prepare_poc_success_stdout_is_one_json_line(self, capsys, tmp_path):
        """prepare_poc success: stdout is exactly one JSON line; nothing else."""
        from nvflare.tool.poc.poc_commands import prepare_poc

        args = self._make_prepare_args(force=True)

        poc_ws = str(tmp_path / "poc")
        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=poc_ws),
            patch("nvflare.tool.poc.poc_commands._prepare_poc", return_value=True),
            patch("nvflare.tool.install_skills.install_skills", return_value=None),
            patch("nvflare.tool.poc.poc_commands.os.path.exists", return_value=False),
        ):
            prepare_poc(args)

        captured = capsys.readouterr()
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1, f"Expected one JSON line on stdout, got: {captured.out!r}"
        data = json.loads(stdout_lines[0])
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["workspace"] == poc_ws
        assert data["data"]["clients"] == ["site-1", "site-2"]
        assert data["data"]["startup_kit"]["changed"] is False
        assert data["data"]["port_preflight"]["checked"] is False

    def test_prepare_poc_human_output_omits_agent_diagnostics(self, capsys, monkeypatch, tmp_path):
        from nvflare.tool.poc.poc_commands import prepare_poc

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        args = self._make_prepare_args(force=True)
        poc_ws = str(tmp_path / "poc")
        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=poc_ws),
            patch("nvflare.tool.poc.poc_commands._prepare_poc", return_value=True),
            patch("nvflare.tool.install_skills.install_skills", return_value=None),
            patch("nvflare.tool.poc.poc_commands.os.path.exists", return_value=False),
        ):
            prepare_poc(args)

        captured = capsys.readouterr()
        assert "POC workspace ready at:" in captured.out
        assert "Clients: site-1, site-2" in captured.out
        assert "Next: place your jobs" in captured.out
        assert "startup_kit:" not in captured.out
        assert "port_preflight:" not in captured.out
        assert "workspace:" not in captured.out
        assert "clients:" not in captured.out
        assert captured.err == ""

    def test_prepare_poc_falls_back_to_requested_clients_if_project_reread_fails(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import prepare_poc

        args = self._make_prepare_args(force=True)
        args.clients = ["site-a", "site-b"]
        args.number_of_clients = 2

        poc_ws = str(tmp_path / "poc")
        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=poc_ws),
            patch("nvflare.tool.poc.poc_commands._prepare_poc", return_value=True),
            patch("nvflare.tool.install_skills.install_skills", return_value=None),
            patch("nvflare.tool.poc.poc_commands.load_yaml", side_effect=yaml.YAMLError("bad yaml")),
        ):
            prepare_poc(args)

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["clients"] == ["site-a", "site-b"]
        assert data["data"]["port_preflight"]["checked"] is False

    def test_prepare_poc_reports_startup_kit_transition_and_port_conflicts(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import prepare_poc

        args = self._make_prepare_args(force=True)
        poc_ws = str(tmp_path / "poc")
        project_config = {
            "participants": [
                {"name": "server", "type": "server", "fed_learn_port": 8002, "admin_port": 8003},
                {"name": "site-1", "type": "client"},
                {"name": "site-2", "type": "client"},
            ]
        }

        def fake_port_available(port, host="127.0.0.1"):
            return (False, "in_use") if port == 8002 else (True, None)

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=poc_ws),
            patch("nvflare.tool.poc.poc_commands._prepare_poc", return_value=True),
            patch("nvflare.tool.install_skills.install_skills", return_value=None),
            patch("nvflare.tool.poc.poc_commands.load_yaml", return_value=project_config),
            patch(
                "nvflare.tool.poc.poc_commands._get_active_startup_kit_id_safely",
                side_effect=["prod-admin", "admin@nvidia.com"],
            ),
            patch("nvflare.tool.poc.poc_commands._is_local_port_available", side_effect=fake_port_available),
        ):
            prepare_poc(args)

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["clients"] == ["site-1", "site-2"]
        assert data["data"]["startup_kit"] == {
            "prior_active": "prod-admin",
            "active": "admin@nvidia.com",
            "changed": True,
        }
        port_preflight = data["data"]["port_preflight"]
        assert port_preflight["checked"] is True
        assert port_preflight["host"] == "127.0.0.1"
        assert port_preflight["scope"] == "loopback"
        assert "loopback" in port_preflight["note"]
        assert port_preflight["conflicts"] == [
            {
                "name": "fed_learn_port",
                "port": 8002,
                "available": False,
                "conflict": True,
                "reason": "in_use",
                "message": "Port 8002 is not available on 127.0.0.1: in_use",
            }
        ]
        assert {item["port"]: item["available"] for item in port_preflight["ports"]} == {8002: False, 8003: True}

    def test_prepare_poc_malformed_participants_returns_invalid_args(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import prepare_poc

        args = self._make_prepare_args(force=True)
        poc_ws = str(tmp_path / "poc")

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=poc_ws),
            patch("nvflare.tool.poc.poc_commands._prepare_poc", return_value=True),
            patch("nvflare.tool.install_skills.install_skills", return_value=None),
            patch("nvflare.tool.poc.poc_commands.load_yaml", return_value={"participants": [{"type": "client"}]}),
        ):
            with pytest.raises(SystemExit) as exc_info:
                prepare_poc(args)

        assert exc_info.value.code == 4
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["error_code"] == "INVALID_ARGS"
        assert "client participant missing name" in data["message"]

    def test_prepare_poc_workspace_exists_non_interactive_exits_4(self, tmp_path):
        """prepare_poc exits 4 when workspace exists and stdin is not a tty (no --force)."""
        from nvflare.tool.poc.poc_commands import prepare_poc

        poc_ws = str(tmp_path / "poc_ws")

        args = self._make_prepare_args(force=False)

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=poc_ws),
            patch("nvflare.tool.poc.poc_commands.os.path.exists", return_value=True),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc_info:
                prepare_poc(args)
        assert exc_info.value.code == 4

    def test_prepare_poc_prompt_on_stderr_not_stdout(self, capsys, tmp_path):
        """When workspace exists and user is prompted interactively, prompt appears on stderr."""
        from nvflare.tool.poc.poc_commands import _prepare_poc

        poc_ws = str(tmp_path / "poc_ws")
        tmp_path.joinpath("poc_ws").mkdir()

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.readline.return_value = "N\n"
            result = _prepare_poc([], 2, poc_ws, force=False)

        captured = capsys.readouterr()
        assert result is False
        assert captured.out.strip() == ""
        assert "Preparing POC workspace at" in captured.err
        assert "This will delete poc workspace directory" in captured.err

    # ------------------------------------------------------------------ stop_poc split-stream test

    def test_stop_poc_success_stdout_is_one_json_line(self, capsys, tmp_path):
        """stop_poc success: stdout is exactly one JSON line."""
        from nvflare.tool.poc.poc_commands import stop_poc

        args = self._make_stop_args()

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._stop_poc", return_value=None),
        ):
            stop_poc(args)

        captured = capsys.readouterr()
        stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(stdout_lines) == 1, f"Expected one JSON line on stdout, got: {captured.out!r}"
        data = json.loads(stdout_lines[0])
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["status"] == "stopped"

    def test_stop_poc_no_human_text_on_stdout(self, capsys, tmp_path):
        """stop_poc: no human-readable text leaks to stdout."""
        from nvflare.tool.poc.poc_commands import stop_poc

        args = self._make_stop_args()

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._stop_poc", return_value=None),
        ):
            stop_poc(args)

        captured = capsys.readouterr()
        # stdout must parse as JSON and contain no prose
        data = json.loads(captured.out.strip())
        assert data["status"] == "ok"
        assert data["exit_code"] == 0

    def test_stop_poc_no_wait_reports_shutdown_initiated(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import stop_poc

        args = self._make_stop_args()
        args.no_wait = True

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._stop_poc", return_value=None) as stop,
        ):
            stop_poc(args)

        stop.assert_called_once_with(str(tmp_path), [], [], wait=False)
        data = json.loads(capsys.readouterr().out)
        assert data["data"]["status"] == "shutdown_initiated"

    def test_stop_poc_timeout_exits_connection_failed(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import stop_poc

        args = self._make_stop_args()

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._stop_poc", side_effect=TimeoutError("server did not stop")),
        ):
            with pytest.raises(SystemExit) as exc_info:
                stop_poc(args)

        assert exc_info.value.code == 2
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "CONNECTION_FAILED"
        assert "--no-wait" in data["hint"]

    def test_start_poc_malformed_participants_omits_missing_names(self, capsys, tmp_path):
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
            patch("nvflare.tool.poc.poc_commands._start_poc", return_value=None),
            patch(
                "nvflare.tool.poc.poc_commands.setup_service_config",
                return_value=({"participants": [{"type": "client"}]}, {}),
            ),
        ):
            start_poc(args)

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["status"] == "running"
        assert data["data"]["ready"] is False
        assert data["data"]["clients"] == []

    def test_start_poc_builds_endpoint_info_after_config_load(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import start_poc

        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None
        args.no_wait = True
        project_config = {"participants": [{"type": "client", "name": "site-1"}]}
        service_config = {}
        endpoint_calls = []

        def _endpoint_info(project, service):
            endpoint_calls.append((project, service))
            return {
                "server_url": "localhost:8002",
                "server_address": "localhost:8002",
                "admin_address": "localhost:8003",
                "default_port": 8002,
                "default_server_port": 8002,
                "default_admin_port": 8003,
            }

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._start_poc", return_value=None),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(project_config, service_config)),
            patch("nvflare.tool.poc.poc_commands._build_poc_endpoint_info", side_effect=_endpoint_info),
        ):
            start_poc(args)

        assert endpoint_calls == [(project_config, service_config)]
        data = json.loads(capsys.readouterr().out)
        assert data["data"]["server_address"] == "localhost:8002"

    def test_start_poc_human_output_omits_agent_diagnostics(self, capsys, monkeypatch, tmp_path):
        from nvflare.tool.poc.poc_commands import start_poc

        monkeypatch.setattr(cli_output, "_output_format", "txt")
        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None
        args.no_wait = False
        project_config = {"participants": [{"type": "client", "name": "site-1"}]}
        service_config = {}
        endpoint_info = {
            "server_url": "grpc://localhost:8002",
            "server_address": "localhost:8002",
            "admin_address": "localhost:8003",
            "default_port": 8002,
            "default_server_port": 8002,
            "default_admin_port": 8003,
        }

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._start_poc", return_value=None),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(project_config, service_config)),
            patch("nvflare.tool.poc.poc_commands._build_poc_endpoint_info", return_value=endpoint_info),
            patch("nvflare.tool.poc.poc_commands._is_local_port_available", return_value=(True, None)),
            patch("nvflare.tool.poc.poc_commands._wait_for_poc_system_ready", return_value=True),
        ):
            start_poc(args)

        captured = capsys.readouterr()
        assert "POC system started. Server: grpc://localhost:8002" in captured.out
        assert "Server address: localhost:8002" in captured.out
        assert "port_preflight:" not in captured.out
        assert "default_server_port:" not in captured.out
        assert "ready_timeout:" not in captured.out
        assert captured.err == ""

    def test_clean_poc_running_system_raises(self, tmp_path):
        """clean_poc should propagate CLIException when the system is still running."""
        from nvflare.tool.poc.poc_commands import clean_poc

        args = MagicMock()

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch(
                "nvflare.tool.poc.poc_commands._clean_poc",
                side_effect=CLIException("system is still running, please stop the system first."),
            ),
        ):
            with pytest.raises(CLIException):
                clean_poc(args)

    def test_clean_poc_invalid_loaded_project_raises(self, tmp_path):
        """clean_poc should raise CLIException when setup_service_config yields no project config."""
        from nvflare.cli_exception import CLIException
        from nvflare.tool.poc.poc_commands import clean_poc

        args = MagicMock()

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.os.path.isdir", return_value=True),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(None, None)),
        ):
            with pytest.raises(CLIException):
                clean_poc(args)

    def test_start_poc_reports_configured_server_port(self, capsys, tmp_path):
        """start_poc should use the configured fed-learn port, not a hard-coded default."""
        from nvflare.lighter.constants import PropKey
        from nvflare.tool.poc.poc_commands import start_poc
        from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None

        project_config = {
            "participants": [
                {"name": "server", "type": "server", PropKey.FED_LEARN_PORT: 9443},
                {"name": "site-1", "type": "client"},
                {"name": "site-2", "type": "client"},
            ]
        }
        service_config = {SC.FLARE_SERVER: "server", SC.FLARE_PROJ_ADMIN: "admin@nvidia.com"}

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._start_poc", return_value=None),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(project_config, service_config)),
            patch("nvflare.tool.poc.poc_commands._is_local_port_available", return_value=(True, None)),
            patch("nvflare.tool.poc.poc_commands._wait_for_poc_system_ready", return_value=True),
        ):
            start_poc(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["server_url"] == "grpc://localhost:9443"
        assert data["data"]["server_address"] == "localhost:9443"
        assert data["data"]["admin_address"] == "localhost:8003"
        assert data["data"]["default_port"] == 8002
        assert data["data"]["default_server_port"] == 8002
        assert data["data"]["default_admin_port"] == 8003
        assert data["data"]["port_conflict"] is False
        assert data["data"]["warnings"] == []
        assert data["data"]["port_preflight"]["checked"] is True
        assert data["data"]["clients"] == ["site-1", "site-2"]
        assert data["data"]["ready"] is True

    def test_start_poc_reports_bound_addresses_and_port_conflicts(self, capsys, tmp_path):
        from nvflare.lighter.constants import PropKey
        from nvflare.tool.poc.poc_commands import start_poc
        from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None
        args.no_wait = False

        project_config = {
            "participants": [
                {
                    "name": "server",
                    "type": "server",
                    PropKey.FED_LEARN_PORT: 8002,
                    PropKey.ADMIN_PORT: 8003,
                },
                {"name": "site-1", "type": "client"},
            ]
        }
        service_config = {
            SC.FLARE_SERVER: "server",
            SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
            SC.FLARE_CLIENTS: ["site-1"],
        }

        def fake_port_available(port, host="127.0.0.1"):
            return (False, "in_use") if port == 8002 else (True, None)

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._start_poc", return_value=None),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(project_config, service_config)),
            patch("nvflare.tool.poc.poc_commands._is_local_port_available", side_effect=fake_port_available),
            patch("nvflare.tool.poc.poc_commands._wait_for_poc_system_ready", return_value=True),
        ):
            start_poc(args)

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "ok"
        assert data["data"]["server_url"] == "grpc://localhost:8002"
        assert data["data"]["server_address"] == "localhost:8002"
        assert data["data"]["admin_address"] == "localhost:8003"
        assert data["data"]["port_conflict"] is True
        assert data["data"]["warnings"] == ["Port 8002 is not available on 127.0.0.1: in_use"]
        assert data["data"]["port_preflight"]["scope"] == "loopback"
        assert "loopback" in data["data"]["port_preflight"]["note"]
        assert data["data"]["port_preflight"]["checked"] is True
        assert data["data"]["port_preflight"]["conflicts"] == [
            {
                "name": "fed_learn_port",
                "port": 8002,
                "available": False,
                "conflict": True,
                "reason": "in_use",
                "message": "Port 8002 is not available on 127.0.0.1: in_use",
            }
        ]
        assert {item["port"]: item["available"] for item in data["data"]["port_preflight"]["ports"]} == {
            8002: False,
            8003: True,
        }
        assert data["data"]["ready"] is True

    def test_start_poc_timeout_controls_readiness_wait(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import start_poc
        from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None
        args.no_wait = False
        args.timeout = 12

        project_config = {"participants": [{"name": "server", "type": "server"}, {"name": "site-1", "type": "client"}]}
        service_config = {
            SC.FLARE_SERVER: "server",
            SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
            SC.FLARE_CLIENTS: ["site-1"],
        }

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._start_poc", return_value=None),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(project_config, service_config)),
            patch("nvflare.tool.poc.poc_commands._is_local_port_available", return_value=(True, None)),
            patch("nvflare.tool.poc.poc_commands._wait_for_poc_system_ready", return_value=True) as wait_ready,
        ):
            start_poc(args)

        assert wait_ready.call_args.kwargs["timeout_in_sec"] == 12
        data = json.loads(capsys.readouterr().out)
        assert data["data"]["ready"] is True
        assert data["data"]["ready_timeout"] == 12

    def test_start_poc_invalid_timeout_exits_4(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import start_poc

        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None
        args.no_wait = False
        args.timeout = 0

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands._start_poc") as start,
        ):
            with pytest.raises(SystemExit) as exc_info:
                start_poc(args)

        assert exc_info.value.code == 4
        start.assert_not_called()
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["error_code"] == "INVALID_ARGS"
        assert "--timeout must be greater than 0 seconds" in data["message"]

    def test_start_poc_error_includes_port_conflict_metadata(self, capsys, tmp_path):
        from nvflare.cli_exception import CLIException
        from nvflare.lighter.constants import PropKey
        from nvflare.tool.poc.poc_commands import start_poc
        from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None
        args.no_wait = False

        project_config = {
            "participants": [
                {
                    "name": "server",
                    "type": "server",
                    PropKey.FED_LEARN_PORT: 8002,
                    PropKey.ADMIN_PORT: 8003,
                }
            ]
        }
        service_config = {
            SC.FLARE_SERVER: "server",
            SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
            SC.FLARE_CLIENTS: [],
        }

        def fake_port_available(port, host="127.0.0.1"):
            return (False, "in_use") if port == 8002 else (True, None)

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._start_poc", side_effect=CLIException("port already in use")),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(project_config, service_config)),
            patch("nvflare.tool.poc.poc_commands._is_local_port_available", side_effect=fake_port_available),
        ):
            with pytest.raises(SystemExit) as exc_info:
                start_poc(args)

        assert exc_info.value.code == 4
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["data"]["port_conflict"] is True
        assert data["data"]["warnings"] == ["Port 8002 is not available on 127.0.0.1: in_use"]
        assert data["data"]["port_preflight"]["scope"] == "loopback"
        assert "loopback" in data["data"]["port_preflight"]["note"]
        assert data["data"]["port_preflight"]["conflicts"][0]["port"] == 8002

    def test_start_poc_no_wait_reports_starting_and_skips_readiness(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import start_poc
        from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None
        args.no_wait = True

        project_config = {"participants": [{"name": "server", "type": "server"}, {"name": "site-1", "type": "client"}]}
        service_config = {
            SC.FLARE_SERVER: "server",
            SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
            SC.FLARE_CLIENTS: ["site-1"],
        }

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._start_poc", return_value=None),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(project_config, service_config)),
            patch("nvflare.tool.poc.poc_commands._is_local_port_available", return_value=(True, None)),
            patch("nvflare.tool.poc.poc_commands._wait_for_poc_system_ready") as wait_ready,
        ):
            start_poc(args)

        wait_ready.assert_not_called()
        data = json.loads(capsys.readouterr().out)
        assert data["data"]["status"] == "starting"
        assert data["data"]["ready"] is False
        assert data["data"]["server_address"] == "localhost:8002"
        assert data["data"]["admin_address"] == "localhost:8003"
        assert data["data"]["default_admin_port"] == 8003
        assert data["data"]["port_conflict"] is False

    def test_start_poc_readiness_timeout_exits_connection_failed(self, capsys, tmp_path):
        from nvflare.tool.api_utils import SystemStartTimeout
        from nvflare.tool.poc.poc_commands import start_poc
        from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

        args = MagicMock()
        args.service = "all"
        args.exclude = ""
        args.gpu = None
        args.study = None
        args.no_wait = False

        project_config = {"participants": [{"name": "server", "type": "server"}, {"name": "site-1", "type": "client"}]}
        service_config = {
            SC.FLARE_SERVER: "server",
            SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
            SC.FLARE_CLIENTS: ["site-1"],
        }

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_gpis", return_value=[]),
            patch("nvflare.tool.poc.poc_commands._start_poc", return_value=None),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(project_config, service_config)),
            patch("nvflare.tool.poc.poc_commands._is_local_port_available", return_value=(True, None)),
            patch(
                "nvflare.tool.poc.poc_commands._wait_for_poc_system_ready",
                side_effect=SystemStartTimeout("cannot connect to server with 1 clients within 30 sec"),
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                start_poc(args)

        assert exc_info.value.code == 2
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "CONNECTION_FAILED"
        assert data["exit_code"] == 2
        assert "--no-wait" in data["hint"]

    def test_wait_for_poc_system_ready_wraps_unexpected_wait_errors(self, tmp_path):
        from nvflare.tool.api_utils import SystemStartTimeout
        from nvflare.tool.poc.poc_commands import _wait_for_poc_system_ready
        from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

        project_config = {"name": "test_project"}
        service_config = {
            SC.FLARE_SERVER: "server",
            SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
            SC.FLARE_CLIENTS: ["site-1"],
        }

        with patch("nvflare.tool.poc.poc_commands.wait_for_system_start", side_effect=RuntimeError("boom")):
            with pytest.raises(SystemStartTimeout, match="boom"):
                _wait_for_poc_system_ready(
                    str(tmp_path),
                    project_config,
                    service_config,
                    services_list=[],
                    excluded=[],
                    timeout_in_sec=1,
                )

    # ------------------------------------------------------------------ poc prepare parsers

    def test_poc_prepare_parser_has_force_flag(self):
        """poc prepare parser should have --force flag."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "prepare", "--force"])
        assert args.force is True

    def test_poc_start_parser_has_no_wait_flag(self):
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "start", "--no-wait"])
        assert args.no_wait is True

    def test_poc_start_parser_has_timeout_flag(self):
        import argparse

        from nvflare.tool.poc.poc_commands import POC_START_READY_TIMEOUT, def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "start"])
        assert args.timeout == POC_START_READY_TIMEOUT

        args = root.parse_args(["poc", "start", "--timeout", "45"])
        assert args.timeout == 45

    def test_poc_stop_parser_has_no_wait_flag(self):
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "stop", "--no-wait"])
        assert args.no_wait is True

    def test_stop_poc_invalid_service_name_exits_4(self, capsys, tmp_path):
        """stop_poc with an unknown -p/--service name exits 4 (structured error), not 1."""
        from nvflare.tool.poc.poc_commands import stop_poc

        args = self._make_stop_args()

        def fake_stop(workspace, excluded, services_list, **_kwargs):
            from nvflare.cli_exception import CLIException

            raise CLIException("participant 'bad-site' is not defined, expecting one of: ['server', 'site-1']")

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.get_excluded", return_value=[]),
            patch("nvflare.tool.poc.poc_commands.get_service_list", return_value=["bad-site"]),
            patch("nvflare.tool.poc.poc_commands._stop_poc", side_effect=fake_stop),
        ):
            with pytest.raises(SystemExit) as exc_info:
                stop_poc(args)
        assert exc_info.value.code == 4

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "INVALID_ARGS"

    def test_poc_prepare_parser_has_schema_flag(self):
        """poc prepare parser should have --schema flag."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "prepare", "--schema"])
        assert args.schema is True

    def test_poc_start_schema_includes_command_contract_metadata(self, capsys):
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser, start_poc

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        with patch("sys.argv", ["nvflare", "poc", "start", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                start_poc(MagicMock())

        assert exc_info.value.code == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema["command"] == "nvflare poc start"
        assert schema["output_modes"] == ["json"]
        assert schema["streaming"] is False
        assert schema["mutating"] is True

    def test_poc_root_schema_outputs_schema_json(self, capsys):
        """nvflare poc --schema should render root schema instead of touching runtime-only flags."""
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser, handle_poc_cmd

        root = argparse.ArgumentParser(prog="nvflare")
        subs = root.add_subparsers(dest="sub_command")
        def_poc_parser(subs)

        with patch("sys.argv", ["nvflare", "poc", "--schema"]):
            with pytest.raises(SystemExit) as exc_info:
                handle_poc_cmd(
                    argparse.Namespace(
                        poc_sub_cmd=None,
                        _argv=["poc", "--schema"],
                    )
                )

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert schema["command"] == "nvflare poc"

    def test_handle_poc_cmd_unknown_subcommand_exits_4(self, capsys):
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser, handle_poc_cmd

        root = argparse.ArgumentParser(prog="nvflare")
        subs = root.add_subparsers(dest="sub_command")
        def_poc_parser(subs)

        with pytest.raises(SystemExit) as exc_info:
            handle_poc_cmd(argparse.Namespace(poc_sub_cmd="bogus", _argv=["poc", "bogus"]))

        assert exc_info.value.code == 4
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "error"
        assert data["error_code"] == "INVALID_ARGS"

    def test_poc_start_parser_accepts_study(self):
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "start", "-p", "admin@nvidia.com", "--study", "cancer_research"])
        assert args.study == "cancer_research"

    def test_is_docker_run_returns_false_without_static_file_builder(self):
        from nvflare.tool.poc.poc_commands import is_docker_run

        project_config = {
            "builders": [
                {
                    "path": "nvflare.lighter.impl.cert.CertBuilder",
                    "args": {},
                }
            ]
        }

        assert is_docker_run(project_config) is False

    def test_get_service_command_adds_study_only_for_admin_start(self):
        from nvflare.tool.poc.poc_commands import get_service_command
        from nvflare.tool.poc.service_constants import FlareServiceConstants as SC

        service_config = {
            SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
            SC.FLARE_OTHER_ADMINS: [],
            SC.IS_DOCKER_RUN: False,
        }

        admin_cmd = get_service_command(
            SC.CMD_START, "/tmp/prod", "admin@nvidia.com", service_config, study="cancer_research"
        )
        server_cmd = get_service_command(SC.CMD_START, "/tmp/prod", "server", service_config, study="cancer_research")

        assert admin_cmd.endswith("fl_admin.sh --study cancer_research")
        assert server_cmd.endswith("start.sh")
        assert "--study" not in server_cmd
