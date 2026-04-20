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

    # ------------------------------------------------------------------ prepare_poc split-stream tests

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
        assert data["data"]["clients"] == []

    def test_prepare_jobs_dir_user_decline_emits_no_success(self, capsys, tmp_path):
        """prepare_jobs_dir should not emit output_ok when the user declines replacement."""
        from nvflare.tool.poc.poc_commands import prepare_jobs_dir

        args = MagicMock()
        args.jobs_dir = str(tmp_path / "jobs")
        args.force = False
        os_jobs = tmp_path / "jobs"
        os_jobs.mkdir()

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands._prepare_jobs_dir", return_value=False),
            patch("nvflare.tool.install_skills.install_skills", return_value=None),
        ):
            prepare_jobs_dir(args)

        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_clean_poc_running_system_exits_4(self, capsys, tmp_path):
        """clean_poc should exit 4 instead of reporting cleaned when the system is still running."""
        from nvflare.tool.poc.poc_commands import clean_poc

        args = MagicMock()
        args.force = True

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch(
                "nvflare.tool.poc.poc_commands._clean_poc",
                side_effect=CLIException("system is still running, please stop the system first."),
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                clean_poc(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "INVALID_ARGS"

    def test_clean_poc_invalid_loaded_project_exits_4(self, capsys, tmp_path):
        """clean_poc should not emit cleaned when setup_service_config yields no project config."""
        from nvflare.tool.poc.poc_commands import clean_poc

        args = MagicMock()
        args.force = True

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("nvflare.tool.poc.poc_commands.os.path.isdir", return_value=True),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=(None, None)),
        ):
            with pytest.raises(SystemExit) as exc_info:
                clean_poc(args)

        assert exc_info.value.code == 4
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "INVALID_ARGS"

    def test_clean_poc_non_interactive_without_force_exits_4(self, capsys, tmp_path):
        from nvflare.tool.poc.poc_commands import clean_poc

        args = MagicMock()
        args.force = False

        with (
            patch("nvflare.tool.poc.poc_commands.get_poc_workspace", return_value=str(tmp_path)),
            patch("sys.stdin") as mock_stdin,
            patch("nvflare.tool.poc.poc_commands.os.path.isdir", return_value=True),
            patch("nvflare.tool.poc.poc_commands.setup_service_config", return_value=({"name": "proj"}, {})),
            patch("nvflare.tool.poc.poc_commands.is_poc_ready", return_value=True),
            patch("nvflare.tool.poc.poc_commands.is_poc_running", return_value=False),
        ):
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc_info:
                clean_poc(args)

        assert exc_info.value.code == 4
        data = json.loads(capsys.readouterr().out)
        assert data["error_code"] == "INVALID_ARGS"

    def test_clean_poc_parser_has_force_flag(self):
        import argparse

        from nvflare.tool.poc.poc_commands import def_poc_parser

        root = argparse.ArgumentParser()
        subs = root.add_subparsers()
        def_poc_parser(subs)

        args = root.parse_args(["poc", "clean", "--force"])
        assert args.force is True

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
        ):
            start_poc(args)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["server_url"] == "grpc://localhost:9443"
        assert data["data"]["clients"] == ["site-1", "site-2"]

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

    def test_stop_poc_invalid_service_name_exits_4(self, capsys, tmp_path):
        """stop_poc with an unknown -p/--service name exits 4 (structured error), not 1."""
        from nvflare.tool.poc.poc_commands import stop_poc

        args = self._make_stop_args()

        def fake_stop(workspace, excluded, services_list):
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
