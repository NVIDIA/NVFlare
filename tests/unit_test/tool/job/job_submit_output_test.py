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

from nvflare.tool import cli_output
from nvflare.tool.job.job_cli import def_job_cli_parser


class TestJobSubmitOutput:
    """Tests for nvflare job submit output format."""

    @pytest.fixture(autouse=True)
    def agent_mode(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.study = kwargs.get("study", "default")
        args.debug = False
        args.job_folder = kwargs.get("job_folder", "/fake/job")
        args.config_file = None
        args.target = kwargs.get("target", None)
        args.startup_kit = kwargs.get("startup_kit", None)
        return args

    def test_json_envelope_on_success(self, capsys):
        """On success, output_ok emits JSON envelope with job_id."""
        from nvflare.tool.cli_output import output_ok

        output_ok({"job_id": "abc123"})
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "1"
        assert data["status"] == "ok"
        assert data["exit_code"] == 0
        assert data["data"]["job_id"] == "abc123"

    def test_output_error_exits_with_code(self):
        """output_error should raise SystemExit with given exit_code."""
        from nvflare.tool.cli_output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("JOB_INVALID", exit_code=1)
        assert exc_info.value.code == 1

    def test_connection_failed_exits_2(self):
        """CONNECTION_FAILED should exit with code 2."""
        from nvflare.tool.cli_output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("CONNECTION_FAILED", exit_code=2)
        assert exc_info.value.code == 2

    def test_timeout_exits_3(self):
        """TIMEOUT should exit with code 3."""
        from nvflare.tool.cli_output import output_error

        with pytest.raises(SystemExit) as exc_info:
            output_error("TIMEOUT", exit_code=3)
        assert exc_info.value.code == 3

    def test_internal_submit_job_json_mode_keeps_stdout_clean(self, capsys):
        from nvflare.tool.job.job_cli import internal_submit_job

        fake_session = MagicMock()
        fake_session.submit_job.return_value = "abc123"

        with patch("nvflare.tool.job.job_cli.new_cli_session", return_value=fake_session):
            internal_submit_job("/tmp/startup", "admin@nvidia.com", "/tmp/job")

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert data["data"]["job_id"] == "abc123"
        assert "trying to connect to the server" not in captured.out
        assert "trying to connect to the server" not in captured.err

    def test_internal_submit_job_protocol_error_is_not_misreported_as_job_invalid(self, capsys):
        from nvflare.fuel.flare_api.api_spec import InternalError
        from nvflare.tool.job.job_cli import internal_submit_job

        fake_session = MagicMock()
        fake_session.submit_job.side_effect = InternalError("protocol error: ERROR_SYNTAX")

        with patch("nvflare.tool.job.job_cli.new_cli_session", return_value=fake_session):
            with pytest.raises(SystemExit) as exc_info:
                internal_submit_job("/tmp/startup", "admin@nvidia.com", "/tmp/job")

        assert exc_info.value.code == 5
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "INTERNAL_ERROR"
        assert "ERROR_SYNTAX" in data["message"]

    def test_internal_submit_job_authorization_error_maps_to_auth_failed(self, capsys):
        from nvflare.fuel.flare_api.api_spec import AuthorizationError
        from nvflare.tool.job.job_cli import internal_submit_job

        fake_session = MagicMock()
        fake_session.submit_job.side_effect = AuthorizationError("user not authorized for the action 'submit_job'")

        with patch("nvflare.tool.job.job_cli.new_cli_session", return_value=fake_session):
            with pytest.raises(SystemExit) as exc_info:
                internal_submit_job("/tmp/startup", "admin@nvidia.com", "/tmp/job")

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error_code"] == "AUTH_FAILED"
        assert "not authorized" in data["message"].lower()

    def test_internal_submit_job_forwards_all_study_literal(self):
        from nvflare.tool.job.job_cli import internal_submit_job

        fake_session = MagicMock()
        fake_session.submit_job.return_value = "abc123"
        cmd_args = MagicMock()
        cmd_args.study = "all"

        with patch("nvflare.tool.job.job_cli.new_cli_session", return_value=fake_session) as new_session:
            internal_submit_job("/tmp/startup", "admin@nvidia.com", "/tmp/job", cmd_args=cmd_args)

        assert new_session.call_args.kwargs["study"] == "all"

    def test_internal_submit_job_exits_before_output_ok_when_output_error_is_mocked(self):
        from nvflare.fuel.flare_api.api_spec import InvalidJobDefinition
        from nvflare.tool.job.job_cli import internal_submit_job

        fake_session = MagicMock()
        fake_session.submit_job.side_effect = InvalidJobDefinition("bad job")

        with patch("nvflare.tool.job.job_cli.new_cli_session", return_value=fake_session):
            with patch("nvflare.tool.cli_output.output_error") as output_error:
                with patch("nvflare.tool.cli_output.output_ok") as output_ok:
                    with pytest.raises(SystemExit) as exc_info:
                        internal_submit_job("/tmp/startup", "admin@nvidia.com", "/tmp/job")

        assert exc_info.value.code == 1
        output_error.assert_called_once()
        output_ok.assert_not_called()

    def test_submit_parser_no_longer_accepts_wait_or_timeout(self):
        root = argparse.ArgumentParser()
        parser = def_job_cli_parser(root.add_subparsers(dest="sub_command"))["job"]

        with pytest.raises(SystemExit):
            parser.parse_args(["submit", "-j", "./my_job", "--wait"])

        with pytest.raises(SystemExit):
            parser.parse_args(["submit", "-j", "./my_job", "--timeout", "30"])

    def test_submit_parser_no_longer_accepts_config_file(self):
        root = argparse.ArgumentParser()
        parser = def_job_cli_parser(root.add_subparsers(dest="sub_command"))["job"]

        with pytest.raises(SystemExit):
            parser.parse_args(["submit", "-j", "./my_job", "-f", "config_fed_server.conf", "num_rounds=1"])

    def test_submit_parser_accepts_startup_target_or_startup_kit(self):
        root = argparse.ArgumentParser()
        parser = def_job_cli_parser(root.add_subparsers(dest="sub_command"))["job"]

        args = parser.parse_args(["submit", "-j", "./my_job", "--startup-target", "prod"])
        assert args.startup_target == "prod"
        assert args.startup_kit is None

        args = parser.parse_args(["submit", "-j", "./my_job", "--startup_kit", "/tmp/startup"])
        assert args.startup_kit == "/tmp/startup"
        assert args.startup_target is None

    def test_submit_parser_rejects_startup_target_with_startup_kit(self):
        root = argparse.ArgumentParser()
        parser = def_job_cli_parser(root.add_subparsers(dest="sub_command"))["job"]

        with pytest.raises(SystemExit):
            parser.parse_args(["submit", "-j", "./my_job", "--startup-target", "prod", "--startup_kit", "/tmp/startup"])

    def test_submit_parser_rejects_legacy_target_alias(self):
        root = argparse.ArgumentParser()
        parser = def_job_cli_parser(root.add_subparsers(dest="sub_command"))["job"]

        with pytest.raises(SystemExit):
            parser.parse_args(["submit", "-j", "./my_job", "--target", "prod"])
