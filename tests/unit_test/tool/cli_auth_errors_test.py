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

from nvflare import cli as cli_mod
from nvflare.cli_exception import CLIException
from nvflare.fuel.flare_api.api_spec import AuthenticationError, AuthorizationError
from nvflare.tool import cli_output


def test_auth_hint_for_unknown_study():
    assert cli_mod._auth_hint_from_detail("unknown study 'cancer_research'") == (
        "Add the study under 'studies:' in project.yml with api_version: 4, reprovision, redeploy or restart the server, then try again."
    )


def test_auth_hint_for_missing_study_mapping():
    assert cli_mod._auth_hint_from_detail("user 'admin@nvidia.com' is not mapped to study 'cancer_research'") == (
        "Add this user under the study's admins mapping in project.yml, reprovision, redeploy or restart the server, then try again."
    )


def test_auth_hint_for_invalid_study_name():
    assert cli_mod._auth_hint_from_detail("invalid study name 'bad study'") == (
        "Use a valid study name in project.yml, reprovision, redeploy or restart the server, then try again."
    )


def test_auth_hint_defaults_to_credentials():
    assert cli_mod._auth_hint_from_detail("Incorrect user name or password") == "Check startup kit credentials."


def test_auth_hint_uses_cert_specific_guidance_for_cert_error():
    assert cli_mod._auth_hint_from_detail("certificate validation failed") == (
        "Check that the startup kit certificate, key, and root CA match the server trust chain."
    )


def test_auth_hint_uses_structured_auth_code():
    assert cli_mod._auth_hint_from_detail(
        "Incorrect user name or password", auth_code="AUTH_STUDY_USER_NOT_MAPPED"
    ) == (
        "Add this user under the study's admins mapping in project.yml, reprovision, redeploy or restart the server, then try again."
    )


def test_run_uses_study_specific_auth_hint_in_json_mode(capsys):
    auth_error = AuthenticationError("unknown study 'cancer_research'")
    auth_error.auth_code = "AUTH_UNKNOWN_STUDY"

    args = MagicMock()
    args.out_format = "json"
    args.connect_timeout = 5.0
    args.sub_command = "job"
    args.version = False

    with patch.object(cli_mod, "parse_args", return_value=(MagicMock(), args, {})):
        with patch(
            "nvflare.tool.cli_output.set_output_format",
            side_effect=lambda fmt: setattr(cli_output, "_output_format", fmt),
        ):
            with patch("nvflare.tool.cli_output.set_connect_timeout"):
                with patch.object(cli_mod, "handlers", {"job": lambda _args: (_ for _ in ()).throw(auth_error)}):
                    with pytest.raises(SystemExit) as exc_info:
                        cli_mod.run("nvflare")

    assert exc_info.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "AUTH_FAILED"
    assert payload["hint"].startswith("Add the study under 'studies:'")


def test_run_uses_cert_specific_hint_for_error_cert_in_json_mode(capsys):
    auth_error = AuthenticationError("certificate validation failed")

    args = MagicMock()
    args.out_format = "json"
    args.connect_timeout = 5.0
    args.sub_command = "job"
    args.version = False

    with patch.object(cli_mod, "parse_args", return_value=(MagicMock(), args, {})):
        with patch(
            "nvflare.tool.cli_output.set_output_format",
            side_effect=lambda fmt: setattr(cli_output, "_output_format", fmt),
        ):
            with patch("nvflare.tool.cli_output.set_connect_timeout"):
                with patch.object(cli_mod, "handlers", {"job": lambda _args: (_ for _ in ()).throw(auth_error)}):
                    with pytest.raises(SystemExit) as exc_info:
                        cli_mod.run("nvflare")

    assert exc_info.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "AUTH_FAILED"
    assert payload["hint"] == "Check that the startup kit certificate, key, and root CA match the server trust chain."


def test_run_routes_authorization_error_through_auth_failed_envelope(capsys):
    auth_error = AuthorizationError("user not authorized for the action 'configure_job_log'")

    args = MagicMock()
    args.out_format = "json"
    args.connect_timeout = 5.0
    args.sub_command = "job"
    args.version = False

    with patch.object(cli_mod, "parse_args", return_value=(MagicMock(), args, {})):
        with patch(
            "nvflare.tool.cli_output.set_output_format",
            side_effect=lambda fmt: setattr(cli_output, "_output_format", fmt),
        ):
            with patch("nvflare.tool.cli_output.set_connect_timeout"):
                with patch.object(cli_mod, "handlers", {"job": lambda _args: (_ for _ in ()).throw(auth_error)}):
                    with pytest.raises(SystemExit) as exc_info:
                        cli_mod.run("nvflare")

    assert exc_info.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "AUTH_FAILED"
    assert payload["message"] == "Authorization failed. — user not authorized for the action 'configure_job_log'"


def test_run_routes_cli_exception_through_error_envelope(capsys):
    args = MagicMock()
    args.out_format = "json"
    args.connect_timeout = 5.0
    args.sub_command = "job"
    args.version = False

    with patch.object(cli_mod, "parse_args", return_value=(MagicMock(), args, {})):
        with patch(
            "nvflare.tool.cli_output.set_output_format",
            side_effect=lambda fmt: setattr(cli_output, "_output_format", fmt),
        ):
            with patch("nvflare.tool.cli_output.set_connect_timeout"):
                with patch.object(
                    cli_mod, "handlers", {"job": lambda _args: (_ for _ in ()).throw(CLIException("boom"))}
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        cli_mod.run("nvflare")

    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "CLI_ERROR"
    assert payload["message"].endswith("boom")


def test_run_routes_missing_handler_to_invalid_args_envelope(capsys):
    args = MagicMock()
    args.out_format = "json"
    args.connect_timeout = 5.0
    args.sub_command = "bogus"
    args.version = False

    with patch.object(cli_mod, "parse_args", return_value=(MagicMock(), args, {})):
        with patch(
            "nvflare.tool.cli_output.set_output_format",
            side_effect=lambda fmt: setattr(cli_output, "_output_format", fmt),
        ):
            with patch("nvflare.tool.cli_output.set_connect_timeout"):
                with patch.object(cli_mod, "handlers", {}):
                    with pytest.raises(SystemExit) as exc_info:
                        cli_mod.run("nvflare")

    assert exc_info.value.code == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "INVALID_ARGS"
    assert "unknown command: bogus" in payload["message"].lower()
