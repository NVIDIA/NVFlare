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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nvflare.tool import cli_output


@pytest.fixture(autouse=True)
def reset_cli_output_state(monkeypatch):
    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr(cli_output, "_connect_timeout", 5.0)


def _run_main(argv):
    from nvflare import cli

    with patch("sys.argv", argv), patch("nvflare.cli.version_check"):
        try:
            cli.main()
        except SystemExit as e:
            return e.code
    return 0


def _parse_for_agent_parser():
    from nvflare import cli

    with patch("sys.argv", ["nvflare", "agent", "--schema"]):
        _prog_parser, args, sub_cmd_parsers = cli.parse_args("nvflare")

    assert args.sub_command == "agent"
    assert "agent" in sub_cmd_parsers
    return sub_cmd_parsers["agent"]


def _subparser_choices(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _required_operational_args(parser):
    required_args = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        if action.dest == "schema" or action.help == argparse.SUPPRESS:
            continue
        positional_required = not action.option_strings and action.nargs not in (
            argparse.OPTIONAL,
            argparse.ZERO_OR_MORE,
        )
        optional_required = bool(getattr(action, "required", False))
        if positional_required or optional_required:
            required_args.append(action.dest)
    return required_args


def _minimal_agent_command():
    agent_parser = _parse_for_agent_parser()
    choices = _subparser_choices(agent_parser)
    assert choices, "nvflare agent should register at least one read-only subcommand"

    for name, parser in choices.items():
        if _required_operational_args(parser):
            continue
        return name, parser

    assert False, "nvflare agent should expose a read-only subcommand that needs no operational arguments"


def _load_single_stdout_json(captured):
    stdout = captured.out.strip()
    assert stdout
    assert len(stdout.splitlines()) == 1
    return json.loads(stdout)


def _assert_envelope_shape(payload, expected_status, require_data=True):
    assert payload["schema_version"] == "1"
    assert payload["status"] == expected_status
    assert "message" in payload
    assert "hint" in payload
    if require_data:
        assert "data" in payload
    if "code" in payload:
        assert payload["code"]
    else:
        assert payload["error_code"]


def test_agent_command_parser_is_registered(monkeypatch):
    from nvflare import cli

    command_name, _parser = _minimal_agent_command()

    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["nvflare", "agent", command_name, "--format", "json"],
    )

    _prog_parser, args, _sub_cmd_parsers = cli.parse_args("nvflare")
    assert args.sub_command == "agent"


def test_agent_success_envelope_fields_are_supported(capsys, monkeypatch):
    monkeypatch.setattr(cli_output, "_output_format", "json")

    cli_output.output_ok(
        {"ready": True},
        code="AGENT_OK",
        message="Agent command completed.",
        hint="",
        recovery_category="FIXABLE_BY_CONFIG",
        suggested_skill="agent",
    )

    payload = _load_single_stdout_json(capsys.readouterr())
    assert payload["schema_version"] == "1"
    assert payload["status"] == "ok"
    assert payload["code"] == "AGENT_OK"
    assert payload["message"] == "Agent command completed."
    assert payload["hint"] == ""
    assert payload["recovery_category"] == "FIXABLE_BY_CONFIG"
    assert payload["suggested_skill"] == "agent"
    assert payload["data"] == {"ready": True}


def test_agent_error_envelope_fields_are_supported(capsys, monkeypatch):
    monkeypatch.setattr(cli_output, "_output_format", "json")

    with pytest.raises(SystemExit) as exc_info:
        cli_output.output_error_message(
            "AGENT_ERROR",
            "Agent command failed.",
            hint="Use a valid agent command.",
            exit_code=4,
            recovery_category="FIXABLE_BY_CONFIG",
            suggested_skill="agent",
        )

    assert exc_info.value.code == 4
    payload = _load_single_stdout_json(capsys.readouterr())
    assert payload["schema_version"] == "1"
    assert payload["status"] == "error"
    assert payload["error_code"] == "AGENT_ERROR"
    assert "code" not in payload
    assert payload["message"] == "Agent command failed."
    assert payload["hint"] == "Use a valid agent command."
    assert payload["recovery_category"] == "FIXABLE_BY_CONFIG"
    assert payload["suggested_skill"] == "agent"


def test_agent_schema_exits_zero_and_emits_raw_schema_json(capsys):
    exit_code = _run_main(["nvflare", "agent", "--schema"])

    assert exit_code == 0
    captured = capsys.readouterr()
    schema = json.loads(captured.out)
    assert captured.err == ""
    assert schema["schema_version"] == "1"
    assert schema["command"].startswith("nvflare agent")
    assert "status" not in schema
    assert "args" in schema
    assert "examples" in schema


def test_agent_minimal_subcommand_schema_does_not_require_operational_args(capsys):
    command_name, _parser = _minimal_agent_command()

    exit_code = _run_main(["nvflare", "agent", command_name, "--schema"])

    assert exit_code == 0
    captured = capsys.readouterr()
    schema = json.loads(captured.out)
    assert captured.err == ""
    assert schema["schema_version"] == "1"
    assert schema["command"] == f"nvflare agent {command_name}"
    assert "status" not in schema
    assert schema["output_modes"] == ["json"]
    assert schema["streaming"] is False
    assert schema["mutating"] is False
    assert schema["idempotent"] is True


def test_agent_minimal_subcommand_json_success_is_single_stdout_envelope(capsys):
    command_name, _parser = _minimal_agent_command()

    exit_code = _run_main(["nvflare", "agent", command_name, "--format", "json"])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = _load_single_stdout_json(captured)
    _assert_envelope_shape(payload, "ok")


def test_agent_human_output_stays_off_stdout_in_json_mode(capsys):
    command_name, _parser = _minimal_agent_command()

    exit_code = _run_main(["nvflare", "agent", command_name, "--format", "json"])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = _load_single_stdout_json(captured)
    assert payload["status"] == "ok"


def test_agent_missing_subcommand_json_error_is_non_interactive(capsys):
    exit_code = _run_main(["nvflare", "agent", "--format", "json"])

    assert exit_code == 4
    captured = capsys.readouterr()
    payload = _load_single_stdout_json(captured)
    _assert_envelope_shape(payload, "error")
    assert payload["data"] is None
    assert captured.err == ""


def test_agent_missing_subcommand_stops_when_error_helper_is_mocked():
    from nvflare.tool.agent.agent_cli import handle_agent_cmd

    with patch("sys.argv", ["nvflare", "agent"]), patch("nvflare.tool.cli_output.output_error_message") as output_error:
        handle_agent_cmd(SimpleNamespace(agent_sub_cmd=None))

    output_error.assert_called_once()


def test_agent_invalid_subcommand_json_error_is_structured(capsys):
    exit_code = _run_main(["nvflare", "agent", "not-a-pr1-command", "--format", "json"])

    assert exit_code == 4
    captured = capsys.readouterr()
    payload = _load_single_stdout_json(captured)
    _assert_envelope_shape(payload, "error")
    assert "event" not in payload
    assert "terminal" not in payload
    assert "not-a-pr1-command" in payload["message"]
    assert captured.err == ""


def test_agent_info_does_not_advertise_removed_skills_commands(capsys):
    exit_code = _run_main(["nvflare", "agent", "info", "--format", "json"])

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    commands = {item["command"] for item in payload["data"]["commands"]}
    assert commands == {"nvflare agent info", "nvflare agent inspect"}
    assert not any(command.startswith("nvflare agent skills") for command in commands)


def test_agent_info_does_not_register_skills_subcommand():
    agent_parser = _parse_for_agent_parser()
    choices = _subparser_choices(agent_parser)
    assert set(choices) == {"info", "inspect"}


def test_agent_inspect_json_reports_static_framework_evidence(capsys, tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import torch\n\ndef train():\n    return None\n", encoding="utf-8")

    exit_code = _run_main(["nvflare", "agent", "inspect", str(script), "--format", "json"])

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "ok")
    assert payload["data"]["static_only"] is True
    assert payload["data"]["frameworks"][0]["name"] == "pytorch"
    assert payload["data"]["conversion_state"] == "not_converted"
    assert payload["data"]["skill_selection"]["recommended_skills"] == ["nvflare-convert-pytorch"]


def test_agent_inspect_json_reports_lightning_evidence_and_recommends_lightning_skill(capsys, tmp_path):
    script = tmp_path / "train_lightning.py"
    script.write_text(
        "from lightning.pytorch import LightningModule, Trainer\n"
        "\n"
        "class Net(LightningModule):\n"
        "    pass\n"
        "\n"
        "trainer = Trainer(max_epochs=1)\n",
        encoding="utf-8",
    )

    exit_code = _run_main(["nvflare", "agent", "inspect", str(script), "--format", "json"])

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "ok")
    assert payload["data"]["frameworks"][0]["name"] == "pytorch_lightning"
    assert payload["data"]["conversion_state"] == "not_converted"
    assert payload["data"]["skill_selection"]["recommended_skills"] == ["nvflare-convert-lightning"]


def test_agent_inspect_json_classifies_aliased_lightning_patch_import_as_converted(capsys, tmp_path):
    script = tmp_path / "client_lightning.py"
    script.write_text(
        "import lightning as L\n"
        "from nvflare.client.lightning import patch as flare_patch\n"
        "\n"
        "trainer = L.Trainer(max_epochs=1)\n"
        "flare_patch(trainer)\n",
        encoding="utf-8",
    )

    exit_code = _run_main(["nvflare", "agent", "inspect", str(script), "--format", "json"])

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "ok")
    assert payload["data"]["frameworks"][0]["name"] == "pytorch_lightning"
    assert payload["data"]["flare_integration"]["calls"] == ["flare_patch"]
    assert payload["data"]["conversion_state"] == "client_api_converted"


def test_agent_inspect_reports_installed_skills_block(capsys, tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import torch\n", encoding="utf-8")

    exit_code = _run_main(["nvflare", "agent", "inspect", str(script), "--format", "json"])

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "ok")
    assert "installed_skills" in payload["data"]
    assert isinstance(payload["data"]["installed_skills"], list)


def test_agent_inspect_missing_path_is_structured_json_error(capsys, tmp_path):
    exit_code = _run_main(["nvflare", "agent", "inspect", str(tmp_path / "missing"), "--format", "json"])

    assert exit_code == 4
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "error")
    assert payload["error_code"] == "AGENT_INSPECT_PATH_NOT_FOUND"
    assert payload["data"] is None


def test_agent_inspect_schema_exits_zero(capsys):
    exit_code = _run_main(["nvflare", "agent", "inspect", "--schema"])

    assert exit_code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["command"] == "nvflare agent inspect"
    assert schema["mutating"] is False
    assert schema["output_modes"] == ["json"]
    path_arg = next(arg for arg in schema["args"] if arg["name"] == "path")
    assert path_arg["required"] is True
