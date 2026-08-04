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
        except SystemExit as error:
            return error.code
    return 0


def _agent_parser():
    from nvflare import cli

    with patch("sys.argv", ["nvflare", "agent", "--schema"]):
        _program, args, parsers = cli.parse_args("nvflare")
    assert args.sub_command == "agent"
    return parsers["agent"]


def _subparsers(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _load_json(captured):
    output = captured.out.strip()
    assert output and len(output.splitlines()) == 1
    return json.loads(output)


def _assert_envelope(payload, status):
    assert payload["schema_version"] == "1"
    assert payload["status"] == status
    assert "message" in payload and "hint" in payload and "data" in payload


def test_agent_parser_registers_nested_inspection_capabilities():
    choices = _subparsers(_agent_parser())
    assert set(choices) == {"info", "inspect"}
    assert set(_subparsers(choices["inspect"])) == {"data", "source"}


def test_inspect_rejects_non_positive_walk_limits():
    inspect_parser = _subparsers(_agent_parser())["inspect"]
    with pytest.raises(SystemExit):
        inspect_parser.parse_args(["source", "some/path", "--max-files", "-5"])
    with pytest.raises(SystemExit):
        inspect_parser.parse_args(["data", "some/path", "--max-file-bytes", "0"])
    args = inspect_parser.parse_args(["source", "some/path", "--max-files", "400"])
    assert args.max_files == 400


def test_agent_success_and_error_envelope_extensions(capsys, monkeypatch):
    monkeypatch.setattr(cli_output, "_output_format", "json")
    cli_output.output_ok(
        {"ready": True},
        code="AGENT_OK",
        message="done",
        hint="",
        recovery_category="FIXABLE_BY_CONFIG",
        suggested_skill="agent",
    )
    payload = _load_json(capsys.readouterr())
    _assert_envelope(payload, "ok")
    assert payload["data"] == {"ready": True}

    with pytest.raises(SystemExit) as error:
        cli_output.output_error_message("AGENT_ERROR", "failed", hint="fix", exit_code=4, include_data=True)
    assert error.value.code == 4
    payload = _load_json(capsys.readouterr())
    _assert_envelope(payload, "error")


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["nvflare", "agent", "--schema"], "nvflare agent"),
        (["nvflare", "agent", "info", "--schema"], "nvflare agent info"),
        (["nvflare", "agent", "inspect", "--schema"], "nvflare agent inspect"),
        (["nvflare", "agent", "inspect", "source", "--schema"], "nvflare agent inspect source"),
        (["nvflare", "agent", "inspect", "data", "--schema"], "nvflare agent inspect data"),
    ],
)
def test_agent_schema_paths_exit_zero(capsys, argv, command):
    assert _run_main(argv) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["schema_version"] == "1"
    assert schema["command"] == command
    assert schema["mutating"] is False
    assert schema["output_modes"] == ["json"]
    if command in {"nvflare agent inspect source", "nvflare agent inspect data"}:
        args = {item["name"]: item for item in schema["args"]}
        assert args["--max-files"]["type"] == "integer"
        assert args["--max-files"]["default"] == 250
        assert args["--max-file-bytes"]["type"] == "integer"
        assert args["--max-file-bytes"]["default"] == 512 * 1024


def test_agent_info_advertises_concrete_capabilities(capsys):
    assert _run_main(["nvflare", "agent", "info", "--format", "json"]) == 0
    payload = _load_json(capsys.readouterr())
    commands = {item["command"] for item in payload["data"]["commands"]}
    assert commands == {
        "nvflare agent info",
        "nvflare agent inspect data",
        "nvflare agent inspect source",
    }


def test_agent_missing_subcommand_is_structured(capsys):
    assert _run_main(["nvflare", "agent", "--format", "json"]) == 4
    payload = _load_json(capsys.readouterr())
    _assert_envelope(payload, "error")
    assert payload["error_code"] == "AGENT_SUBCOMMAND_REQUIRED"


def test_agent_missing_inspection_capability_is_structured(capsys):
    assert _run_main(["nvflare", "agent", "inspect", "--format", "json"]) == 4
    payload = _load_json(capsys.readouterr())
    _assert_envelope(payload, "error")
    assert payload["error_code"] == "AGENT_INSPECT_CAPABILITY_REQUIRED"


def test_missing_subcommand_returns_after_mocked_error():
    from nvflare.tool.agent.agent_cli import handle_agent_cmd

    with patch("nvflare.tool.cli_output.output_error_message") as output_error:
        handle_agent_cmd(SimpleNamespace(agent_sub_cmd=None))
    output_error.assert_called_once()


def test_invalid_agent_command_and_capability_are_json_errors(capsys):
    assert _run_main(["nvflare", "agent", "unknown", "--format", "json"]) == 4
    assert _load_json(capsys.readouterr())["status"] == "error"
    assert _run_main(["nvflare", "agent", "inspect", "unknown", "--format", "json"]) == 4
    assert _load_json(capsys.readouterr())["status"] == "error"


def test_source_cli_reports_direct_owner(capsys, tmp_path):
    script = tmp_path / "train.py"
    script.write_text("from torch.optim import Adam\no = Adam(params)\no.step()\n", encoding="utf-8")

    assert _run_main(["nvflare", "agent", "inspect", "source", str(script), "--format", "json"]) == 0
    payload = _load_json(capsys.readouterr())
    _assert_envelope(payload, "ok")
    assert payload["data"]["schema_version"] == "3"
    assert payload["data"]["capability"] == "source"
    assert payload["data"]["routing"] == {
        "recommended_skill": "nvflare-convert-pytorch",
        "reason": "clear_owner",
    }


def test_source_cli_reports_lightning_owner(capsys, tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import lightning as L\nt = L.Trainer()\nt.fit(model)\n", encoding="utf-8")

    assert _run_main(["nvflare", "agent", "inspect", "source", str(script), "--format", "json"]) == 0
    payload = _load_json(capsys.readouterr())
    assert payload["data"]["ownership"]["framework"] == "lightning"


def test_source_cli_reports_converted_client_api(capsys, tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import lightning as L\nfrom nvflare.client.lightning import patch\n"
        "t = L.Trainer()\nt.fit(model)\npatch(t)\n",
        encoding="utf-8",
    )

    assert _run_main(["nvflare", "agent", "inspect", "source", str(script), "--format", "json"]) == 0
    payload = _load_json(capsys.readouterr())
    assert payload["data"]["routing"]["reason"] == "already_integrated"


def test_data_cli_reports_dataset(capsys, tmp_path):
    (tmp_path / "data.csv").write_text("age,income\n1,2\n", encoding="utf-8")

    assert _run_main(["nvflare", "agent", "inspect", "data", str(tmp_path), "--format", "json"]) == 0
    payload = _load_json(capsys.readouterr())
    assert payload["data"]["capability"] == "data"
    assert payload["data"]["routing"]["recommended_skill"] == "nvflare-fed-stats"


def test_source_missing_path_is_structured(capsys, tmp_path):
    target = tmp_path / "missing.py"
    assert _run_main(["nvflare", "agent", "inspect", "source", str(target), "--format", "json"]) == 4
    payload = _load_json(capsys.readouterr())
    assert payload["error_code"] == "AGENT_INSPECT_PATH_NOT_FOUND"


def test_data_file_target_is_structured_config_error(capsys, tmp_path):
    target = tmp_path / "data.csv"
    target.write_text("a\n1\n", encoding="utf-8")
    assert _run_main(["nvflare", "agent", "inspect", "data", str(target), "--format", "json"]) == 4
    payload = _load_json(capsys.readouterr())
    assert payload["error_code"] == "AGENT_INSPECT_INVALID_TARGET"


@pytest.mark.parametrize(("redact", "path_visible"), [("on", False), ("off", True)])
def test_unexpected_inspection_error_respects_redaction(capsys, tmp_path, monkeypatch, redact, path_visible):
    discovered_path = tmp_path / "private" / "secret.py"

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"failed while reading {discovered_path}")

    monkeypatch.setattr("nvflare.tool.agent.inspector.inspect_source", fail)

    assert (
        _run_main(
            [
                "nvflare",
                "agent",
                "inspect",
                "source",
                str(tmp_path),
                "--redact",
                redact,
                "--format",
                "json",
            ]
        )
        == 1
    )
    payload = _load_json(capsys.readouterr())
    assert payload["error_code"] == "AGENT_INSPECT_FAILED"
    assert (str(discovered_path) in payload["message"]) is path_visible


def test_ast_recursion_is_reported_without_aborting(capsys, tmp_path, monkeypatch):
    import nvflare.tool.agent.inspection.files as source_files

    script = tmp_path / "generated.py"
    script.write_text("x = 1\n", encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise RecursionError("depth")

    monkeypatch.setattr(source_files, "analyze_tree", fail)
    assert _run_main(["nvflare", "agent", "inspect", "source", str(script), "--format", "json"]) == 0
    payload = _load_json(capsys.readouterr())
    assert payload["data"]["scan"]["complete"] is False
    assert payload["data"]["scan"]["findings"][0]["code"] == "PYTHON_AST_DEPTH_LIMIT"
