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
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nvflare.tool import cli_output
from nvflare.tool.agent.skill_manager import SkillSource
from nvflare.tool.agent.skill_manifest import build_skill_manifest


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


def test_agent_skills_install_dry_run_json_uses_native_source(capsys, monkeypatch, tmp_path):
    _patch_skill_source(monkeypatch, tmp_path)
    target = tmp_path / "target"

    exit_code = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "install",
            "--agent",
            "codex",
            "--target",
            str(target),
            "--dry-run",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "ok")
    assert payload["data"]["applied"] is False
    assert payload["data"]["source"]["type"] == "editable"
    assert payload["data"]["skills"][0]["name"] == "nvflare-test-skill"
    assert not target.exists()


def test_agent_skills_install_accepts_system_temp_alias_target(capsys, monkeypatch, tmp_path):
    _patch_skill_source(monkeypatch, tmp_path)
    target = os.path.join(tempfile.gettempdir(), f"nvflare-skill-target-{tmp_path.name}")

    exit_code = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "install",
            "--agent",
            "codex",
            "--target",
            target,
            "--dry-run",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "ok")
    assert payload["data"]["target_path"] == str(Path(target).resolve(strict=False))
    assert payload["data"]["applied"] is False


def test_agent_skills_install_and_list_json(capsys, monkeypatch, tmp_path):
    _patch_skill_source(monkeypatch, tmp_path)
    target = tmp_path / "target"

    install_exit = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "install",
            "--agent",
            "codex",
            "--target",
            str(target),
            "--format",
            "json",
        ]
    )
    install_payload = _load_single_stdout_json(capsys.readouterr())
    assert install_exit == 0
    assert install_payload["data"]["applied"] is True
    assert target.joinpath("nvflare-test-skill", "SKILL.md").is_file()

    list_exit = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "list",
            "--agent",
            "codex",
            "--target",
            str(target),
            "--format",
            "json",
        ]
    )
    list_payload = _load_single_stdout_json(capsys.readouterr())
    assert list_exit == 0
    assert list_payload["data"]["available"][0]["name"] == "nvflare-test-skill"
    assert list_payload["data"]["installed"][0]["name"] == "nvflare-test-skill"


def test_agent_skills_install_human_output_is_summarized(capsys, monkeypatch, tmp_path):
    _patch_skill_source(monkeypatch, tmp_path)
    target = tmp_path / "target"

    exit_code = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "install",
            "--agent",
            "codex",
            "--target",
            str(target),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "NVFLARE Agent Skills Install" in captured.out
    assert "agent: codex" in captured.out
    assert "mode: applied" in captured.out
    assert "summary: installed 1" in captured.out
    assert "skills:" in captured.out
    assert "- nvflare-test-skill: installed" in captured.out
    assert "Use --format json for the full machine-readable install plan." in captured.out
    assert "'files':" not in captured.out
    assert "[{'name':" not in captured.out


def test_agent_skills_list_human_output_is_summarized(capsys, monkeypatch, tmp_path):
    _patch_skill_source(monkeypatch, tmp_path)
    target = tmp_path / "target"

    exit_code = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "list",
            "--agent",
            "codex",
            "--target",
            str(target),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "NVFLARE Agent Skills" in captured.out
    assert "agent: codex" in captured.out
    assert "available:" in captured.out
    assert "installed:" in captured.out
    assert "conflicts:" in captured.out
    assert "- nvflare-test-skill" in captured.out
    assert "[{'name':" not in captured.out


def test_removed_agent_skill_report_commands_are_not_advertised(capsys):
    exit_code = _run_main(["nvflare", "agent", "info", "--format", "json"])

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    commands = {item["command"] for item in payload["data"]["commands"]}
    assert "nvflare agent skills evaluate" not in commands
    assert "nvflare agent skills performance" not in commands
    assert "nvflare agent skills benchmark" not in commands


@pytest.mark.parametrize("subcommand", ["evaluate", "performance", "benchmark"])
def test_unsupported_agent_skills_commands_are_rejected(capsys, subcommand):
    exit_code = _run_main(["nvflare", "agent", "skills", subcommand, "--format", "json"])

    assert exit_code == 4
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "error")
    assert payload["error_code"] == "INVALID_ARGS"
    assert payload["data"]["choices"] == ["install", "list"]


@pytest.mark.parametrize("subcommand", ["performance", "benchmark"])
def test_removed_agent_skills_command_schema_is_rejected(capsys, subcommand):
    exit_code = _run_main(["nvflare", "agent", "skills", subcommand, "--schema", "--format", "json"])

    assert exit_code == 4
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "error")
    assert payload["error_code"] == "INVALID_ARGS"
    assert payload["data"]["choices"] == ["install", "list"]


def test_agent_skills_missing_subcommand_json_error_is_non_interactive(capsys):
    exit_code = _run_main(["nvflare", "agent", "skills", "--format", "json"])

    assert exit_code == 4
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "error")
    assert payload["error_code"] == "AGENT_SKILLS_SUBCOMMAND_REQUIRED"
    assert payload["data"] is None


def test_agent_skills_schema_subcommand_does_not_scan_option_values():
    from nvflare.tool.agent.agent_cli import _schema_agent_skills_sub_cmd

    _parse_for_agent_parser()

    assert _schema_agent_skills_sub_cmd(["agent", "skills", "list", "--schema"]) == "list"
    assert _schema_agent_skills_sub_cmd(["agent", "skills", "--schema", "install"]) == "install"
    assert _schema_agent_skills_sub_cmd(["agent", "skills", "--schema", "--format", "json", "install"]) == "install"
    assert _schema_agent_skills_sub_cmd(["agent", "skills", "--schema", "--format=json", "list"]) == "list"
    assert _schema_agent_skills_sub_cmd(["agent", "skills", "--unknown-flag", "install", "--schema"]) == "install"
    assert _schema_agent_skills_sub_cmd(["agent", "skills", "--unknown-flag", "list", "--schema"]) == "list"
    assert _schema_agent_skills_sub_cmd(["agent", "skills", "benchmark", "--schema"]) is None
    assert _schema_agent_skills_sub_cmd(["agent", "--target", "skills", "skills", "install", "--schema"]) == "install"
    assert _schema_agent_skills_sub_cmd(["agent", "skills", "--skill", "benchmark", "--schema"]) is None
    assert _schema_agent_skills_sub_cmd(["agent", "skills", "--schema", "--skill", "install"]) is None


def test_agent_skills_missing_named_skill_is_structured_json_error(capsys, monkeypatch, tmp_path):
    _patch_skill_source(monkeypatch, tmp_path)

    exit_code = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "install",
            "--agent",
            "codex",
            "--target",
            str(tmp_path / "target"),
            "--skill",
            "nvflare-missing",
            "--format",
            "json",
        ]
    )

    assert exit_code == 4
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "error")
    assert payload["error_code"] == "AGENT_SKILL_NOT_FOUND"
    assert "code" not in payload
    assert payload["data"]["missing"] == ["nvflare-missing"]


@pytest.mark.parametrize("subcommand", ["install", "list"])
def test_agent_skills_source_unavailable_is_structured_json_error(capsys, monkeypatch, subcommand):
    from nvflare.tool.agent import skill_manager

    def _raise_source_error():
        raise FileNotFoundError("bundled agent skills must be available from an unpacked filesystem package")

    monkeypatch.setattr(skill_manager, "find_skill_source", _raise_source_error)

    exit_code = _run_main(["nvflare", "agent", "skills", subcommand, "--agent", "codex", "--format", "json"])

    assert exit_code == 1
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "error")
    assert payload["error_code"] == "AGENT_SKILL_SOURCE_UNAVAILABLE"
    assert payload["recovery_category"] == "ENVIRONMENT_FAILURE"
    assert payload["data"] is None
    assert "unpacked filesystem package" in payload["message"]


def test_agent_skills_install_failure_is_structured_json_error(capsys, monkeypatch, tmp_path):
    from nvflare.tool.agent import skill_manager

    monkeypatch.setattr(
        skill_manager,
        "install_skills",
        lambda **_kwargs: {
            "agent": "codex",
            "target_path": str(tmp_path / "target"),
            "requested_skill": None,
            "source": {},
            "available": [],
            "skills": [],
            "conflicts": [],
            "errors": [{"skill": "nvflare-test-skill", "code": "skill_install_failed", "message": "disk full"}],
            "deprecated_skills_skipped": [],
            "missing": [],
            "applied": False,
        },
    )

    exit_code = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "install",
            "--agent",
            "codex",
            "--target",
            str(tmp_path / "target"),
            "--format",
            "json",
        ]
    )

    assert exit_code == 1
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "error")
    assert payload["error_code"] == "AGENT_SKILL_INSTALL_FAILED"
    assert "code" not in payload
    assert payload["recovery_category"] == "FIXABLE_BY_ENV"
    assert payload["data"]["errors"][0]["code"] == "skill_install_failed"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_agent_skills_target_symlink_is_structured_json_error(capsys, monkeypatch, tmp_path):
    _patch_skill_source(monkeypatch, tmp_path)
    actual_target = tmp_path / "actual-target"
    actual_target.mkdir()
    link_target = tmp_path / "link-target"
    link_target.symlink_to(actual_target, target_is_directory=True)

    exit_code = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "install",
            "--agent",
            "codex",
            "--target",
            str(link_target),
            "--dry-run",
            "--format",
            "json",
        ]
    )

    assert exit_code == 4
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "error")
    assert payload["error_code"] == "AGENT_SKILL_TARGET_INVALID"
    assert payload["recovery_category"] == "FIXABLE_BY_CONFIG"
    assert "user-created symlink" in payload["hint"]
    assert payload["data"]["target"] == str(link_target)


def test_agent_skills_install_schema_exits_zero(capsys):
    exit_code = _run_main(["nvflare", "agent", "skills", "install", "--schema"])

    assert exit_code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["command"] == "nvflare agent skills install"
    assert schema["mutating"] is True
    assert schema["output_modes"] == ["json"]
    args_by_name = {arg["name"]: arg for arg in schema["args"]}
    assert args_by_name["--skill"]["required"] is False


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


def test_agent_doctor_json_reports_local_readiness(capsys, monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    exit_code = _run_main(["nvflare", "agent", "doctor", "--format", "json"])

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    _assert_envelope_shape(payload, "ok")
    assert payload["data"]["nvflare"]["import_ok"] is True
    assert payload["data"]["online"] == {"enabled": False, "status": "not_requested"}
    assert any(finding["code"] == "STARTUP_KIT_NOT_CONFIGURED" for finding in payload["data"]["findings"])
    assert not home.joinpath(".nvflare", "config.conf").exists()


def test_agent_doctor_human_output_is_summarized(capsys, monkeypatch, tmp_path):
    home = tmp_path / "home"
    poc_workspace = tmp_path / "poc"
    poc_workspace.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NVFLARE_POC_WORKSPACE", str(poc_workspace))
    monkeypatch.delenv("NVFLARE_STARTUP_KIT_DIR", raising=False)

    exit_code = _run_main(["nvflare", "agent", "doctor"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "NVFLARE Agent Doctor" in captured.out
    assert "status: attention" in captured.out
    assert "startup kits: 0/0 valid (active: none)" in captured.out
    assert "findings (1):" in captured.out
    assert "STARTUP_KIT_NOT_CONFIGURED" in captured.out
    assert "startup_kits:" not in captured.out
    assert "{'import_ok':" not in captured.out


def test_agent_doctor_schema_exits_zero(capsys):
    exit_code = _run_main(["nvflare", "agent", "doctor", "--schema"])

    assert exit_code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["command"] == "nvflare agent doctor"
    assert schema["mutating"] is False
    assert schema["output_modes"] == ["json"]
    assert any(arg["name"] == "--online" for arg in schema["args"])


def _patch_skill_source(
    monkeypatch, tmp_path, *, with_behavior=False, with_compound_trigger=False, with_no_skill_trigger=False
):
    from nvflare.tool.agent import skill_manager

    root = tmp_path / "skills"
    _write_skill(
        root,
        "nvflare-test-skill",
        with_behavior=with_behavior,
        with_compound_trigger=with_compound_trigger,
        with_no_skill_trigger=with_no_skill_trigger,
    )
    source = SkillSource(
        source_type="editable",
        root=root,
        manifest=build_skill_manifest(root, source_type="editable", nvflare_version="2.8.0"),
    )
    monkeypatch.setattr(skill_manager, "find_skill_source", lambda: source)
    return source


def _write_skill(root, name, *, with_behavior=False, with_compound_trigger=False, with_no_skill_trigger=False):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Test skill fixture.\n"
        'min_flare_version: "2.8.0"\n'
        "blast_radius: read_only\n"
        "---\n"
        "\n"
        "# Test Skill\n",
        encoding="utf-8",
    )
    evals_dir = skill_dir / "evals"
    evals_dir.mkdir()
    nvflare = {
        "expected_skill": name,
        "process_metrics": [
            {
                "id": "elapsed_seconds",
                "description": "time used for the conversion",
            },
            {
                "id": "token_count",
                "description": "total conversation token count when available",
            },
            {
                "id": "user_correction_count",
                "description": "number of user corrections",
            },
            {
                "id": "missed_instruction_count",
                "description": "number of applicable explicit instructions the agent missed",
            },
            {
                "id": "conversion_quality",
                "description": "reviewer-rated conversion quality",
            },
        ],
    }
    if with_behavior:
        nvflare.update(
            {
                "mandatory_behavior": [{"id": "inspect-first", "description": "inspect local code before editing"}],
                "prohibited_behavior": [{"id": "no-production-submit", "description": "do not submit to production"}],
            }
        )
    eval_cases = [
        {
            "id": "test-conversion",
            "prompt": "Convert this training code.",
            "expected_output": "A validated conversion.",
            "nvflare": nvflare,
        }
    ]
    if with_compound_trigger:
        eval_cases.extend(
            [
                {
                    "id": "compound-trigger",
                    "prompt": "Route this to the test skill and not the other skill.",
                    "expected_output": "The selected skill is nvflare-test-skill.",
                    "nvflare": {
                        "expected_skill": name,
                        "negative_for": "nvflare-other-skill",
                    },
                },
                {
                    "id": "invalid-compound-trigger",
                    "prompt": "Invalid trigger fixture.",
                    "expected_output": "This fixture is invalid by design.",
                    "nvflare": {
                        "expected_skill": name,
                        "negative_for": name,
                    },
                },
            ]
        )
    if with_no_skill_trigger:
        eval_cases.append(
            {
                "id": "global-negative",
                "prompt": "A non-FLARE task should not trigger this skill.",
                "expected_output": "No FLARE skill is selected.",
                "nvflare": {
                    "expected_skill": None,
                    "negative_for": "*",
                },
            }
        )
    evals_dir.joinpath("evals.json").write_text(
        json.dumps(
            {
                "skill_name": name,
                "evals": eval_cases,
            }
        ),
        encoding="utf-8",
    )
    return skill_dir
