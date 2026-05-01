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
from types import SimpleNamespace

import pytest

from nvflare.tool import cli_output


def test_unrecognized_argument_human_mode_prints_help_then_error(capsys, monkeypatch):
    from nvflare import cli as cli_mod

    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr(
        cli_mod.sys,
        "argv",
        ["nvflare", "package", "--project_name", "example_project"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.parse_args("nvflare")
    assert exc_info.value.code == 4

    captured = capsys.readouterr()
    assert "usage: nvflare package" in captured.err
    assert "Invalid arguments. — unrecognized arguments: --project_name example_project" in captured.err
    assert "Hint: Run with -h for usage." in captured.err
    assert "Code: INVALID_ARGS (exit 4)" in captured.err
    assert captured.err.index("usage: nvflare package") < captured.err.index("Invalid arguments.")


def test_invalid_subcommand_json_error(capsys, monkeypatch):
    from nvflare import cli as cli_mod

    monkeypatch.setattr(
        cli_mod.sys,
        "argv",
        ["nvflare", "job", "lis", "--format", "json"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.parse_args("nvflare")
    assert exc_info.value.code == 4

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["error_code"] == "INVALID_ARGS"
    assert "usage" in payload["data"]
    assert "list" in payload["data"]["choices"]


def test_display_unknown_args_preserves_consumed_unknown_option_value():
    from nvflare import cli as cli_mod

    args = SimpleNamespace(input="example_project")

    result = cli_mod._display_unknown_args(
        ["package", "--project_name", "example_project"],
        cli_mod.CMD_PACKAGE,
        args,
        ["--project_name"],
    )

    assert result == ["--project_name", "example_project"]


def test_display_unknown_args_keeps_signed_zip_as_positional_input():
    from nvflare import cli as cli_mod

    args = SimpleNamespace(input="hospital.signed.zip")

    result = cli_mod._display_unknown_args(
        ["package", "--unknown-flag", "hospital.signed.zip"],
        cli_mod.CMD_PACKAGE,
        args,
        ["--unknown-flag"],
    )

    assert result == ["--unknown-flag"]


def test_jsonl_rejected_for_non_streaming_command(capsys, monkeypatch):
    from nvflare import cli as cli_mod

    monkeypatch.setattr(
        cli_mod.sys,
        "argv",
        ["nvflare", "job", "wait", "abc123", "--format", "jsonl"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.parse_args("nvflare")
    assert exc_info.value.code == 4

    payload = json.loads(capsys.readouterr().out)
    assert payload["event"] == "terminal"
    assert payload["terminal"] is True
    assert payload["error_code"] == "INVALID_ARGS"
    assert "nvflare job monitor" in payload["message"]
