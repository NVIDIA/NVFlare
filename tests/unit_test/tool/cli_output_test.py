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
from unittest.mock import patch

import pytest

from nvflare.tool import cli_output
from nvflare.tool.cli_output import (
    SCHEMA_VERSION,
    output,
    output_error,
    output_error_message,
    output_jsonl_event,
    output_ok,
    print_human,
)


@pytest.fixture(autouse=True)
def reset_cli_output_state(monkeypatch):
    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr(cli_output, "_connect_timeout", 5.0)


# --- output() tests (cert/package commands) ---


class TestOutput:
    def test_json_dict(self, capsys):
        output({"key": "value"}, "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["status"] == "ok"
        assert result["exit_code"] == 0
        assert result["data"] == {"key": "value"}

    def test_json_list(self, capsys):
        output(["a", "b", "c"], "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["status"] == "ok"
        assert result["exit_code"] == 0
        assert result["data"] == ["a", "b", "c"]

    def test_json_string(self, capsys):
        output("hello", "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["exit_code"] == 0
        assert result["data"] == "hello"

    def test_json_none_data(self, capsys):
        output(None, "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["status"] == "ok"
        assert result["exit_code"] == 0
        assert result["data"] is None

    def test_quiet_dict(self, capsys):
        output({"first": "val1", "second": "val2"}, "quiet")
        captured = capsys.readouterr()
        assert captured.out.strip() == "val1"

    def test_quiet_list(self, capsys):
        output(["item0", "item1"], "quiet")
        captured = capsys.readouterr()
        assert captured.out.strip() == "item0"

    def test_quiet_empty_list(self, capsys):
        output([], "quiet")
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_quiet_string(self, capsys):
        output("hello world", "quiet")
        captured = capsys.readouterr()
        assert captured.out.strip() == "hello world"

    def test_table_dict(self, capsys):
        output({"name": "Alice", "role": "admin"}, None)
        captured = capsys.readouterr()
        assert "name: Alice" in captured.out
        assert "role: admin" in captured.out

    def test_table_list_of_dicts(self, capsys):
        data = [{"id": "1", "status": "running"}, {"id": "2", "status": "done"}]
        output(data, None)
        captured = capsys.readouterr()
        assert "id" in captured.out
        assert "status" in captured.out
        assert "running" in captured.out
        assert "done" in captured.out
        assert "---" in captured.out

    def test_table_list_of_strings(self, capsys):
        output(["alpha", "beta", "gamma"], None)
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" in captured.out
        assert "gamma" in captured.out

    def test_table_empty_list(self, capsys):
        output([], None)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_table_string(self, capsys):
        output("plain text", None)
        captured = capsys.readouterr()
        assert "plain text" in captured.out

    def test_table_list_of_dicts_column_widths(self, capsys):
        data = [{"name": "short", "value": "a"}, {"name": "a-much-longer-name", "value": "bb"}]
        output(data, None)
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(lines) >= 3  # header + separator + 2 rows


# --- output_ok() tests (Phase 0+1 commands) ---


class TestOutputOk:
    """Tests for output_ok() — JSON envelope in agent mode, human table in default mode."""

    def test_agent_mode_envelope_shape(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        output_ok({"key": "value"})
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["schema_version"] == SCHEMA_VERSION
        assert envelope["status"] == "ok"
        assert envelope["exit_code"] == 0
        assert envelope["data"] == {"key": "value"}

    def test_agent_mode_list_data(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        output_ok([1, 2, 3])
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["exit_code"] == 0
        assert envelope["data"] == [1, 2, 3]

    def test_agent_mode_string_data(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        output_ok("hello")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["data"] == "hello"
        assert envelope["status"] == "ok"
        assert envelope["exit_code"] == 0
        assert captured.err == ""

    def test_agent_mode_human_output_goes_to_stderr(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        print_human("progress message")
        output_ok({"key": "value"})
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["data"] == {"key": "value"}
        assert "progress message" in captured.err

    def test_jsonl_mode_human_output_goes_to_stderr(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")
        print_human("progress message")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "progress message" in captured.err

    def test_jsonl_mode_output_ok_emits_terminal_event(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")
        output_ok({"key": "value"})
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["event"] == "terminal"
        assert payload["status"] == "ok"
        assert payload["terminal"] is True
        assert payload["data"] == {"key": "value"}

    def test_jsonl_event_flushes_stdout(self):
        with patch("builtins.print") as mock_print:
            output_jsonl_event({"event": "progress"})

        mock_print.assert_called_once()
        assert mock_print.call_args.kwargs["flush"] is True

    def test_human_mode_dict_renders_as_table(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        output_ok({"status": "running", "id": "abc"})
        captured = capsys.readouterr()
        assert "status: running" in captured.out
        assert "id: abc" in captured.out

    def test_human_mode_no_json_envelope(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        output_ok({"key": "value"})
        captured = capsys.readouterr()
        try:
            json.loads(captured.out)
            pytest.fail("Expected non-JSON output in human mode")
        except (json.JSONDecodeError, ValueError):
            pass


# --- output_error_message() tests: explicit message/hint/fmt) ---


class TestOutputErrorCertPackage:
    def test_exits_with_code_1_by_default(self):
        with pytest.raises(SystemExit) as exc_info:
            output_error_message("SOME_ERROR", "Something went wrong.", "Try again.", None)
        assert exc_info.value.code == 1

    def test_exits_with_custom_code(self):
        with pytest.raises(SystemExit) as exc_info:
            output_error_message("SOME_ERROR", "Something went wrong.", "Try again.", None, exit_code=4)
        assert exc_info.value.code == 4

    def test_stderr_text_format(self, capsys):
        with pytest.raises(SystemExit):
            output_error_message("MY_CODE", "Error message here.", "Fix hint.", None)
        captured = capsys.readouterr()
        assert "Error message here." in captured.err
        assert "Hint: Fix hint." in captured.err
        assert "Code: MY_CODE (exit 1)" in captured.err
        assert captured.out == ""

    def test_json_format_goes_to_stdout(self, capsys):
        with pytest.raises(SystemExit):
            output_error_message("MY_CODE", "Error message here.", "Fix hint.", "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["status"] == "error"
        assert result["exit_code"] == 1
        assert result["error_code"] == "MY_CODE"
        assert result["message"] == "Error message here."
        assert result["hint"] == "Fix hint."
        assert captured.err == ""

    def test_jsonl_format_goes_to_stdout_as_terminal_event(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")
        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                output_error_message("MY_CODE", "Error message here.", "Fix hint.", None)
        assert exc_info.value.code == 1

        mock_print.assert_called_once()
        assert mock_print.call_args.kwargs["flush"] is True
        result = json.loads(mock_print.call_args.args[0])
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["event"] == "terminal"
        assert result["status"] == "error"
        assert result["terminal"] is True
        assert result["error_code"] == "MY_CODE"

    def test_explicit_jsonl_format_goes_to_stdout_as_terminal_event(self):
        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                output_error_message("MY_CODE", "Error message here.", "Fix hint.", "jsonl")
        assert exc_info.value.code == 1

        mock_print.assert_called_once()
        assert mock_print.call_args.kwargs["flush"] is True
        result = json.loads(mock_print.call_args.args[0])
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["event"] == "terminal"
        assert result["status"] == "error"
        assert result["terminal"] is True
        assert result["error_code"] == "MY_CODE"


class TestOutputErrorWithData:
    def test_json_error_can_include_data(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with pytest.raises(SystemExit) as exc_info:
            output_error("JOB_FAILED", exit_code=1, data={"status": "FAILED", "job_id": "abc123"}, job_id="abc123")
        assert exc_info.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "error"
        assert payload["error_code"] == "JOB_FAILED"
        assert payload["data"] == {"status": "FAILED", "job_id": "abc123"}

    def test_jsonl_error_is_terminal_event(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")
        with pytest.raises(SystemExit) as exc_info:
            output_error("JOB_FAILED", exit_code=1, data={"status": "FAILED", "job_id": "abc123"}, job_id="abc123")
        assert exc_info.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "error"
        assert payload["error_code"] == "JOB_FAILED"
        assert payload["event"] == "terminal"
        assert payload["terminal"] is True

    def test_jsonl_error_flushes_stdout(self, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "jsonl")
        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                output_error("JOB_FAILED", exit_code=1, data={"status": "FAILED"}, job_id="abc123")
        assert exc_info.value.code == 1

        mock_print.assert_called_once()
        assert mock_print.call_args.kwargs["flush"] is True
        payload = json.loads(mock_print.call_args.args[0])
        assert payload["event"] == "terminal"
        assert payload["terminal"] is True

    def test_human_error_with_data_renders_context_then_hint_and_code(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        with pytest.raises(SystemExit) as exc_info:
            output_error(
                "JOB_FAILED",
                exit_code=1,
                hint="Use 'nvflare job logs <job_id>' and 'nvflare job meta <job_id>' to inspect the failure.",
                data={"status": "FAILED", "job_id": "abc123"},
                job_id="abc123",
            )
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "status: FAILED" in captured.out
        assert "job_id: abc123" in captured.out
        assert "Hint: Use 'nvflare job logs <job_id>'" in captured.err
        assert "Code: JOB_FAILED (exit 1)" in captured.err

    def test_json_format_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            output_error_message("MY_CODE", "msg", "hint", "json")
        assert exc_info.value.code == 1


# --- output_error() tests: Phase 0+1 pattern (ERROR_REGISTRY lookup) ---


class TestOutputError:
    """Agent-mode JSON output for Phase 0+1 error paths."""

    def test_error_envelope_shape(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with pytest.raises(SystemExit) as exc_info:
            output_error("CONNECTION_FAILED")
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["schema_version"] == SCHEMA_VERSION
        assert envelope["status"] == "error"
        assert envelope["error_code"] == "CONNECTION_FAILED"
        assert "message" in envelope
        assert "hint" in envelope

    def test_custom_exit_code(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with pytest.raises(SystemExit) as exc_info:
            output_error("AUTH_FAILED", exit_code=2)
        assert exc_info.value.code == 2

    def test_unknown_error_code_uses_code_as_message(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with pytest.raises(SystemExit):
            output_error("UNKNOWN_CODE_XYZ")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "UNKNOWN_CODE_XYZ"
        assert envelope["message"] == "UNKNOWN_CODE_XYZ"
        assert envelope["hint"] == ""

    def test_format_substitution(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with pytest.raises(SystemExit):
            output_error("JOB_NOT_FOUND", job_id="abc123")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert "abc123" in envelope["message"]

    def test_missing_substitution_key_uses_template(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with pytest.raises(SystemExit):
            output_error("JOB_NOT_FOUND", wrong_key="abc")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert "{job_id}" in envelope["message"]

    def test_missing_substitution_key_logs_warning(self, caplog, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with caplog.at_level("WARNING", logger="nvflare.tool.cli_output"):
            with pytest.raises(SystemExit):
                output_error("JOB_NOT_FOUND", wrong_key="abc")
        assert "Missing format key for error JOB_NOT_FOUND" in caplog.text

    def test_detail_appended_to_message(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with pytest.raises(SystemExit):
            output_error("INTERNAL_ERROR", detail="something went wrong")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert "something went wrong" in envelope["message"]

    def test_detail_appended_with_separator(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with pytest.raises(SystemExit):
            output_error("INTERNAL_ERROR", detail="extra context")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert " \u2014 " in envelope["message"]

    def test_agent_mode_output_goes_to_stdout_not_stderr(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "json")
        with pytest.raises(SystemExit):
            output_error("TIMEOUT")
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert captured.err == ""

    def test_human_mode_output_goes_to_stderr(self, capsys, monkeypatch):
        monkeypatch.setattr(cli_output, "_output_format", "txt")
        with pytest.raises(SystemExit):
            output_error("TIMEOUT")
        captured = capsys.readouterr()
        assert "TIMEOUT" in captured.err
        assert captured.out == ""
