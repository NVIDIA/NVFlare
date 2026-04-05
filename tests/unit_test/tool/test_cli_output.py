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

import pytest

from nvflare.tool.cli_output import SCHEMA_VERSION, output, output_error, output_ok


# --- output() tests (cert/package commands) ---


class TestOutput:
    def test_json_dict(self, capsys):
        output({"key": "value"}, "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["status"] == "ok"
        assert result["data"] == {"key": "value"}

    def test_json_list(self, capsys):
        output(["a", "b", "c"], "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["status"] == "ok"
        assert result["data"] == ["a", "b", "c"]

    def test_json_string(self, capsys):
        output("hello", "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["data"] == "hello"

    def test_json_none_data(self, capsys):
        output(None, "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["status"] == "ok"
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


class TestOutputOkJson:
    def test_envelope_shape(self, capsys):
        output_ok({"key": "value"}, fmt="json")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["schema_version"] == SCHEMA_VERSION
        assert envelope["status"] == "ok"
        assert envelope["data"] == {"key": "value"}

    def test_list_data(self, capsys):
        output_ok([1, 2, 3], fmt="json")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["data"] == [1, 2, 3]

    def test_unknown_fmt_falls_back_to_json(self, capsys):
        output_ok("hello", fmt="xml")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "ok"


class TestOutputOkTxt:
    def test_string_printed_as_is(self, capsys):
        output_ok("hello world", fmt="txt")
        captured = capsys.readouterr()
        assert "hello world" in captured.out
        try:
            json.loads(captured.out)
            pytest.fail("Expected non-JSON output")
        except (json.JSONDecodeError, ValueError):
            pass

    def test_dict_rendered_as_key_value(self, capsys):
        output_ok({"status": "running", "id": "abc"}, fmt="txt")
        captured = capsys.readouterr()
        assert "status: running" in captured.out
        assert "id: abc" in captured.out

    def test_list_of_dicts_rendered_as_table(self, capsys):
        rows = [{"id": "job1", "status": "RUNNING"}, {"id": "job2", "status": "FINISHED_OK"}]
        output_ok(rows, fmt="txt")
        captured = capsys.readouterr()
        assert "id" in captured.out
        assert "status" in captured.out
        assert "-" in captured.out
        assert "job1" in captured.out
        assert "FINISHED_OK" in captured.out


# --- output_error() tests: cert/package pattern (explicit message/hint) ---


class TestOutputError:
    def test_exits_with_code_1_by_default(self):
        with pytest.raises(SystemExit) as exc_info:
            output_error("SOME_ERROR", "Something went wrong.", "Try again.", None)
        assert exc_info.value.code == 1

    def test_exits_with_custom_code(self):
        with pytest.raises(SystemExit) as exc_info:
            output_error("SOME_ERROR", "Something went wrong.", "Try again.", None, exit_code=4)
        assert exc_info.value.code == 4

    def test_stderr_text_format(self, capsys):
        with pytest.raises(SystemExit):
            output_error("MY_CODE", "Error message here.", "Fix hint.", None)
        captured = capsys.readouterr()
        assert "ERROR_CODE: MY_CODE" in captured.err
        assert "Error message here." in captured.err
        assert "Fix hint." in captured.err
        assert captured.out == ""

    def test_json_format_goes_to_stdout(self, capsys):
        with pytest.raises(SystemExit):
            output_error("MY_CODE", "Error message here.", "Fix hint.", "json")
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["status"] == "error"
        assert result["error_code"] == "MY_CODE"
        assert result["message"] == "Error message here."
        assert result["hint"] == "Fix hint."
        assert captured.err == ""

    def test_json_format_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            output_error("MY_CODE", "msg", "hint", "json")
        assert exc_info.value.code == 1


# --- output_error() tests: Phase 0+1 pattern (ERROR_REGISTRY lookup) ---


class TestOutputErrorJson:
    def test_error_envelope_shape(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            output_error("CONNECTION_FAILED", fmt="json")
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["schema_version"] == SCHEMA_VERSION
        assert envelope["status"] == "error"
        assert envelope["error_code"] == "CONNECTION_FAILED"
        assert "message" in envelope
        assert "hint" in envelope

    def test_custom_exit_code(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            output_error("AUTH_FAILED", fmt="json", exit_code=2)
        assert exc_info.value.code == 2

    def test_unknown_error_code_uses_code_as_message(self, capsys):
        with pytest.raises(SystemExit):
            output_error("UNKNOWN_CODE_XYZ", fmt="json")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["error_code"] == "UNKNOWN_CODE_XYZ"
        assert envelope["message"] == "UNKNOWN_CODE_XYZ"
        assert envelope["hint"] == ""


class TestOutputErrorTxt:
    def test_stderr_output(self, capsys):
        with pytest.raises(SystemExit):
            output_error("TIMEOUT", fmt="txt")
        captured = capsys.readouterr()
        assert "ERROR_CODE: TIMEOUT" in captured.err
        assert len(captured.err) > 0

    def test_hint_included(self, capsys):
        with pytest.raises(SystemExit):
            output_error("TIMEOUT", fmt="txt")
        captured = capsys.readouterr()
        assert "Hint:" in captured.err


class TestOutputErrorKwargs:
    def test_format_substitution(self, capsys):
        with pytest.raises(SystemExit):
            output_error("JOB_NOT_FOUND", fmt="json", job_id="abc123")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert "abc123" in envelope["message"]

    def test_missing_substitution_key_uses_template(self, capsys):
        with pytest.raises(SystemExit):
            output_error("JOB_NOT_FOUND", fmt="json", wrong_key="abc")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert "{job_id}" in envelope["message"]


class TestOutputErrorDetail:
    def test_detail_appended_to_message(self, capsys):
        with pytest.raises(SystemExit):
            output_error("INTERNAL_ERROR", fmt="json", detail="something went wrong")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert "something went wrong" in envelope["message"]

    def test_detail_appended_with_separator(self, capsys):
        with pytest.raises(SystemExit):
            output_error("INTERNAL_ERROR", fmt="json", detail="extra context")
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert " \u2014 " in envelope["message"]

    def test_detail_appended_txt_fmt(self, capsys):
        with pytest.raises(SystemExit):
            output_error("INTERNAL_ERROR", fmt="txt", detail="extra context txt")
        captured = capsys.readouterr()
        assert "extra context txt" in captured.err
