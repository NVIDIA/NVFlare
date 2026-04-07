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

from nvflare.tool.cli_output import SCHEMA_VERSION, output, output_error


class TestOutput:
    # --- json format ---

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

    # --- quiet format ---

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

    # --- default (table) format ---

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
        # header separator present
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
        # Columns should be padded — rows should have consistent spacing
        lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(lines) >= 3  # header + separator + 2 rows


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
