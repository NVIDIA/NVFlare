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

from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project
from nvflare.tool import cli_output


def _make_ctx(tmp_path):
    project = Project(
        "proj",
        "desc",
        [Participant(name="server", org="org", type="server")],
    )
    return ProvisionContext(str(tmp_path), project)


def test_error_and_warning_are_captured_and_suppressed_in_json_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_output, "_output_format", "json")
    ctx = _make_ctx(tmp_path)

    ctx.error("bad thing")
    ctx.warning("careful now")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert ctx.get_errors() == ["bad thing"]
    assert ctx.get_warnings() == ["careful now"]


def test_error_and_warning_print_to_stderr_in_human_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_output, "_output_format", "txt")
    ctx = _make_ctx(tmp_path)

    ctx.error("bad thing")
    ctx.warning("careful now")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ERROR: bad thing" in captured.err
    assert "WARNING: careful now" in captured.err
    assert ctx.get_errors() == ["bad thing"]
    assert ctx.get_warnings() == ["careful now"]
