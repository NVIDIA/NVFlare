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
import os
import subprocess
import sys

import pytest

from tests.hello_pt_test_utils import run_hello_pt_export


def test_export_runner_preserves_script_arguments_and_sibling_imports(tmp_path):
    script = tmp_path / "export job.py"
    (tmp_path / "sibling.py").write_text("VALUE = 'sibling imported'\n")
    script.write_text(
        "import json, sys\n"
        "from sibling import VALUE\n"
        "print(json.dumps([__name__, sys.argv, VALUE]))\n"
        "raise SystemExit(0)\n"
    )
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    arguments = [str(script), "--export", "--data_root", "remote site's cache"]

    result = run_hello_pt_export([sys.executable, *arguments], cwd=working_dir, env=os.environ.copy())

    assert json.loads(result.stdout) == ["__main__", arguments, "sibling imported"]
    assert result.stderr == ""


def test_export_timeout_reports_child_stack_and_captured_output(tmp_path, capsys):
    script = tmp_path / "blocked.py"
    script.write_text(
        "import sys, time\nprint('export started')\nprint('before stall', file=sys.stderr)\ntime.sleep(60)\n"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_hello_pt_export([sys.executable, str(script)], cwd=tmp_path, env=os.environ.copy(), timeout=4)

    diagnostics = capsys.readouterr().err
    assert "export started" in diagnostics
    assert "before stall" in diagnostics
    assert "Timeout (" in diagnostics
    assert f'File "{script}"' in diagnostics


def test_export_failure_reports_child_output_and_preserves_exit_code(tmp_path, capsys):
    script = tmp_path / "failed.py"
    script.write_text("import sys\nprint('export started')\nprint('export failed', file=sys.stderr)\nsys.exit(7)\n")

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_hello_pt_export([sys.executable, str(script)], cwd=tmp_path, env=os.environ.copy())

    assert error.value.returncode == 7
    diagnostics = capsys.readouterr().err
    assert "export started" in diagnostics
    assert "export failed" in diagnostics
