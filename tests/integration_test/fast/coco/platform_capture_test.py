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

"""CoCo external-runtime contracts; run serially outside unit workers."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.coco_test_utils.platform_capture import capture_fixture, workflow
from tests.coco_test_utils.process import run_coco_command

pytest.importorskip("tomllib", reason="CoCo deployment helpers require Python 3.11+")

ROOT = Path(__file__).resolve().parents[4]

TRUSTED = ROOT / "examples/devops/coco/trusted_system"


def test_capture_cli_has_no_separate_approval_step(tmp_path):
    result = run_coco_command(
        [sys.executable, str(TRUSTED / "capture-platform.py"), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "output_directory" in result.stdout
    assert "--approve" not in result.stdout
    assert not (TRUSTED / "approve-platform.py").exists()


@pytest.mark.parametrize("available_to_sudo", [False, True])
def test_crictl_probe_checks_the_effective_sudo_python_path(tmp_path, monkeypatch, available_to_sudo):
    caller = tmp_path / "caller-bin"
    privileged = tmp_path / "sudo-bin"
    caller.mkdir()
    privileged.mkdir()
    available = privileged if available_to_sudo else caller
    binary = available / "crictl"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(caller))
    run = run_coco_command

    def effective_sudo(command, **kwargs):
        assert command[:3] == ["sudo", "python3", "-c"]
        # Execute the actual probe program with a distinct effective PATH, as
        # sudo's secure_path does, without requiring sudo privileges in tests.
        return run([sys.executable, *command[2:]], env={**os.environ, "PATH": str(privileged)}, **kwargs)

    with patch.object(workflow.subprocess, "run", side_effect=effective_sudo):
        if available_to_sudo:
            workflow.require_capture_tools()
        else:
            with pytest.raises(ValueError, match=r"Capture requires Python 3\.11\+"):
                workflow.require_capture_tools()


@pytest.mark.parametrize("failure", ["old-python", "helper-import"])
def test_privileged_interpreter_failures_stop_capture_before_side_effects(tmp_path, monkeypatch, failure):
    probe = workflow.require_capture_tools
    output, config, _, _, calls = capture_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(workflow, "require_capture_tools", probe)
    helper_root = tmp_path / "probe-helper"
    (helper_root / "lib").mkdir(parents=True)
    (helper_root / "capture-running-launch.py").write_text("import missing_capture_dependency\n")
    runtime = workflow.runtime_api()
    monkeypatch.setattr(workflow, "runtime_api", lambda: runtime)
    monkeypatch.setattr(workflow, "HERE", helper_root / "lib")
    run = run_coco_command
    diagnostics = []

    def effective_sudo(command, **kwargs):
        program = command[3]
        if failure == "old-python":
            program = "import sys; sys.version_info = (3, 10)\n" + program
        try:
            return run([sys.executable, "-c", program, *command[4:]], **kwargs)
        except subprocess.CalledProcessError as error:
            diagnostics.append(error.stderr)
            raise

    with (
        patch.object(workflow.subprocess, "run", side_effect=effective_sudo),
        patch.object(workflow, "collect") as collect,
    ):
        with pytest.raises(ValueError, match=r"Capture requires Python 3\.11\+"):
            workflow.capture(config, output)
    expected = "Python 3.11+ is required" if failure == "old-python" else "missing_capture_dependency"
    assert len(diagnostics) == 1 and expected in diagnostics[0]
    collect.assert_not_called()
    assert not calls
    assert not output.exists()
