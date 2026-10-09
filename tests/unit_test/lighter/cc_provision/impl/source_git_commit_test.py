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

import runpy
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

GIT_COMMAND = ["git", "rev-parse", "--verify", "HEAD"]


@pytest.fixture
def source_git_probe():
    namespace = runpy.run_path(str(Path(__file__).with_name("conftest.py")))
    return namespace["require_source_git_commit"].__wrapped__


def test_valid_source_commit_proceeds(source_git_probe, monkeypatch):
    result = subprocess.CompletedProcess(GIT_COMMAND, 0, stdout="a" * 40 + "\n", stderr="")
    run = Mock(return_value=result)
    monkeypatch.setattr(subprocess, "run", run)

    assert source_git_probe() is None

    run.assert_called_once()
    args, kwargs = run.call_args
    assert args == (GIT_COMMAND,)
    assert kwargs["cwd"] == Path(__file__).resolve().parents[5]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["env"]["LC_ALL"] == "C"


def test_missing_source_repository_skips(source_git_probe, monkeypatch):
    result = subprocess.CompletedProcess(
        GIT_COMMAND,
        128,
        stdout="",
        stderr="fatal: not a git repository (or any parent up to mount point /home/jenkins)\n",
    )
    monkeypatch.setattr(subprocess, "run", Mock(return_value=result))

    with pytest.raises(pytest.skip.Exception, match="Test requires Git metadata in the source checkout"):
        source_git_probe()


@pytest.mark.parametrize(
    "stderr",
    [
        "fatal: detected dubious ownership in repository at '/source'\n",
        "fatal: Needed a single revision\n",
    ],
    ids=["dubious-ownership", "invalid-head"],
)
def test_unrelated_git_errors_propagate(source_git_probe, monkeypatch, stderr):
    result = subprocess.CompletedProcess(GIT_COMMAND, 128, stdout="", stderr=stderr)
    monkeypatch.setattr(subprocess, "run", Mock(return_value=result))

    with pytest.raises(subprocess.CalledProcessError) as raised:
        source_git_probe()

    assert raised.value.returncode == 128
    assert raised.value.cmd == GIT_COMMAND
    assert raised.value.stderr == stderr


def test_missing_git_executable_propagates(source_git_probe, monkeypatch):
    error = FileNotFoundError(2, "No such file or directory", "git")
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))

    with pytest.raises(FileNotFoundError) as raised:
        source_git_probe()

    assert raised.value is error
