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

"""Offline tests of callable deployment helpers; no external operations."""

import shutil
import subprocess
from pathlib import Path

import pytest

from nvflare.lighter.cc_provision.workload_security import normalize_resources
from tests.coco_test_utils.shell import require_coco_bash

ROOT = Path(__file__).resolve().parents[5] / "examples/devops/coco"


@pytest.mark.parametrize("available,returncode", [(False, 0), (True, 1), (True, 0)])
def test_shell_prerequisite_detection(monkeypatch, available, returncode):
    monkeypatch.setattr(shutil, "which", lambda _: "/test/bash" if available else None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(command, returncode, "test version", "unsupported shell feature")

    monkeypatch.setattr(subprocess, "run", run)
    if available and returncode == 0:
        assert require_coco_bash() == "/test/bash"
    else:
        with pytest.raises(pytest.skip.Exception, match="CoCo shell tests require"):
            require_coco_bash()
    assert len(calls) == int(available)


def test_shell_prerequisite_probe_is_bounded(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/test/bash")

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(pytest.skip.Exception, match="prerequisite probe timed out"):
        require_coco_bash()


def test_unit_process_boundary_rejects_real_launches():
    with pytest.raises(pytest.fail.Exception, match="CoCo unit test attempted to launch a real process"):
        subprocess.run(["this-command-must-never-start"])


@pytest.mark.parametrize("quantity", [1, "1"])
def test_profile_resource_quantities_match_kubernetes(quantity):
    resources = {"limits": {"nvidia.com/pgpu": quantity}}
    assert normalize_resources(resources) == {"limits": {"nvidia.com/pgpu": "1"}, "requests": {"nvidia.com/pgpu": "1"}}
