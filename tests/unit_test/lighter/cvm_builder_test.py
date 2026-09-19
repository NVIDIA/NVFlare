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

"""Run the standalone Linux builder contracts in the regular NVFlare test suite."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[3] / "nvflare/lighter/cc/image_builder"
TESTS = Path(__file__).resolve().parents[3] / "tests/unit_test/lighter/cc/image_builder"


@pytest.mark.skipif(sys.platform != "linux", reason="CVM Builder requires Linux memfd and /proc interfaces")
def test_cvm_builder_contracts():
    env = dict(os.environ, PYTHONPATH=str(BUILDER))
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(TESTS), "-p", "test_*.py", "-v"],
        cwd=BUILDER,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "entrypoint",
    [
        "",
        "build",
        "vault",
        "pull",
        "publish",
        "finalize",
        "admin approve",
        "admin install",
        "admin retire",
        "admin revoke",
        "provenance",
        "references",
        "preflight host",
        "preflight trustee",
        "inspect-tcb",
    ],
)
def test_cvm_builder_entrypoints(entrypoint):
    result = subprocess.run(
        [
            str(BUILDER / "cvmctl"),
            *entrypoint.split(),
            "--help",
        ],
        cwd=BUILDER.parent,
        env=dict(os.environ, CVM_BUILDER_PYTHON=sys.executable),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout
