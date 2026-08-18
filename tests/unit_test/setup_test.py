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

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

FAILING_VERSIONEER = """
def get_versions():
    return {"version": "0+unknown", "error": "unable to compute version"}


def get_cmdclass():
    return {}
"""


def _run_setup_version(tmp_path: Path, base_version: str = None) -> subprocess.CompletedProcess:
    shutil.copy2(ROOT_DIR / "setup.py", tmp_path / "setup.py")
    (tmp_path / "versioneer.py").write_text(FAILING_VERSIONEER)

    env = os.environ.copy()
    if base_version is None:
        env.pop("NVFL_BASE_VERSION", None)
    else:
        env["NVFL_BASE_VERSION"] = base_version

    return subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_versioneer_failure_uses_tracked_base_version(tmp_path):
    before = datetime.date.today()
    result = _run_setup_version(tmp_path)
    after = datetime.date.today()

    assert result.returncode == 0, result.stderr
    expected_versions = {f"2.9.0.dev{date:%y%m%d}" for date in (before, after)}
    assert result.stdout.strip() in expected_versions


def test_versioneer_failure_uses_configured_base_version_override(tmp_path):
    before = datetime.date.today()
    result = _run_setup_version(tmp_path, base_version="3.0.0")
    after = datetime.date.today()

    assert result.returncode == 0, result.stderr
    expected_versions = {f"3.0.0.dev{date:%y%m%d}" for date in (before, after)}
    assert result.stdout.strip() in expected_versions
