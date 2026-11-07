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

"""Shared Recipe logging arguments fail as usage errors before job execution."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "arguments,inherited,error",
    [
        (["--log_config", "progres"], None, "unknown logging configuration 'progres'"),
        (["--log_config"], None, "--log_config requires a non-empty configuration"),
        (["--log_config="], None, "--log_config requires a non-empty configuration"),
        ([], "progres", "FL_LOG_LEVEL: unknown logging configuration 'progres'"),
    ],
)
def test_invalid_log_config_exits_cleanly_before_job_code(tmp_path, arguments, inherited, error):
    job = tmp_path / "job.py"
    job.write_text(
        "from pathlib import Path\n"
        "from nvflare.recipe import SimEnv\n"
        "Path('job-started').write_text('unexpected')\n"
        "SimEnv(num_clients=1)\n"
    )
    env = os.environ.copy()
    env.pop("FL_LOG_LEVEL", None)
    if inherited is not None:
        env["FL_LOG_LEVEL"] = inherited
    root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
    # This interpreter never starts the simulator. subprocess.run kills and reaps
    # its owned child on timeout; the bound permits interpreter startup on CI.
    proc = subprocess.run(
        [sys.executable, str(job), *arguments], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2, output
    assert "usage:" in proc.stderr and error in proc.stderr
    assert "Traceback" not in output
    assert "Starting server and clients" not in output
    assert not (tmp_path / "job-started").exists()
