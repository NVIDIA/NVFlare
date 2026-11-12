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

"""Shared Recipe selectors are validated at deployment, while help/export/setup remain usable."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_job(tmp_path, source, arguments=(), inherited=None):
    job = tmp_path / "job.py"
    job.write_text(source)
    env = os.environ.copy()
    env.pop("FL_LOG_LEVEL", None)
    env.pop("NVFLARE_SIMULATOR_WORKSPACE_ROOT", None)
    if inherited is not None:
        env["FL_LOG_LEVEL"] = inherited
    root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
    # Scripts mock simulator_run, so only the owned Python interpreter runs.
    # subprocess.run kills and reaps it on timeout; allow for CI startup contention.
    return subprocess.run(
        [sys.executable, str(job), *arguments], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )


_RUNTIME_SCRIPT = (
    "from pathlib import Path\n"
    "from nvflare.recipe import SimEnv\n"
    "from nvflare.job_config.api import FedJob\n"
    "job = FedJob(name='logging-check')\n"
    "def simulator_run(**kwargs):\n"
    "    Path('simulator-started').write_text('called')\n"
    "    return 0\n"
    "job.simulator_run = simulator_run\n"
)


@pytest.mark.parametrize(
    "arguments,inherited,error",
    [
        (["--log_config", "progres"], None, "unknown logging configuration 'progres'"),
        (["--log_config"], None, "--log_config requires a non-empty configuration"),
        (["--log_config="], None, "--log_config requires a non-empty configuration"),
        ([], "progres", "FL_LOG_LEVEL: unknown logging configuration 'progres'"),
    ],
)
def test_invalid_log_config_exits_cleanly_before_runtime_startup(tmp_path, arguments, inherited, error):
    proc = _run_job(
        tmp_path,
        _RUNTIME_SCRIPT + "SimEnv(num_clients=1, workspace_root='workspace').deploy(job)\n",
        arguments,
        inherited,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2, output
    assert "usage:" in proc.stderr and error in proc.stderr
    assert "Traceback" not in output
    assert "Starting server and clients" not in output
    assert not (tmp_path / "simulator-started").exists()
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("source", ["argument", "environment"])
def test_help_ignores_invalid_runtime_logging(tmp_path, source):
    arguments = ["--help"] + (["--log_config", "missing.json"] if source == "argument" else [])
    proc = _run_job(
        tmp_path,
        "from nvflare.recipe import SimEnv\n"
        "import argparse\n"
        "argparse.ArgumentParser(description='JOB HELP').parse_args()\n",
        arguments,
        "missing.json" if source == "environment" else None,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "JOB HELP" in proc.stdout
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("source", ["argument", "environment", "api"])
def test_export_only_ignores_invalid_runtime_logging(tmp_path, source):
    arguments = ["--export", "--export-dir", "exported"]
    if source == "argument":
        arguments += ["--log_config", "missing.json"]
    log_config = "'missing.json'" if source == "api" else "None"
    proc = _run_job(
        tmp_path,
        "from nvflare.recipe import SimEnv\n"
        "from nvflare.recipe.spec import Recipe\n"
        "from nvflare.job_config.api import FedJob\n"
        f"env = SimEnv(num_clients=1, log_config={log_config}, workspace_root='workspace')\n"
        "Recipe(FedJob(name='exported-job')).execute(env)\n",
        arguments,
        "missing.json" if source == "environment" else None,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "exported/exported-job/meta.json").is_file()
    assert not (tmp_path / "workspace").exists()
    assert "Starting server and clients" not in proc.stdout


@pytest.mark.parametrize("source", ["argument", "environment", "api"])
def test_script_can_generate_logging_file_after_imports_and_construction(tmp_path, source):
    log_config = "'generated.json'" if source == "api" else "None"
    proc = _run_job(
        tmp_path,
        _RUNTIME_SCRIPT
        + f"env = SimEnv(num_clients=1, log_config={log_config}, workspace_root='workspace')\n"
        + "Path('generated.json').write_text('{\"version\": 1}')\n"
        + "env.deploy(job)\nprint('READY')\n",
        ["--log_config", "generated.json"] if source == "argument" else [],
        "generated.json" if source == "environment" else None,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "READY" in proc.stdout
    assert (tmp_path / "simulator-started").read_text() == "called"


def test_script_can_repair_logging_environment_after_imports(tmp_path):
    proc = _run_job(
        tmp_path,
        _RUNTIME_SCRIPT
        + "import os\nos.environ['FL_LOG_LEVEL'] = 'full'\n"
        + "SimEnv(num_clients=1, workspace_root='workspace').deploy(job)\nprint('READY')\n",
        inherited="missing.json",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "READY" in proc.stdout
    assert (tmp_path / "simulator-started").read_text() == "called"
