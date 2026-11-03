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

"""Normal Python startup remains compatible with existing job environments."""

import json
import os
import shlex
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessArgs
from nvflare.utils import job_launcher_utils

_GENERATORS = [job_launcher_utils.generate_client_command, job_launcher_utils.generate_server_command]


def _command(generate, module, options=None):
    args = {JobProcessArgs.EXE_MODULE: ("-m", module)}
    if options is not None:
        args[JobProcessArgs.OPTIONS] = options
    context = FLContext()
    context.set_prop(FLContextKey.JOB_PROCESS_ARGS, args, private=True, sticky=False)
    return shlex.split(generate(context))


@pytest.mark.parametrize("generate", _GENERATORS)
def test_configured_module_imports_inherited_dependencies_and_sitecustomize(tmp_path, generate):
    modules = tmp_path / "platform"
    dependencies = tmp_path / "dependencies"
    custom = tmp_path / "job-custom"
    for directory in (modules, dependencies, custom):
        directory.mkdir()
    (dependencies / "external_requirement.py").write_text("VALUE = 'from-inherited-path'\n", encoding="utf-8")
    (custom / "sitecustomize.py").write_text(
        "import os\nos.environ['NVFL_TEST_STARTUP_HOOK'] = 'sitecustomize-ran'\n", encoding="utf-8"
    )
    (modules / "configured_worker.py").write_text(
        "import json, os, sys\n"
        "import external_requirement\n"
        "print(json.dumps({'dependency': external_requirement.VALUE, "
        "'hook': os.environ.get('NVFL_TEST_STARTUP_HOOK'), 'isolated': sys.flags.isolated}))\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("NVFL_TEST_STARTUP_HOOK", None)
    environment["PYTHONPATH"] = os.pathsep.join(map(str, (modules, dependencies, custom)))

    command = _command(generate, "configured_worker")
    assert command == [sys.executable, "-m", "configured_worker"]
    completed = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)

    assert json.loads(completed.stdout) == {
        "dependency": "from-inherited-path",
        "hook": "sitecustomize-ran",
        "isolated": 0,
    }


@pytest.mark.parametrize(
    ("generate", "module"),
    [
        (job_launcher_utils.generate_client_command, "nvflare.private.fed.app.client.worker_process"),
        (job_launcher_utils.generate_server_command, "nvflare.private.fed.app.server.runner_process"),
    ],
)
def test_real_job_process_imports_source_and_dependencies_from_pythonpath(tmp_path, monkeypatch, generate, module):
    # A source checkout can legitimately use dependencies outside the chosen
    # Python installation. No package installation or job execution is needed.
    environment_root = tmp_path / "python-env"
    venv.EnvBuilder(with_pip=False).create(environment_root)
    python = environment_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    repository = Path(__file__).resolve().parents[5]
    dependency_paths = [path for path in sys.path if path and os.path.isdir(path)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(repository), *dependency_paths])
    monkeypatch.setattr(job_launcher_utils.sys, "executable", str(python))

    completed = subprocess.run(
        _command(generate, module, ("--help", "")),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


@pytest.mark.parametrize("generate", _GENERATORS)
def test_normal_startup_honors_user_site_and_usercustomize(tmp_path, generate):
    # Virtual environments can deliberately disable user-site imports. Use the
    # base interpreter to exercise the supported normal-Python installation.
    base_python = getattr(sys, "_base_executable", sys.executable)
    environment = os.environ.copy()
    environment.pop("PYTHONNOUSERSITE", None)
    environment.pop("NVFL_TEST_USER_HOOK", None)
    environment["PYTHONUSERBASE"] = str(tmp_path / "userbase")
    probe = subprocess.run(
        [base_python, "-c", "import json,site; print(json.dumps([site.ENABLE_USER_SITE,site.getusersitepackages()]))"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    enabled, directory = json.loads(probe.stdout)
    if enabled is not True:
        pytest.skip("base Python disables user-site imports in this environment")
    user_site = Path(directory)
    user_site.mkdir(parents=True)
    (user_site / "user_requirement.py").write_text("VALUE = 'from-user-site'\n", encoding="utf-8")
    (user_site / "usercustomize.py").write_text(
        "import os\nos.environ['NVFL_TEST_USER_HOOK'] = 'usercustomize-ran'\n", encoding="utf-8"
    )
    modules = tmp_path / "platform"
    modules.mkdir()
    (modules / "configured_worker.py").write_text(
        "import json,os\nimport user_requirement\n"
        "print(json.dumps([user_requirement.VALUE,os.environ.get('NVFL_TEST_USER_HOOK')]))\n",
        encoding="utf-8",
    )
    environment["PYTHONPATH"] = str(modules)
    command = _command(generate, "configured_worker")
    command[0] = base_python

    completed = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)

    assert json.loads(completed.stdout) == ["from-user-site", "usercustomize-ran"]
