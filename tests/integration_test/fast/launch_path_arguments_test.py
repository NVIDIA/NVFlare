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


"""Real local service/job launch contracts for shell-sensitive paths."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nvflare.tool.poc import poc_commands
from nvflare.tool.poc.poc_commands import SC
from tests.process_test_utils import OwnedProcesses, run_owned_process
from tests.timing_utils import wait_for


@pytest.mark.parametrize("launcher_name", ["ClientProcessJobLauncher", "ServerProcessJobLauncher"])
@pytest.mark.parametrize(
    "interpreter_dir, workspace_name",
    [("venv space", "workspace"), ("venv#hash", "workspace"), ("venv", "workspace space"), ("venv", "workspace#hash")],
)
def test_job_launch_preserves_real_argv(tmp_path, launcher_name, interpreter_dir, workspace_name):
    interpreter = tmp_path / interpreter_dir / "python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    workspace = tmp_path / workspace_name
    workspace.mkdir()
    output = tmp_path / "argv.json"
    (tmp_path / "launch_probe.py").write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['ARGV_OUTPUT']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    driver = r"""
import shlex, sys
from types import SimpleNamespace
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessArgs
from nvflare.app_common.job_launcher.client_process_launcher import ClientProcessJobLauncher
from nvflare.app_common.job_launcher.server_process_launcher import ServerProcessJobLauncher
workspace, interpreter, launcher_name = sys.argv[1:]
ctx = FLContext()
ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, SimpleNamespace(
    get_root_dir=lambda: workspace, get_app_custom_dir=lambda _: '', get_app_dir=lambda _: workspace))
ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, {
    JobProcessArgs.EXE_MODULE: ('-m', 'launch_probe'),
    JobProcessArgs.WORKSPACE: ('-m', workspace),
    JobProcessArgs.OPTIONS: ('--set', shlex.join(["data=/train set#1", "label=Alice's model", r"path=C:\data", "empty="])),
})
sys.executable = interpreter
handle = globals()[launcher_name]().launch_job({JobConstants.JOB_ID: 'job-1'}, ctx)
handle.wait()
"""
    env = dict(os.environ, ARGV_OUTPUT=str(output))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", "")]
    )
    result = run_owned_process(
        [sys.executable, "-c", driver, str(workspace), str(interpreter), launcher_name],
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text()) == [
        "-m",
        str(workspace),
        "--set",
        "data=/train set#1",
        "label=Alice's model",
        r"path=C:\data",
        "empty=",
    ]


@pytest.mark.parametrize("path_name", ["poc space", "poc#hash"])
@pytest.mark.parametrize("service_type", ["client", "server", "admin"])
def test_poc_launch_runs_generated_local_scripts(tmp_path, monkeypatch, path_name, service_type):
    templates = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "nvflare/lighter/templates/master_template.yml").read_text()
    )
    prod_dir = tmp_path / path_name
    service = "admin" if service_type == "admin" else service_type
    workspace = prod_dir / service
    startup = workspace / "startup"
    startup.mkdir(parents=True)
    output = workspace / "argv.json"
    # Exercise real Bash startup and argument forwarding without opening federation ports.
    # The Python stand-in records the received argv and remains live until POC stop.
    fake_bin = tmp_path / "python bin#1"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env python\n"
        "import json, os, pathlib, sys, time\n"
        "output = pathlib.Path(os.environ['ARGV_OUTPUT'])\n"
        "pending = output.with_suffix('.tmp')\n"
        "pending.write_text(json.dumps(sys.argv[1:]))\n"
        "pending.replace(output)\n"
        "if os.environ['SERVICE_TYPE'] != 'admin':\n"
        "    while not pathlib.Path(os.environ['STOP_FILE']).exists(): time.sleep(0.05)\n"
    )
    fake_python.chmod(0o755)
    monkeypatch.setenv("ARGV_OUTPUT", str(output))
    monkeypatch.setenv("SERVICE_TYPE", service_type)
    monkeypatch.setenv("STOP_FILE", str(workspace / "shutdown.fl"))
    monkeypatch.delenv("NVFL_WORKSPACE", raising=False)
    # POC chooses its interpreter from sys.executable's directory.
    real_python = sys.executable
    interpreter = fake_bin / "python"
    interpreter.symlink_to(real_python)
    monkeypatch.setattr(sys, "executable", str(interpreter))
    config = {SC.FLARE_PROJ_ADMIN: "admin", SC.FLARE_OTHER_ADMINS: []}
    if service_type == "admin":
        script_name, script = "fl_admin.sh", templates["fl_admin_sh"]
    else:
        key = "start_cln_sh" if service_type == "client" else "start_svr_sh"
        script_name, script = "start.sh", templates[key]
        sub_start = templates["sub_start_sh"]
        for key, value in {
            "type": service_type,
            "app_name": "client_train" if service_type == "client" else "server_train",
            "cln_uid": "uid=site-1" if service_type == "client" else "",
            "org_name": "test",
            "config_folder": "config",
        }.items():
            sub_start = sub_start.replace("{~~" + key + "~~}", value)
        (startup / "sub_start.sh").write_text(sub_start)
        (startup / "sub_start.sh").chmod(0o755)
    (startup / script_name).write_text(script)
    (startup / script_name).chmod(0o755)
    study = "study space#hash'quote"
    command = poc_commands.get_service_command(SC.CMD_START, str(prod_dir), service, config, study=study)
    # Register ownership before any launch, including the detached service supervisor.
    with OwnedProcesses(process_factory=subprocess.Popen) as owner:
        if service_type == "admin":
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", "")])
            result = run_owned_process(
                [
                    real_python,
                    "-c",
                    "import sys; from nvflare.tool.poc.poc_commands import sync_process; "
                    "sys.executable = sys.argv[1]; sync_process('admin', sys.argv[2])",
                    str(interpreter),
                    command,
                ],
                env=env,
                timeout=30,
            )
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            monkeypatch.setattr(poc_commands.subprocess, "Popen", owner.start)
            poc_commands.async_process(service, command, [], config)
        wait_for(output.exists, timeout=30, message=f"No worker argv recorded in {workspace}")
        argv = json.loads(output.read_text())
        module_index = argv.index("-m")
        assert argv[module_index + 2] == "-m"
        expected_workspace = str(startup / "..") if service_type == "admin" else str(workspace)
        assert argv[module_index + 3] == expected_workspace
        if service_type == "admin":
            assert argv[-2:] == ["--study", study]
            assert (workspace / "transfer").is_dir()
        else:
            stop_command = poc_commands.get_service_command(SC.CMD_STOP, str(prod_dir), service, config)
            assert shlex.split(stop_command, comments=False) == ["touch", str(workspace / "shutdown.fl")]
            poc_commands._run_stop_command(service, stop_command, timeout=10)
            wait_for(lambda: not (workspace / "daemon_pid.fl").exists(), timeout=30)
