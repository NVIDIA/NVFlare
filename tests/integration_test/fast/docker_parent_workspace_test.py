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

"""Generated Docker CP startup preserves ordinary and native workspace views."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from nvflare.tool.deploy.docker_deploy import _write_docker_start_script

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Docker startup requires Bash and Unix sockets")


def test_docker_parent_keeps_job_workspace_and_mounts_native_task_view(tmp_path):
    workspace = tmp_path / "prepared site"
    (workspace / "startup").mkdir(parents=True)
    script = _write_docker_start_script(
        SimpleNamespace(kit_dir=workspace, name="site-1", role="client", org="nvidia"),
        docker_image="site/parent:tested",
        network="site-network",
    )
    commands = tmp_path / "commands"
    commands.mkdir()
    record = tmp_path / "docker-args.json"
    docker = commands / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "if 'run' in sys.argv[1:]:\n"
        "    with open(os.environ['DOCKER_ARGS_RECORD'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    docker.chmod(0o700)
    # pytest's full temporary path exceeds macOS's Unix-socket path limit.
    with TemporaryDirectory(prefix="nvfl-docker-", dir="/tmp") as socket_dir, socket.socket(socket.AF_UNIX) as sock:
        endpoint = Path(socket_dir) / "docker.sock"
        env = {
            **os.environ,
            "PATH": str(commands) + os.pathsep + os.environ["PATH"],
            "DOCKER_HOST": "unix://" + str(endpoint),
            "NVFL_DOCKER_SOCK": str(endpoint),
            "NVFL_DOCKER_SOCK_GID": str(os.getgid()),
            "DOCKER_ARGS_RECORD": str(record),
        }
        sock.bind(str(endpoint))
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(record.read_text())
    assert f"{workspace}:/var/tmp/nvflare/workspace" in args
    assert f"type=bind,src={workspace},dst={workspace}" in args
    module_index = args.index("nvflare.private.fed.app.client.client_train")
    assert args[args.index("-m", module_index + 1) + 1] == "/var/tmp/nvflare/workspace"
