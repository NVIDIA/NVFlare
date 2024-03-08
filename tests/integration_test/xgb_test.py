# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import shlex
import subprocess
import sys

from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port


def run_command_in_subprocess(script_path: str, command_args: str):
    new_env = os.environ.copy()
    python_path = ":".join(sys.path)[1:]  # strip leading colon
    new_env["PYTHONPATH"] = f"{python_path}:{os.path.dirname(script_path)}"
    process = subprocess.Popen(
        shlex.split(f"python3 {script_path} {command_args}"),
        env=new_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


class TestXGB:
    def test_xgb_mock_server_and_client(self):
        num_clients = 2
        port = get_open_tcp_port({})
        server_addr = f"localhost:{port}"
        integration_test_dir = os.path.dirname(os.path.abspath(__file__))
        server_script_path = os.path.join(integration_test_dir, "src", "mock_xgb", "run_server.py")

        server_process = run_command_in_subprocess(
            script_path=server_script_path, command_args=f"-a {server_addr} -c {num_clients} -w 10"
        )
        assert server_process is not None

        client_processes = {}
        client_script_path = os.path.join(integration_test_dir, "src", "mock_xgb", "run_client.py")
        for i in range(num_clients):
            client_process = run_command_in_subprocess(
                script_path=client_script_path, command_args=f"-a {server_addr} -r {i} -n {10}"
            )
            client_processes[i] = client_process
            assert client_process is not None

        for i in range(num_clients):
            stdout, stderr = client_processes[i].communicate()
            assert "ERROR" not in stdout.decode("utf-8")
            assert stderr.decode("utf-8") == ""
            client_processes[i].terminate()
        server_process.terminate()
        stdout, stderr = server_process.communicate()
        assert "ERROR" not in stdout.decode("utf-8")
        assert "ERROR" not in stderr.decode("utf-8")
        print(f"Server Process Output (stdout):\n{stdout.decode('utf-8')}")
        print(f"Server Process Output (stderr):\n{stderr.decode('utf-8')}")
