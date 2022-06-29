# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import subprocess
import sys
import tempfile

import yaml

from tests.integration_test.utils import read_yaml

from .site_launcher import ServerProperties, SiteLauncher, SiteProperties, kill_process, run_command_in_subprocess

REF_PROJECT_YML = "./data/project.yml"
PROJECT_NAME = "integration_test"
WORKSPACE = "ci_workspace"
PROVISION_SCRIPT = "nvflare.lighter.provision"
PROD_FOLDER_NAME = "prod_00"


def _get_script_dir(site_name: str):
    return os.path.join(WORKSPACE, PROJECT_NAME, PROD_FOLDER_NAME, site_name)


class ProvisionSiteLauncher(SiteLauncher):
    def prepare_workspace(self, n_servers: int, n_clients: int) -> str:
        project_yaml = read_yaml(REF_PROJECT_YML)
        _, temp_yaml = tempfile.mkstemp()
        with open(temp_yaml, "w") as f:
            yaml.dump(project_yaml, f, default_flow_style=False)
        command = f"{sys.executable} -m {PROVISION_SCRIPT} -p {temp_yaml} -w {WORKSPACE}"
        process = run_command_in_subprocess(command)
        process.wait()
        os.remove(temp_yaml)
        return os.path.join(WORKSPACE, PROJECT_NAME, PROD_FOLDER_NAME)

    def start_overseer(self):
        overseer_dir = _get_script_dir("localhost")
        process = run_command_in_subprocess(f"bash {os.path.join(overseer_dir, 'startup', 'start.sh')}")
        print("Starting overseer ...")
        self.overseer_properties = SiteProperties(name="overseer", root_dir=overseer_dir, process=process)

    def stop_overseer(self):
        try:
            # Kill the process
            if self.overseer_properties:
                kill_process(self.overseer_properties)
                subprocess.call(["pkill", "-9", "-f", "gunicorn"])
            else:
                print("No overseer process.")
        except Exception as e:
            print(f"Exception in stopping overseer: {e.__str__()}")
        finally:
            self.overseer_properties = None

    def start_server(self, server_id: int):
        server_name = f"localhost{server_id}"
        server_dir = _get_script_dir(server_name)
        process = run_command_in_subprocess(f"bash {os.path.join(server_dir, 'startup', 'start.sh')}")
        print(f"Starting server {server_name} ...")
        self.server_properties[server_id] = ServerProperties(
            name=server_name, root_dir=server_dir, process=process, port=f"8{server_id}03"
        )

    def start_client(self, client_id):
        client_name = f"site-{client_id}"
        client_dir = _get_script_dir(client_name)
        process = run_command_in_subprocess(f"bash {os.path.join(client_dir, 'startup', 'start.sh')}")
        print(f"Starting client {client_name} ...")
        self.client_properties[client_id] = SiteProperties(name=client_name, root_dir=client_dir, process=process)

    def stop_server(self, server_id):
        server_name = f"localhost{server_id}"
        server_dir = _get_script_dir(server_name)
        run_command_in_subprocess(f"echo 'y' | {os.path.join(server_dir, 'startup', 'stop_fl.sh')}")
        print(f"Stopping server {server_name} ...")
        super().stop_server(server_id)

    def stop_client(self, client_id):
        client_name = f"site-{client_id}"
        client_dir = _get_script_dir(client_name)
        run_command_in_subprocess(f"echo 'y' | {os.path.join(client_dir, 'startup', 'stop_fl.sh')}")
        print(f"Stopping client {client_name} ...")
        super().stop_client(client_id)

    def cleanup(self):
        subprocess.call(["pkill", "-9", "-f", PROD_FOLDER_NAME])
        shutil.rmtree(WORKSPACE)
