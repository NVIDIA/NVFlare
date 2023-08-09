# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile
import time

import yaml

from .site_launcher import ServerProperties, SiteLauncher, SiteProperties, kill_process
from .utils import (
    cleanup_job_and_snapshot,
    read_yaml,
    run_command_in_subprocess,
    run_provision_command,
    update_job_store_path_in_workspace,
    update_snapshot_path_in_workspace,
)

WORKSPACE = "ci_workspace"
PROD_FOLDER_NAME = "prod_00"


def _start_site(site_properties: SiteProperties):
    process = run_command_in_subprocess(f"bash {os.path.join(site_properties.root_dir, 'startup', 'start.sh')}")
    print(f"Starting {site_properties.name} ...")
    site_properties.process = process


def _stop_site(site_properties: SiteProperties):
    run_command_in_subprocess(f"echo 'y' | {os.path.join(site_properties.root_dir, 'startup', 'stop_fl.sh')}")
    print(f"Stopping {site_properties.name} ...")


class ProvisionSiteLauncher(SiteLauncher):
    def __init__(self, project_yaml: str):
        super().__init__()
        self.admin_user_names = []
        self.project_yaml = read_yaml(project_yaml)
        for p in self.project_yaml["participants"]:
            name = p["name"]
            script_dir = os.path.join(self._get_workspace_dir(), name)
            if p["type"] == "server":
                admin_port = p["admin_port"]
                self.server_properties[name] = ServerProperties(name, script_dir, None, admin_port)
            elif p["type"] == "client":
                self.client_properties[name] = SiteProperties(name, script_dir, None)
            elif p["type"] == "overseer":
                self.overseer_properties = SiteProperties(name, script_dir, None)
            elif p["type"] == "admin":
                self.admin_user_names.append(name)

    def _get_workspace_dir(self):
        return os.path.join(WORKSPACE, self.project_yaml["name"], PROD_FOLDER_NAME)

    def prepare_workspace(self) -> str:
        _, temp_yaml = tempfile.mkstemp()
        with open(temp_yaml, "w") as f:
            yaml.dump(self.project_yaml, f, default_flow_style=False)
        run_provision_command(project_yaml=temp_yaml, workspace=WORKSPACE)
        os.remove(temp_yaml)
        new_job_store = None
        new_snapshot_store = None
        for k in self.server_properties:
            server_name = self.server_properties[k].name
            new_job_store = update_job_store_path_in_workspace(self._get_workspace_dir(), server_name, new_job_store)
            new_snapshot_store = update_snapshot_path_in_workspace(
                self._get_workspace_dir(), server_name, new_snapshot_store
            )
            cleanup_job_and_snapshot(self._get_workspace_dir(), server_name)
        return os.path.join(WORKSPACE, self.project_yaml["name"], PROD_FOLDER_NAME)

    def start_overseer(self):
        _start_site(self.overseer_properties)

    def stop_overseer(self):
        try:
            # Kill the process
            if self.overseer_properties:
                kill_process(self.overseer_properties)
                process = run_command_in_subprocess("pkill -9 -f gunicorn")
                process.wait()
            else:
                print("No overseer process to stop.")
        except Exception as e:
            print(f"Exception in stopping overseer: {e.__str__()}")
        finally:
            self.overseer_properties = None

    def start_servers(self):
        for k in self.server_properties:
            self.start_server(k)
            time.sleep(3.0)  # makes the first one always primary

    def start_clients(self):
        for k in self.client_properties:
            self.start_client(k)

    def start_server(self, server_id: str):
        _start_site(self.server_properties[server_id])

    def stop_server(self, server_id: str):
        _stop_site(self.server_properties[server_id])
        super().stop_server(server_id)

    def start_client(self, client_id: str):
        _start_site(self.client_properties[client_id])

    def stop_client(self, client_id: str):
        _stop_site(self.client_properties[client_id])
        super().stop_client(client_id)

    def cleanup(self):
        process = run_command_in_subprocess(f"pkill -9 -f {PROD_FOLDER_NAME}")
        process.wait()
        for server_name in self.server_properties:
            cleanup_job_and_snapshot(self._get_workspace_dir(), server_name)
        shutil.rmtree(WORKSPACE)
        super().cleanup()
