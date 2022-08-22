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
import sys
import tempfile

from .site_launcher import ServerProperties, SiteLauncher, SiteProperties, run_command_in_subprocess

SERVER_SCRIPT = "nvflare.private.fed.app.server.server_train"
CLIENT_SCRIPT = "nvflare.private.fed.app.client.client_train"
SERVER_NVF_CONFIG = "fed_server.json"
CLIENT_NVF_CONFIG = "fed_client.json"

ADMIN_NAME = "admin"


def _get_client_name(client_id: int):
    return f"site-{client_id}"


def _get_server_name(server_id: int):
    return f"server_{server_id}"


class POCSiteLauncher(SiteLauncher):
    def __init__(self, poc_dir: str):
        """Launches and keeps track of servers and clients."""
        super().__init__()

        self.poc_dir = tempfile.mkdtemp()
        if os.path.exists(self.poc_dir):
            shutil.rmtree(self.poc_dir)
        shutil.copytree(poc_dir, self.poc_dir)
        print(f"Using POC at dir: {self.poc_dir}")

    def start_overseer(self):
        raise RuntimeError("POC mode does not have overseer.")

    def stop_overseer(self):
        pass

    def prepare_workspace(self, n_servers: int, n_clients: int) -> str:
        # prepare server
        src_server_dir = os.path.join(self.poc_dir, "server")

        # Create upload directory
        os.makedirs(os.path.join(src_server_dir, "transfer"), exist_ok=True)

        for i in range(n_servers):
            server_id = i
            server_name = _get_server_name(server_id)

            # Copy and create new directory
            server_dir_name = os.path.join(self.poc_dir, server_name)
            shutil.copytree(src_server_dir, server_dir_name)

            # replace SERVER_NVF_CONFIG ports
            fed_server_path = os.path.join(server_dir_name, "startup", SERVER_NVF_CONFIG)
            with open(fed_server_path, "r") as f:
                fed_server_json = f.read()

            fed_server_json = fed_server_json.replace("8002", f"8{i}02").replace("8003", f"8{i}03")

            with open(fed_server_path, "w") as f:
                f.write(fed_server_json)

        # prepare client
        src_client_directory = os.path.join(self.poc_dir, "client")
        for i in range(1, n_clients + 1):
            client_id = i
            client_name = _get_client_name(client_id)

            # Copy and create new directory
            client_dir_name = os.path.join(self.poc_dir, client_name)
            shutil.copytree(src_client_directory, client_dir_name)
        return self.poc_dir

    def start_server(self, server_id: int):
        server_name = _get_server_name(server_id)
        server_dir_name = os.path.join(self.poc_dir, server_name)

        command = (
            f"{sys.executable} -m {SERVER_SCRIPT}"
            f" -m {server_dir_name} -s {SERVER_NVF_CONFIG}"
            " --set secure_train=false config_folder=config"
        )
        process = run_command_in_subprocess(command)

        self.server_properties[server_id] = ServerProperties(
            name=server_name, root_dir=server_dir_name, process=process, port=f"8{server_id}03"
        )
        print(f"Launched server ({server_id}) using {command}. process_id: {process.pid}")

    def start_client(self, client_id: int):
        client_name = _get_client_name(client_id)
        client_dir_name = os.path.join(self.poc_dir, client_name)

        # Launch the new client
        command = (
            f"{sys.executable} -m {CLIENT_SCRIPT}"
            f" -m {client_dir_name} -s {CLIENT_NVF_CONFIG}"
            f" --set secure_train=false config_folder=config uid={client_name}"
        )
        process = run_command_in_subprocess(command)

        self.client_properties[client_id] = SiteProperties(name=client_name, root_dir=client_dir_name, process=process)
        print(f"Launched client {client_name} process using {command}. process_id: {process.pid}")

    def cleanup(self):
        print(f"Deleting temporary directory: {self.poc_dir}.")
        shutil.rmtree(self.poc_dir)
