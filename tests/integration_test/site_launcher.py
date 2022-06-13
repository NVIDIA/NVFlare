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

import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Dict, Optional


class POCDirectory:
    def __init__(
        self,
        poc_dir,
        ha=False,
        server_dir_name="server",
        client_dir_name="client",
        admin_dir_name="admin",
        overseer_dir_name="overseer",
    ):
        if not os.path.isdir(poc_dir):
            raise ValueError(
                f"poc_dir {poc_dir} is not a directory. " "Please run POC command first and provide the POC path!"
            )
        for f in [server_dir_name, client_dir_name, admin_dir_name]:
            if not os.path.isdir(os.path.join(poc_dir, f)):
                raise ValueError(f"{f} is not a directory inside poc.")
        self.poc_dir = poc_dir
        self.ha = ha
        self.server_dir_name = server_dir_name
        self.client_dir_name = client_dir_name
        self.admin_dir_name = admin_dir_name
        self.overseer_dir_name = overseer_dir_name

    def copy_to(self, dst):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(self.poc_dir, dst)
        if self.ha:
            for name in ("admin", "server", "client"):
                d = os.path.join(dst, f"{name}", "startup")
                os.rename(os.path.join(d, f"fed_{name}_HA.json"), os.path.join(d, f"fed_{name}.json"))
        return POCDirectory(
            poc_dir=dst,
            server_dir_name=self.server_dir_name,
            client_dir_name=self.client_dir_name,
            admin_dir_name=self.admin_dir_name,
            overseer_dir_name=self.overseer_dir_name,
        )

    def overseer_dir(self):
        return os.path.join(self.poc_dir, self.overseer_dir_name)

    def server_dir(self):
        return os.path.join(self.poc_dir, self.server_dir_name)


class SiteProperties:
    def __init__(self, name: str, root_dir: str, process):
        if process is None:
            raise ValueError("process can't be None in SiteProperties.")
        self.name = name
        self.root_dir = root_dir
        self.process = process


class ServerProperties(SiteProperties):
    def __init__(self, name: str, root_dir: str, process, port: str):
        super().__init__(name=name, root_dir=root_dir, process=process)
        self.port = port


def _kill_process(site_prop: SiteProperties):
    os.killpg(site_prop.process.pid, signal.SIGTERM)
    subprocess.call(["kill", str(site_prop.process.pid)])
    subprocess.call(["pkill", "-9", "-f", site_prop.name])
    print(f"Kill {site_prop.name}.")
    site_prop.process.communicate()


class SiteLauncher:
    def __init__(
        self,
        poc_directory: POCDirectory,
    ):
        """
        This class sets up the test environment for a test. It will launch and keep track of servers and clients.
        """
        super().__init__()

        self.original_poc_directory = poc_directory

        self.overseer_properties: Optional[SiteProperties] = None
        self.server_properties: Dict[int, ServerProperties] = {}
        self.client_properties: Dict[int, SiteProperties] = {}

        self.logger = logging.getLogger(self.__class__.__name__)

        root_dir = tempfile.mkdtemp()
        self.poc_directory = self.original_poc_directory.copy_to(root_dir)
        print(f"Using POC at dir: {self.poc_directory.poc_dir}")

    def start_overseer(self):
        overseer_dir = self.poc_directory.overseer_dir()
        new_env = os.environ.copy()
        command = f"{sys.executable} -m nvflare.ha.overseer.overseer"

        process = subprocess.Popen(
            shlex.split(command),
            preexec_fn=os.setsid,
            env=new_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Starting overseer ...")

        self.overseer_properties = SiteProperties(name="overseer", root_dir=overseer_dir, process=process)

    def start_servers(self, n=1):
        src_server_dir = self.poc_directory.server_dir()

        # Create upload directory
        os.makedirs(os.path.join(src_server_dir, "transfer"), exist_ok=True)

        for i in range(n):
            server_id = i
            server_name = self.poc_directory.server_dir_name + f"_{server_id}"

            # Copy and create new directory
            server_dir_name = os.path.join(self.poc_directory.poc_dir, server_name)
            shutil.copytree(src_server_dir, server_dir_name)

            # replace fed_server.json ports
            fed_server_path = os.path.join(server_dir_name, "startup", "fed_server.json")
            with open(fed_server_path, "r") as f:
                fed_server_json = f.read()

            fed_server_json = fed_server_json.replace("8002", f"8{i}02").replace("8003", f"8{i}03")

            with open(fed_server_path, "w") as f:
                f.write(fed_server_json)

            self.start_server(server_id)
            time.sleep(5)

    def start_server(self, server_id: int):
        server_name = self.poc_directory.server_dir_name + f"_{server_id}"
        server_dir_name = os.path.join(self.poc_directory.poc_dir, server_name)

        command = (
            f"{sys.executable} -m nvflare.private.fed.app.server.server_train"
            f" -m {server_dir_name} -s fed_server.json"
            " --set secure_train=false config_folder=config"
        )
        new_env = os.environ.copy()
        process = subprocess.Popen(
            shlex.split(command),
            preexec_fn=os.setsid,
            env=new_env,
        )

        self.server_properties[server_id] = ServerProperties(
            name=server_name, root_dir=server_dir_name, process=process, port=f"8{server_id}03"
        )
        print(f"Launched server ({server_id}) using {command}. process_id: {process.pid}")

    def start_clients(self, n=2):
        # Make sure that previous clients are killed
        self.stop_all_clients()
        self.client_properties.clear()

        # For each client, copy the directory
        src_client_directory = os.path.join(self.poc_directory.poc_dir, self.poc_directory.client_dir_name)

        for i in range(1, n + 1):
            client_id = i
            client_name = f"site-{client_id}"

            # Copy and create new directory
            client_dir_name = os.path.join(self.poc_directory.poc_dir, client_name)
            shutil.copytree(src_client_directory, client_dir_name)

            self.start_client(client_id)

    def start_client(self, client_id: int):
        client_name = f"site-{client_id}"
        client_dir_name = os.path.join(self.poc_directory.poc_dir, client_name)
        new_env = os.environ.copy()

        # Launch the new client
        client_startup_dir = os.path.join(client_dir_name)
        command = (
            f"{sys.executable} -m nvflare.private.fed.app.client.client_train"
            f" -m {client_startup_dir} -s fed_client.json --set secure_train=false config_folder=config"
            f" uid={client_name}"
        )

        process = subprocess.Popen(
            shlex.split(command),
            preexec_fn=os.setsid,
            env=new_env,
        )

        self.client_properties[client_id] = SiteProperties(name=client_name, root_dir=client_dir_name, process=process)
        print(f"Launched client {client_id} process using {command}. process_id: {process.pid}")

    def get_active_server_id(self, port):
        active_server_id = -1
        for i in range(len(self.server_properties)):
            if self.server_properties.get(i) and self.server_properties[i].port == str(port):
                active_server_id = i
        return active_server_id

    def get_server_prop(self, server_id: int) -> ServerProperties:
        server_prop = self.server_properties.get(server_id)
        if not server_prop:
            raise RuntimeError(f"Missing server properties for server: {server_id}")

        return server_prop

    def stop_overseer(self):
        try:
            # Kill the process
            if self.overseer_properties:
                _kill_process(self.overseer_properties)
            else:
                print("No overseer process.")
        except Exception as e:
            print(f"Exception in stopping overseer: {e.__str__()}")
        finally:
            self.overseer_properties = None

    def stop_server(self, server_id):
        if server_id not in self.server_properties:
            raise RuntimeError(f"Server {server_id} not in server_properties.")
        server_prop: ServerProperties = self.server_properties[server_id]
        try:
            # Kill the process
            _kill_process(server_prop)
        except Exception as e:
            print(f"Exception in stopping server {server_id}: {e.__str__()}")

    def stop_client(self, client_id) -> bool:
        if client_id not in self.client_properties:
            raise RuntimeError(f"Client {client_id} not in client_properties.")
        client_prop: SiteProperties = self.client_properties[client_id]
        if not client_prop.process:
            print(f"Client {client_id} process is None.")
            self.client_properties.pop(client_id)
            return False

        try:
            _kill_process(client_prop)
        except Exception as e:
            print(f"Exception in stopping client {client_id}: {e.__str__()}")
            return False

        return True

    def stop_all_clients(self):
        for client_id in self.client_properties:
            self.stop_client(client_id)
        self.client_properties.clear()

    def stop_all_servers(self):
        for server_id in self.server_properties:
            self.stop_server(server_id)
        self.server_properties.clear()

    def stop_all_sites(self):
        self.stop_all_clients()
        self.stop_all_servers()
        self.stop_overseer()

    def cleanup(self):
        print(f"Deleting temporary directory: {self.poc_directory.poc_dir}.")
        shutil.rmtree(self.poc_directory.poc_dir)
