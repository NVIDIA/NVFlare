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

import logging
import os
import signal
from abc import ABC, abstractmethod
from typing import Dict, Optional

from .utils import run_command_in_subprocess


class SiteProperties:
    def __init__(self, name: str, root_dir: str, process):
        self.name = name
        self.root_dir = root_dir
        self.process = process


class ServerProperties(SiteProperties):
    def __init__(self, name: str, root_dir: str, process, port: str):
        super().__init__(name=name, root_dir=root_dir, process=process)
        self.port = str(port)


def kill_process(site_prop: SiteProperties):
    if not site_prop.process:
        return
    os.killpg(site_prop.process.pid, signal.SIGTERM)
    p = run_command_in_subprocess(f"kill -9 {str(site_prop.process.pid)}")
    p.wait()
    p = run_command_in_subprocess(f"pkill -9 -f {site_prop.root_dir}")
    p.wait()
    print(f"Kill {site_prop.name}.")
    site_prop.process.wait()


class SiteLauncher(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.overseer_properties: Optional[SiteProperties] = None
        self.server_properties: Dict[str, ServerProperties] = {}
        self.client_properties: Dict[str, SiteProperties] = {}

    @abstractmethod
    def prepare_workspace(self) -> str:
        pass

    @abstractmethod
    def start_overseer(self):
        pass

    @abstractmethod
    def start_server(self, server_id):
        pass

    @abstractmethod
    def start_client(self, client_id):
        pass

    @abstractmethod
    def stop_overseer(self):
        pass

    @abstractmethod
    def start_servers(self):
        pass

    @abstractmethod
    def start_clients(self):
        pass

    def stop_server(self, server_id):
        if server_id not in self.server_properties:
            raise RuntimeError(f"Server {server_id} not in server_properties.")
        server_prop: ServerProperties = self.server_properties[server_id]
        try:
            # Kill the process
            kill_process(server_prop)
        except Exception as e:
            print(f"Exception in stopping server {server_id}: {e.__str__()}")

    def stop_client(self, client_id):
        if client_id not in self.client_properties:
            raise RuntimeError(f"Client {client_id} not in client_properties.")
        client_prop: SiteProperties = self.client_properties[client_id]

        try:
            kill_process(client_prop)
        except Exception as e:
            print(f"Exception in stopping client {client_id}: {e.__str__()}")

    def stop_all_clients(self):
        for client_id in list(self.client_properties.keys()):
            self.stop_client(client_id)

    def stop_all_servers(self):
        for server_id in list(self.server_properties.keys()):
            self.stop_server(server_id)

    def stop_all_sites(self):
        self.stop_all_clients()
        self.stop_all_servers()
        self.stop_overseer()

    def get_active_server_id(self, port) -> str:
        active_server_id = None
        for k in self.server_properties.keys():
            if self.server_properties[k].port == str(port):
                active_server_id = k
        return active_server_id

    def cleanup(self):
        self.overseer_properties = None
        self.server_properties.clear()
        self.client_properties.clear()
