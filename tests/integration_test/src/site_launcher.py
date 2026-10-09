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
import socket
import time
from abc import ABC, abstractmethod
from functools import partial

from .utils import find_site_processes, process_group_alive, run_cleanup_steps, stop_site_processes


class SiteProperties:
    def __init__(self, name: str, root_dir: str, process):
        self.name = name
        self.root_dir = root_dir
        self.process = process
        self.processes_stopped = False


class ServerProperties(SiteProperties):
    def __init__(self, name: str, root_dir: str, process, port: str):
        super().__init__(name=name, root_dir=root_dir, process=process)
        self.port = str(port)


def kill_process(site_prop: SiteProperties, graceful_stop=None):
    site_prop.processes_stopped = False
    owned = set()

    def force_stop():
        stop_site_processes(site_prop.process, site_prop.root_dir, known_processes=owned)
        site_prop.process = None
        site_prop.processes_stopped = True
        print(f"Stopped {site_prop.name}.")

    steps = []
    if graceful_stop is not None:
        steps.extend(
            [
                ("Capture jobs before stop script", lambda: owned.update(find_site_processes(site_prop.root_dir))),
                ("Run stop script", graceful_stop),
            ]
        )
    steps.append(("Force site process cleanup", force_stop))
    run_cleanup_steps(steps)


class SiteLauncher(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.server_properties: dict[str, ServerProperties] = {}
        self.client_properties: dict[str, SiteProperties] = {}

    @abstractmethod
    def prepare_workspace(self) -> str:
        pass

    @abstractmethod
    def start_server(self, server_id):
        pass

    @abstractmethod
    def start_client(self, client_id):
        pass

    @abstractmethod
    def start_servers(self):
        pass

    @abstractmethod
    def start_clients(self):
        pass

    def wait_for_server(self, server_id, timeout=60.0):
        """Require the deployed server's admin listener before continuing startup.

        Server deployment precedes admin listener startup. The driver separately
        authenticates its admin session before running any scenario.
        """
        server = self.server_properties[server_id]
        deadline = time.monotonic() + timeout
        last_error = None
        while True:
            if server.process is not None and not process_group_alive(server.process):
                raise RuntimeError(f"Server {server.name} exited before readiness (code={server.process.returncode})")
            try:
                remaining = max(0.0, deadline - time.monotonic())
                with socket.create_connection(("127.0.0.1", int(server.port)), timeout=min(0.5, remaining)):
                    return
            except OSError as error:
                last_error = error
            if time.monotonic() >= deadline:
                log_path = os.path.join(server.root_dir, "log.txt")
                raise RuntimeError(
                    f"Server {server.name} admin port {server.port} did not become ready: "
                    f"{last_error}. Startup log: {log_path}"
                )
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def stop_server(self, server_id):
        if server_id not in self.server_properties:
            raise RuntimeError(f"Server {server_id} not in server_properties.")
        kill_process(self.server_properties[server_id])

    def stop_client(self, client_id):
        if client_id not in self.client_properties:
            raise RuntimeError(f"Client {client_id} not in client_properties.")
        kill_process(self.client_properties[client_id])

    def stop_all_clients(self):
        run_cleanup_steps(
            [(f"Stop client {client_id}", partial(self.stop_client, client_id)) for client_id in self.client_properties]
        )

    def stop_all_servers(self):
        run_cleanup_steps(
            [(f"Stop server {server_id}", partial(self.stop_server, server_id)) for server_id in self.server_properties]
        )

    def stop_all_sites(self):
        run_cleanup_steps([("Stop clients", self.stop_all_clients), ("Stop servers", self.stop_all_servers)])

    def require_sites_stopped(self):
        survivors = [
            site.name
            for site in [*self.client_properties.values(), *self.server_properties.values()]
            if not site.processes_stopped
        ]
        if survivors:
            raise RuntimeError(f"Preserving workspace: process cleanup was not confirmed for {survivors}")

    def get_active_server_id(self, port) -> str:
        active_server_id = None
        for k in self.server_properties.keys():
            if self.server_properties[k].port == str(port):
                active_server_id = k
        return active_server_id

    def cleanup(self):
        self.server_properties.clear()
        self.client_properties.clear()
