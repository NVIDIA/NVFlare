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
import inspect
import json
import shutil
from typing import Dict
import os

from nvflare.app_common.job.client_app import ClientApp
from nvflare.app_common.job.server_app import ServerApp

CONFIG = "config"
FED_SERVER_JSON = "config_fed_server.json"
FED_CLIENT_JSON = "config_fed_client.json"


class FedApp:
    def __init__(self, server_app: ServerApp, client_app: ClientApp) -> None:
        super().__init__()

        if not isinstance(server_app, ServerApp):
            raise ValueError(f"server_app must be ServerApp, but got {server_app.__class__}")
        if not isinstance(client_app, ClientApp):
            raise ValueError(f"server_app must be ClientApp, but got {client_app.__class__}")

        self.server_app: ServerApp = server_app
        self.client_app: ClientApp = client_app


class FedJob:
    def __init__(self, name, min_clients, mandatory_clients) -> None:
        super().__init__()

        self.min_clients = min_clients
        self.mandatory_clients = mandatory_clients

        self.job_name = name
        self.fed_apps: Dict[str, FedApp] = {}
        self.deploy_map = {}

    def add_fed_app(self, app_name: str, fed_app: FedApp):
        if not isinstance(fed_app, FedApp):
            raise RuntimeError(f"server_app must be FedApp, but got {fed_app.__class__}")

        self.fed_apps[app_name] = fed_app

    def set_site_app(self, site_name: str, app_name: str):
        if not app_name in self.fed_apps.keys():
            raise RuntimeError(f"fed_app {app_name} does not exist.")

        self.deploy_map[site_name] = app_name

    def generate_meta(self):
        """ generate the job meta.json

        Returns:

        """
        pass

    def generate_job_config(self, job_root):
        """ generate the job config

        Returns:

        """
        if os.path.exists(job_root):
            shutil.rmtree(job_root, ignore_errors=True)

        for app_name, fed_app in self.fed_apps.items():
            config_dir = os.path.join(job_root, self.job_name, app_name, CONFIG)
            os.makedirs(config_dir, exist_ok=True)

            if fed_app.server_app:
                server_app = {"format_version": 2}
                server_app["components"] = []
                for cid, component in fed_app.server_app.components.items():
                    server_app["components"].append(
                        {
                            "id": cid,
                            "path": component.__module__ + "." + component.__class__.__name__,
                            "args": self._get_args(component)
                        }
                    )
                server_config = os.path.join(config_dir, FED_SERVER_JSON)
                with open(server_config, "w") as outfile:
                    json_dump = json.dumps(server_app, indent=4)
                    outfile.write(json_dump)

            # if fed_app.client_app:
            #     client_config = os.path.join(job_root, self.job_name, app_name, FED_CLIENT_JSON)
            #     os.makedirs(client_config, exist_ok=True)

    def _get_args(self, component):
        constructor = component.__class__.__init__
        parameters = inspect.signature(constructor).parameters
        attrs = component.__dict__
        args = {}
        for param in parameters:
            if param in attrs.keys() and parameters[param].default != attrs[param]:
                args[param] = attrs[param]

        return args

