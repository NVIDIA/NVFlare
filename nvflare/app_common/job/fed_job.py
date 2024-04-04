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

from nvflare import SimulatorRunner
from nvflare.app_common.job.fed_app import FedApp
from nvflare.private.fed.app.fl_conf import FL_PACKAGES

CONFIG = "config"
CUSTOM = "custom"
FED_SERVER_JSON = "config_fed_server.json"
FED_CLIENT_JSON = "config_fed_client.json"
META_JSON = "meta.json"


class FedJob:
    def __init__(self, job_name, min_clients, mandatory_clients=None) -> None:
        super().__init__()

        self.job_name = job_name
        self.min_clients = min_clients
        self.mandatory_clients = mandatory_clients

        self.fed_apps: Dict[str, FedApp] = {}
        self.deploy_map = {}
        self.resource_specs = {}

        self.custom_modules = []

    def add_fed_app(self, app_name: str, fed_app: FedApp):
        if not isinstance(fed_app, FedApp):
            raise RuntimeError(f"server_app must be type of FedApp, but got {fed_app.__class__}")

        self.fed_apps[app_name] = fed_app

    def set_site_app(self, site_name: str, app_name: str):
        if app_name not in self.fed_apps.keys():
            raise RuntimeError(f"fed_app {app_name} does not exist.")

        self.deploy_map[site_name] = app_name

    def add_resource_spec(self, site_name: str, resource_spec: Dict):
        if site_name in self.resource_specs.keys():
            raise RuntimeError(f"{site_name} resource specs already exist.")
        if not isinstance(resource_spec, dict):
            raise RuntimeError(f"resource_spec must be a dict. But got: {resource_spec.__class__}")

        self.resource_specs[site_name] = resource_spec

    def _generate_meta(self, job_root):
        """ generate the job meta.json

        Returns:

        """
        meta_file = os.path.join(job_root, self.job_name, META_JSON)
        meta_json = {
            "name": self.job_name,
            "resource_spec": self.resource_specs,
            "min_clients": self.min_clients,
            "deploy_map": self._get_deploy_map()
        }
        if self.mandatory_clients:
            meta_json["mandatory_clients"] = self.mandatory_clients

        with open(meta_file, "w") as outfile:
            json_dump = json.dumps(meta_json, indent=4)
            outfile.write(json_dump)

    def generate_job_config(self, job_root):
        """ generate the job config

        Returns:

        """
        if os.path.exists(job_root):
            shutil.rmtree(job_root, ignore_errors=True)

        for app_name, fed_app in self.fed_apps.items():
            self.custom_modules = []
            config_dir = os.path.join(job_root, self.job_name, app_name, CONFIG)
            custom_dir = os.path.join(job_root, self.job_name, app_name, CUSTOM)
            os.makedirs(config_dir, exist_ok=True)

            if fed_app.server_app:
                self._get_server_app(config_dir, custom_dir, fed_app)

            if fed_app.client_app:
                self._get_client_app(config_dir, custom_dir, fed_app)

        self._generate_meta(job_root)

    def simulator_run(self, job_root, workspace, clients=None, n_clients=None, threads=None, gpu=None):
        self.generate_job_config(job_root)

        simulator = SimulatorRunner(
            job_folder=os.path.join(job_root, self.job_name),
            workspace=workspace,
            clients=clients,
            n_clients=n_clients,
            threads=threads,
            gpu=gpu
        )
        simulator.run()

    def _get_server_app(self, config_dir, custom_dir, fed_app):
        server_app = {"format_version": 2, "workflows": []}
        for workflow in fed_app.server_app.workflows:
            server_app["workflows"].append(
                {
                    "id": workflow.id,
                    "path": self._get_class_path(workflow.controller, custom_dir),
                    "args": self._get_args(workflow.controller)
                }
            )
        self._get_base_app(custom_dir, fed_app.server_app, server_app)
        server_config = os.path.join(config_dir, FED_SERVER_JSON)
        with open(server_config, "w") as outfile:
            json_dump = json.dumps(server_app, indent=4)
            outfile.write(json_dump)

    def _get_class_path(self, obj, custom_dir):
        module = obj.__module__
        source_file = inspect.getsourcefile(obj.__class__)
        self._get_custom_file(custom_dir, module, source_file)

        return obj.__module__ + "." + obj.__class__.__name__

    def _get_custom_file(self, custom_dir, module, source_file):
        package = module.split(".")[0]
        if os.path.exists(source_file):
            if package not in FL_PACKAGES and module not in self.custom_modules:
                os.makedirs(custom_dir, exist_ok=True)
                dest_file = os.path.join(custom_dir, module.replace(".", os.sep) + ".py")

                with open(source_file, "r") as sf:
                    import_lines = list(self.locate_imports(sf, dest_file))

                self.custom_modules.append(module)
                for line in import_lines:
                    import_module = line.split(" ")[1]
                    import_source_file = import_module.replace(".", os.sep) + ".py"
                    if os.path.exists(import_source_file):
                        self._get_custom_file(custom_dir, import_module, import_source_file)

    def _get_client_app(self, config_dir, custom_dir, fed_app):
        client_app = {"format_version": 2, "executors": []}
        for e in fed_app.client_app.executors:
            client_app["executors"].append(
                {
                    "tasks": e.tasks,
                    "executor": {
                        "path": self._get_class_path(e.executor, custom_dir),
                        "args": self._get_args(e.executor)
                    }
                }
            )
        self._get_base_app(custom_dir, fed_app.client_app, client_app)
        client_config = os.path.join(config_dir, FED_CLIENT_JSON)
        with open(client_config, "w") as outfile:
            json_dump = json.dumps(client_app, indent=4)
            outfile.write(json_dump)

    def _get_base_app(self, custom_dir, app, app_config):
        app_config["components"] = []
        for cid, component in app.components.items():
            app_config["components"].append(
                {
                    "id": cid,
                    "path": self._get_class_path(component, custom_dir),
                    "args": self._get_args(component)
                }
            )
        app_config["task_data_filters"] = []
        for tasks, filter in app.task_data_filters:
            app_config["task_data_filters"].append(
                {
                    "tasks": tasks,
                    "filters": [
                        {
                            # self._get_filters(task_filter.filter, custom_dir)
                            "path": self._get_class_path(filter, custom_dir)
                        }
                    ]
                }
            )
        app_config["task_result_filters"] = []
        for tasks, filter in app.task_result_filters:
            app_config["task_result_filters"].append(
                {
                    "tasks": tasks,
                    "filters": [
                        {
                            # self._get_filters(result_filer.filter, custom_dir)
                            "path": self._get_class_path(filter, custom_dir)
                        }
                    ]
                }
            )

    def _get_args(self, component):
        constructor = component.__class__.__init__
        parameters = inspect.signature(constructor).parameters
        attrs = component.__dict__
        args = {}
        for param in parameters:
            attr_key = param if param in attrs.keys() else "_" + param
            if attr_key in attrs.keys() and parameters[param].default != attrs[attr_key]:
                args[param] = attrs[attr_key]

        return args

    def _get_filters(self, filters, custom_dir):
        r = []
        for f in filters:
            r.append(
                {
                    "path": self._get_class_path(f, custom_dir)
                }
            )
        return r

    def locate_imports(self, sf, dest_file):
        with open(dest_file, "w") as df:
            for line in sf:
                df.write(line)
                trimmed = line.strip()
                if trimmed.startswith('from ') and ('import ' in trimmed):
                    yield trimmed
                elif trimmed.startswith('import '):
                    yield trimmed

    def _get_deploy_map(self):
        deploy_map = {}
        for site, app_name in self.deploy_map.items():
            deploy_map[app_name] = deploy_map.get(app_name, [])
            deploy_map[app_name].append(site)
        return deploy_map
