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
import builtins
import inspect
import json
import os
import shutil
from enum import Enum
from tempfile import TemporaryDirectory
from typing import Dict

from nvflare import SimulatorRunner
from nvflare.job_config.fed_app_config import FedAppConfig
from nvflare.private.fed.app.fl_conf import FL_PACKAGES

CONFIG = "config"
CUSTOM = "custom"
FED_SERVER_JSON = "config_fed_server.json"
FED_CLIENT_JSON = "config_fed_client.json"
META_JSON = "meta.json"


class FedJobConfig:
    """FedJobConfig represents the job in the NVFlare."""

    def __init__(self, job_name, min_clients, mandatory_clients=None) -> None:
        """FedJobConfig uses the job_name,  min_clients and optional mandatory_clients to create the object.
        It also provides the method to add in the FedApp, the deployment map of the FedApp and participants,
        and the resource _spec requirements of the participants if needed.

        Args:
            job_name: the name of the NVFlare job
            min_clients: the minimum number of clients for the job
            mandatory_clients: mandatory clients to run the job (optional)
        """
        super().__init__()

        self.job_name = job_name
        self.min_clients = min_clients
        self.mandatory_clients = mandatory_clients

        self.fed_apps: Dict[str, FedAppConfig] = {}
        self.deploy_map: Dict[str, str] = {}
        self.resource_specs: Dict[str, Dict] = {}

        self.custom_modules = []

    def add_fed_app(self, app_name: str, fed_app: FedAppConfig):
        if not isinstance(fed_app, FedAppConfig):
            raise RuntimeError(f"server_app must be type of FedAppConfig, but got {fed_app.__class__}")

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

    def _generate_meta(self, job_dir):
        """generate the job meta.json

        Returns:

        """
        meta_file = os.path.join(job_dir, META_JSON)
        meta_json = {
            "name": self.job_name,
            "resource_spec": self.resource_specs,
            "min_clients": self.min_clients,
            "deploy_map": self._get_deploy_map(),
        }
        if self.mandatory_clients:
            meta_json["mandatory_clients"] = self.mandatory_clients

        with open(meta_file, "w") as outfile:
            json_dump = json.dumps(meta_json, indent=4)
            outfile.write(json_dump)

    def generate_job_config(self, job_root):
        """generate the job config

        Returns:

        """
        job_dir = os.path.join(job_root, self.job_name)
        if os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)

        for app_name, fed_app in self.fed_apps.items():
            self.custom_modules = []
            config_dir = os.path.join(job_dir, app_name, CONFIG)
            custom_dir = os.path.join(job_dir, app_name, CUSTOM)
            os.makedirs(config_dir, exist_ok=True)

            if fed_app.server_app:
                self._get_server_app(config_dir, custom_dir, fed_app)

            if fed_app.client_app:
                self._get_client_app(config_dir, custom_dir, fed_app)

        self._generate_meta(job_dir)

    def simulator_run(self, workspace, clients=None, n_clients=None, threads=None, gpu=None):
        with TemporaryDirectory() as job_root:
            self.generate_job_config(job_root)

            simulator = SimulatorRunner(
                job_folder=os.path.join(job_root, self.job_name),
                workspace=workspace,
                clients=clients,
                n_clients=n_clients,
                threads=threads,
                gpu=gpu,
            )
            simulator.run()

    def _get_server_app(self, config_dir, custom_dir, fed_app):
        server_app = {"format_version": 2, "workflows": []}
        for workflow in fed_app.server_app.workflows:
            server_app["workflows"].append(
                {
                    "id": workflow.id,
                    "path": self._get_class_path(workflow.controller, custom_dir),
                    "args": self._get_args(workflow.controller, custom_dir),
                }
            )
        self._get_base_app(custom_dir, fed_app.server_app, server_app)
        server_config = os.path.join(config_dir, FED_SERVER_JSON)
        with open(server_config, "w") as outfile:
            json_dump = json.dumps(server_app, indent=4)
            outfile.write(json_dump)

        self._copy_ext_scripts(custom_dir, fed_app.server_app.ext_scripts)

    def _copy_ext_scripts(self, custom_dir, ext_scripts):
        for script in ext_scripts:
            dest_file = os.path.join(custom_dir, script)
            module = "".join(script.rsplit(".py", 1)).replace(os.sep, ".")
            self._copy_source_file(custom_dir, module, script, dest_file)

    def _get_class_path(self, obj, custom_dir):
        module = obj.__module__
        source_file = inspect.getsourcefile(obj.__class__)
        self._get_custom_file(custom_dir, module, source_file)

        return obj.__module__ + "." + obj.__class__.__name__

    def _get_custom_file(self, custom_dir, module, source_file):
        package = module.split(".")[0]
        if os.path.exists(source_file):
            if package not in FL_PACKAGES and module not in self.custom_modules:
                module_path = module.replace(".", os.sep)
                if module_path in source_file:
                    index = source_file.index(module_path)
                    dest = source_file[index:]

                    self.custom_modules.append(module)
                    os.makedirs(custom_dir, exist_ok=True)
                    # dest_file = os.path.join(custom_dir, module.replace(".", os.sep) + ".py")
                    dest_file = os.path.join(custom_dir, dest)

                    self._copy_source_file(custom_dir, module, source_file, dest_file)

    def _copy_source_file(self, custom_dir, module, source_file, dest_file):
        os.makedirs(custom_dir, exist_ok=True)
        source_dir = os.path.dirname(source_file)
        with open(source_file, "r") as sf:
            import_lines = list(self.locate_imports(sf, dest_file))
        for line in import_lines:
            import_module = line.split(" ")[1]

            import_source = import_module
            if import_module.startswith("."):
                import_source = import_source[1:]
                new_module = module.split(".")[0:-1]
                new_module.append(import_source)
                import_module = ".".join(new_module)

            import_source_file = os.path.join(source_dir, import_source.replace(".", os.sep) + ".py")
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
                        "args": self._get_args(e.executor, custom_dir),
                    },
                }
            )
        self._get_base_app(custom_dir, fed_app.client_app, client_app)
        client_config = os.path.join(config_dir, FED_CLIENT_JSON)
        with open(client_config, "w") as outfile:
            json_dump = json.dumps(client_app, indent=4)
            outfile.write(json_dump)

        self._copy_ext_scripts(custom_dir, fed_app.client_app.ext_scripts)

    def _get_base_app(self, custom_dir, app, app_config):
        app_config["components"] = []
        for cid, component in app.components.items():
            app_config["components"].append(
                {
                    "id": cid,
                    "path": self._get_class_path(component, custom_dir),
                    "args": self._get_args(component, custom_dir),
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
                            "path": self._get_class_path(filter, custom_dir),
                            "args": self._get_args(filter, custom_dir),
                        }
                    ],
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
                            "path": self._get_class_path(filter, custom_dir),
                            "args": self._get_args(filter, custom_dir),
                        }
                    ],
                }
            )

    def _get_args(self, component, custom_dir):
        parameters = self._get_init_parameters(component)
        attrs = component.__dict__
        args = {}

        for param in parameters:
            attr_key = param if param in attrs.keys() else "_" + param

            if attr_key in ["args", "kwargs"]:
                continue

            if attr_key in attrs.keys() and parameters[param].default != attrs[attr_key]:
                if type(attrs[attr_key]).__name__ in dir(builtins):
                    args[param] = attrs[attr_key]
                elif issubclass(attrs[attr_key].__class__, Enum):
                    args[param] = attrs[attr_key].value
                else:
                    args[param] = {
                        "path": self._get_class_path(attrs[attr_key], custom_dir),
                        "args": self._get_args(attrs[attr_key], custom_dir),
                    }

        return args

    def _get_init_parameters(self, component):
        class__ = component.__class__
        parameters = {}
        self._retrieve_parameters(class__, parameters)
        return parameters

    def _retrieve_parameters(self, class__, parameters):
        constructor = class__.__init__
        constructor__parameters = inspect.signature(constructor).parameters
        parameters.update(constructor__parameters)
        if "args" in constructor__parameters.keys() and "kwargs" in constructor__parameters.keys():
            for item in class__.__bases__:
                parameters.update(self._retrieve_parameters(item, parameters))
        return parameters

    def _get_filters(self, filters, custom_dir):
        r = []
        for f in filters:
            r.append({"path": self._get_class_path(f, custom_dir), "args": self._get_args(f, custom_dir)})
        return r

    def locate_imports(self, sf, dest_file):
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        with open(dest_file, "w") as df:
            for line in sf:
                df.write(line)
                trimmed = line.strip()
                if trimmed.startswith("from ") and ("import " in trimmed):
                    yield trimmed
                elif trimmed.startswith("import "):
                    yield trimmed

    def _get_deploy_map(self):
        deploy_map = {}
        for site, app_name in self.deploy_map.items():
            deploy_map[app_name] = deploy_map.get(app_name, [])
            deploy_map[app_name].append(site)
        return deploy_map
