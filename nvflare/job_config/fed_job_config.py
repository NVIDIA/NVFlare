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
import ast
import builtins
import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
from enum import Enum
from tempfile import TemporaryDirectory
from typing import Dict, List

from nvflare.fuel.utils.class_utils import get_component_init_parameters
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.job_config.base_app_config import BaseAppConfig
from nvflare.job_config.fed_app_config import FedAppConfig
from nvflare.private.fed.app.fl_conf import FL_PACKAGES
from nvflare.private.fed.app.utils import kill_child_processes

CONFIG = "config"
CUSTOM = "custom"
FED_SERVER_JSON = "config_fed_server.json"
FED_CLIENT_JSON = "config_fed_client.json"
META_JSON = "meta.json"


class FedJobConfig:
    """FedJobConfig represents the job in the NVFlare."""

    def __init__(self, job_name, min_clients, mandatory_clients=None, meta_props=None) -> None:
        """FedJobConfig uses the job_name,  min_clients and optional mandatory_clients to create the object.
        It also provides the method to add in the FedApp, the deployment map of the FedApp and participants,
        and the resource _spec requirements of the participants if needed.

        Args:
            job_name: the name of the NVFlare job
            min_clients: the minimum number of clients for the job
            mandatory_clients: mandatory clients to run the job (optional)
            meta_props: additional meta properties for the job (optional)
        """
        super().__init__()

        if meta_props:
            check_object_type("meta_props", meta_props, dict)

        self.job_name = job_name
        self.min_clients = min_clients
        self.mandatory_clients = mandatory_clients
        self.meta_props = meta_props
        self.app_packages = []

        self.fed_apps: Dict[str, FedAppConfig] = {}
        self.deploy_map: Dict[str, str] = {}
        self.resource_specs: Dict[str, Dict] = {}

        self.custom_modules = []
        self.logger = get_obj_logger(self)

    def set_app_packages(self, app_packages: List[str]):
        """Set app packages.
        When generating job config, code from these packages will not be included into "custom" folder.

        Args:
            app_packages: app packages

        Returns: None

        """
        if not app_packages:
            app_packages = []
        else:
            check_object_type("app_packages", app_packages, list)

        self.app_packages = app_packages

    def add_fed_app(self, app_name: str, fed_app: FedAppConfig):
        if not isinstance(fed_app, FedAppConfig):
            raise RuntimeError(f"server_app must be type of FedAppConfig, but got {fed_app.__class__}")

        self.fed_apps[app_name] = fed_app

    def set_site_app(self, site_name: str, app_name: str):
        """assign an app to a certain site.

        Args:
            site_name: The target site name.
            app_name: The app name.

        Returns:

        """
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

        if self.meta_props:
            meta_json.update(self.meta_props)

        with open(meta_file, "w") as outfile:
            json_dump = json.dumps(meta_json, indent=4)
            outfile.write(json_dump)

    def generate_job_config(self, job_root):
        """generate the job config

        Returns:

        """
        job_dir = os.path.join(job_root, self.job_name)
        if os.path.exists(job_dir):
            if self._is_valid_job_folder(job_dir) or self._is_partial_export_folder(job_dir):
                shutil.rmtree(job_dir, ignore_errors=True)
            else:
                raise RuntimeError(f"Job folder {job_dir} already exists and is not a valid job folder.")

        for app_name, fed_app in self.fed_apps.items():
            self.custom_modules = []
            config_dir = os.path.join(job_dir, app_name, CONFIG)
            custom_dir = os.path.join(job_dir, app_name, CUSTOM)
            os.makedirs(config_dir, exist_ok=True)
            # custom_dir will be created on-demand if custom code is added.

            if fed_app.server_app:
                self._get_server_app(config_dir, custom_dir, fed_app)

            if fed_app.client_app:
                self._get_client_app(config_dir, custom_dir, fed_app)

        self._generate_meta(job_dir)

    def simulator_run(self, workspace, clients=None, n_clients=None, threads=None, gpu=None, log_config=None):
        with TemporaryDirectory() as job_root:
            self.generate_job_config(job_root)

            try:
                command = (
                    f"{sys.executable} -m nvflare.private.fed.app.simulator.simulator "
                    + os.path.join(job_root, self.job_name)
                    + " -w "
                    + workspace
                )
                if clients:
                    clients = self._trim_whitespace(clients)
                    command += " -c " + str(clients)
                if n_clients:
                    command += " -n " + str(n_clients)
                if threads:
                    command += " -t " + str(threads)
                if gpu:
                    gpu = self._trim_whitespace(gpu)
                    command += " -gpu " + str(gpu)
                if log_config:
                    command += " -l" + str(log_config)

                new_env = os.environ.copy()
                process = subprocess.Popen(shlex.split(command, True), shell=False, preexec_fn=os.setsid, env=new_env)

                return_code = process.wait()
                return return_code

            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt, terminate all the child processes.")
                kill_child_processes(os.getpid())
                return -9

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

        # Add additional system parameters to the server app config
        if fed_app.server_app.additional_params:
            server_app.update(fed_app.server_app.additional_params)

        server_config = os.path.join(config_dir, FED_SERVER_JSON)
        with open(server_config, "w") as outfile:
            json_dump = json.dumps(server_app, indent=4)
            outfile.write(json_dump)

        self._copy_ext_scripts(custom_dir, fed_app.server_app.ext_scripts)
        self._copy_ext_dirs(custom_dir, fed_app.server_app)
        self._copy_file_sources(config_dir, custom_dir, fed_app.server_app.file_sources)

    def _copy_file_sources(self, config_dir, custom_dir, file_sources):
        for s in file_sources:
            # s is a tuple of (src_path, dest_dir)
            src_path, dest_dir, app_folder_type = s

            if app_folder_type == "config":
                target_dir = config_dir
            else:
                target_dir = custom_dir

            if dest_dir:
                dest_path = os.path.join(target_dir, dest_dir)
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path, exist_ok=True)
            else:
                dest_path = target_dir

            if os.path.isfile(src_path):
                base_name = os.path.basename(src_path)
                dest_file = os.path.join(dest_path, base_name)
                os.makedirs(dest_path, exist_ok=True)
                shutil.copy(src_path, dest_file)
            else:
                # this is a dir
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)

    def _copy_ext_scripts(self, custom_dir, ext_scripts):
        for script in ext_scripts:
            if os.path.exists(script):
                if os.path.isabs(script):
                    relative_script = self._get_relative_script(script)
                else:
                    relative_script = script
                relative_script = os.path.normpath(relative_script)
                if (
                    relative_script in ("", os.curdir)
                    or os.path.isabs(relative_script)
                    or relative_script == os.pardir
                    or relative_script.startswith(os.pardir + os.sep)
                ):
                    raise ValueError(f"Invalid external script path: {script}")

                dest_file = os.path.join(custom_dir, relative_script)
                module_path = relative_script[:-3] if relative_script.endswith(".py") else relative_script
                if os.path.basename(module_path) == "__init__" and os.path.dirname(module_path):
                    module_path = os.path.dirname(module_path)
                module = module_path.replace(os.sep, ".")
                path_depth = len(module_path.split(os.sep))
                if os.path.basename(script) != "__init__.py":
                    path_depth -= 1
                source_root = os.path.dirname(os.path.abspath(script))
                for _ in range(max(path_depth, 1)):
                    source_root = os.path.dirname(source_root)
                self._copy_source_file(custom_dir, module, script, dest_file, source_root=source_root)

    def _copy_ext_dirs(self, custom_dir, app_config: BaseAppConfig):
        for dir in app_config.ext_dirs:
            shutil.copytree(dir, custom_dir, dirs_exist_ok=True)

    def _get_relative_script(self, script):
        script_path = os.path.abspath(script)
        package_path = None
        for path in sys.path:
            path = os.path.abspath(path or os.curdir)
            if self._is_path_within(script_path, path):
                if package_path is None or len(path) > len(package_path):
                    package_path = path
        if package_path:
            return os.path.relpath(script_path, package_path)
        return os.path.basename(script_path)

    def _get_class_path(self, obj, custom_dir):
        module = obj.__module__
        source_file = inspect.getsourcefile(obj.__class__)
        if module == "__main__":
            module = os.path.splitext(os.path.basename(source_file))[0]
        self._get_custom_file(custom_dir, module, source_file)

        return module + "." + obj.__class__.__name__

    @staticmethod
    def _resolved_path(path):
        return os.path.normcase(os.path.realpath(os.path.abspath(path)))

    @classmethod
    def _is_path_within(cls, path, root):
        path = cls._resolved_path(path)
        root = cls._resolved_path(root)
        try:
            return os.path.commonpath([path, root]) == root
        except ValueError:
            return False

    @staticmethod
    def _module_parts(module):
        parts = module.split(".") if module else []
        if not parts or any(not part.isidentifier() for part in parts):
            raise ValueError(f"Invalid module path: {module}")
        return parts

    def _derive_source_root(self, module, source_file):
        module_parts = self._module_parts(module)
        if os.path.basename(source_file) == "__init__.py":
            expected_source = os.path.join(*module_parts, "__init__.py")
        else:
            expected_source = os.path.join(*module_parts) + ".py"

        source_file = os.path.normpath(os.path.abspath(source_file))
        source_parts = os.path.normcase(source_file).split(os.sep)
        expected_parts = os.path.normcase(expected_source).split(os.sep)
        if source_parts[-len(expected_parts) :] != expected_parts:
            raise ValueError(f"Source path '{source_file}' does not match module '{module}'")

        source_root = source_file
        for _ in expected_parts:
            source_root = os.path.dirname(source_root)
        return source_root

    def _validate_source_path(self, source_file, source_root):
        source_file = self._resolved_path(source_file)
        source_root = self._resolved_path(source_root)
        if not self._is_path_within(source_file, source_root):
            raise ValueError(f"Source path '{source_file}' resolves outside the allowed source root '{source_root}'")
        return source_file, source_root

    def _validate_copy_paths(self, custom_dir, source_file, source_root, dest_file):
        os.makedirs(custom_dir, exist_ok=True)
        source_file, source_root = self._validate_source_path(source_file, source_root)
        custom_dir = self._resolved_path(custom_dir)
        dest_file = self._resolved_path(dest_file)
        if not self._is_path_within(dest_file, custom_dir):
            raise ValueError(f"Destination path '{dest_file}' resolves outside the custom directory '{custom_dir}'")
        paths_are_same = source_file == dest_file
        if not paths_are_same and os.path.exists(dest_file):
            try:
                paths_are_same = os.path.samefile(source_file, dest_file)
            except OSError:
                paths_are_same = False
        if paths_are_same:
            raise ValueError(f"Source and destination resolve to the same file: {source_file}")
        return source_file, source_root, dest_file

    def _get_custom_file(self, custom_dir, module, source_file, source_root=None):
        module_parts = self._module_parts(module)
        if source_root is None:
            source_root = self._derive_source_root(module=module, source_file=source_file)
        source_file, source_root = self._validate_source_path(source_file, source_root)

        package = module_parts[0]
        if package in FL_PACKAGES or package in self.app_packages or module in self.custom_modules:
            return

        if os.path.basename(source_file) == "__init__.py":
            dest = os.path.join(*module_parts, "__init__.py")
        else:
            dest = os.path.join(*module_parts) + ".py"
        dest_file = os.path.join(custom_dir, dest)

        self.custom_modules.append(module)
        try:
            self._copy_source_file(custom_dir, module, source_file, dest_file, source_root=source_root)
        except Exception:
            self.custom_modules.remove(module)
            raise

    def _resolve_import_module(self, module, import_source, level, source_file):
        import_parts = self._module_parts(import_source) if import_source else []
        if level == 0:
            return ".".join(import_parts)

        module_parts = self._module_parts(module)
        if os.path.basename(source_file) == "__init__.py":
            package_parts = module_parts
        else:
            package_parts = module_parts[:-1]
        if level > len(package_parts):
            relative_import = "." * level + (import_source or "*")
            raise ValueError(
                f"Relative import '{relative_import}' from module '{module}' escapes the allowed source root"
            )
        keep_parts = len(package_parts) - level + 1
        resolved_parts = package_parts[:keep_parts] + import_parts
        return ".".join(resolved_parts) if resolved_parts else None

    def _copy_source_file(self, custom_dir, module, source_file, dest_file, source_root):
        source_file, source_root, dest_file = self._validate_copy_paths(
            custom_dir=custom_dir,
            source_file=source_file,
            source_root=source_root,
            dest_file=dest_file,
        )
        import_specs = []
        if source_file.endswith(".py"):
            with open(source_file, "rb") as sf:
                import_specs = list(self.locate_imports(sf))

        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copyfile(source_file, dest_file)

        source_dir = os.path.dirname(source_file)
        for import_source, level in import_specs:
            import_module = self._resolve_import_module(module, import_source, level, source_file)
            if not import_module:
                continue
            import_path = os.path.join(*self._module_parts(import_module)) + ".py"
            search_roots = [source_root] if level else [source_dir, source_root]
            checked_roots = set()
            for search_root in search_roots:
                search_root = self._resolved_path(search_root)
                if search_root in checked_roots:
                    continue
                checked_roots.add(search_root)
                import_source_file = os.path.join(search_root, import_path)
                if os.path.isfile(import_source_file):
                    self._get_custom_file(
                        custom_dir,
                        import_module,
                        import_source_file,
                        source_root=source_root,
                    )
                    break

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

        # Add additional system parameters to the client app config
        if fed_app.client_app.additional_params:
            client_app.update(fed_app.client_app.additional_params)

        client_config = os.path.join(config_dir, FED_CLIENT_JSON)
        with open(client_config, "w") as outfile:
            json_dump = json.dumps(client_app, indent=4)
            outfile.write(json_dump)

        self._copy_ext_scripts(custom_dir, fed_app.client_app.ext_scripts)
        self._copy_ext_dirs(custom_dir, fed_app.client_app)
        self._copy_file_sources(config_dir, custom_dir, fed_app.client_app.file_sources)

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

        app_config["task_data_filters"] = self._process_filters(app.task_data_filters, custom_dir)
        app_config["task_result_filters"] = self._process_filters(app.task_result_filters, custom_dir)

    def _process_filters(self, taskset_filters: list, custom_dir):
        """Process taskset_filters into app filter configuration

        Args:
            taskset_filters: the list of tuples that contain taskset/filters association.
            custom_dir: custom dir of the app.

        Returns: app filter configuration that is a list of dicts, each dict represents a taskset/filters
            association.

        """
        app_config_filters = []
        for task_set, filter_list in taskset_filters:
            filters = []
            for f in filter_list:
                filters.append(
                    {
                        "path": self._get_class_path(f, custom_dir),
                        "args": self._get_args(f, custom_dir),
                    }
                )

            app_config_filters.append({"tasks": list(task_set), "filters": filters})
        return app_config_filters

    def _values_differ(self, default_val, attr_val):
        """Check if attribute value differs from default. Returns True if they differ."""
        # Handle None values
        if default_val is None or attr_val is None:
            return default_val is not attr_val

        # General comparison
        try:
            result = default_val != attr_val
            # Ensure we get a boolean (numpy arrays return arrays, not bool)
            if isinstance(result, bool):
                return result
            # Non-bool result, assume different
            return True
        except Exception:
            return True

    def _get_args(self, component, custom_dir):
        args = {}
        if hasattr(component, "__dict__"):
            parameters = get_component_init_parameters(component)
            attrs = component.__dict__
            always_serialize_args = set(getattr(component, "_always_serialize_args", ()))

            for param in parameters:
                attr_key = param if param in attrs.keys() else "_" + param

                if attr_key in ["args", "kwargs"]:
                    continue

                if attr_key in attrs.keys() and (
                    param in always_serialize_args or self._values_differ(parameters[param].default, attrs[attr_key])
                ):
                    if attrs[attr_key] is None or type(attrs[attr_key]).__name__ in dir(builtins):
                        args[param] = attrs[attr_key]
                    elif issubclass(attrs[attr_key].__class__, Enum):
                        args[param] = attrs[attr_key].value
                    else:
                        args[param] = {
                            "path": self._get_class_path(attrs[attr_key], custom_dir),
                            "args": self._get_args(attrs[attr_key], custom_dir),
                        }

        return args

    def _get_filters(self, filters, custom_dir):
        r = []
        for f in filters:
            r.append({"path": self._get_class_path(f, custom_dir), "args": self._get_args(f, custom_dir)})
        return r

    def locate_imports(self, sf):
        """Locate imported modules in a Python source file.

        Args:
            sf: source file

        Returns:
            yield (module name or None, relative import level) tuples

        """
        source = sf.read()
        source_file = getattr(sf, "name", "<unknown>")
        tree = ast.parse(source, filename=source_file)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for imported_name in node.names:
                    yield imported_name.name, 0
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    yield node.module, node.level
                else:
                    for imported_name in node.names:
                        import_source = None if imported_name.name == "*" else imported_name.name
                        yield import_source, node.level

    def _get_deploy_map(self):
        deploy_map = {}
        for site, app_name in self.deploy_map.items():
            deploy_map[app_name] = deploy_map.get(app_name, [])
            deploy_map[app_name].append(site)
        return deploy_map

    def _trim_whitespace(self, string: str):
        strings = string.split(",")
        for i in range(len(strings)):
            strings[i] = strings[i].strip()
        return ",".join(strings)

    @staticmethod
    def _is_valid_job_folder(job_folder: str) -> bool:
        meta_file = os.path.join(job_folder, META_JSON)
        return os.path.exists(meta_file)

    def _is_partial_export_folder(self, job_folder: str) -> bool:
        """True when a previous export created the directory but did not finish writing meta.json.

        A partial export only contains app-named subdirectories (no foreign files), so it is
        safe to delete and retry.  Any other content means the folder was not created by NVFlare.
        """
        try:
            app_names = set(self.fed_apps.keys())
            entries = os.listdir(job_folder)
            return all(os.path.isdir(os.path.join(job_folder, e)) and e in app_names for e in entries)
        except OSError:
            return False
