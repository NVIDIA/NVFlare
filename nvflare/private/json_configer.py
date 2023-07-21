# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from typing import List, Union

from nvflare.fuel.common.excepts import ComponentNotAuthorized, ConfigError
from nvflare.fuel.utils.class_utils import ModuleScanner, get_class
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.dict_utils import augment, extract_first_level_primitive
from nvflare.fuel.utils.json_scanner import JsonObjectProcessor, JsonScanner, Node
from nvflare.fuel.utils.wfconf import _EnvUpdater
from nvflare.security.logging import secure_format_exception


class ConfigContext(object):
    def __init__(self):
        """To init thee ConfigContext."""
        self.config_json = None
        self.pass_num = 0


class JsonConfigurator(JsonObjectProcessor, ComponentBuilder):
    def __init__(
        self,
        config_file_name: Union[str, List[str]],
        base_pkgs: List[str],
        module_names: List[str],
        exclude_libs=True,
        num_passes=1,
    ):
        """To init the JsonConfigurator.

        Args:
            config_file_name: config filename or list of JSON config file names
            base_pkgs: base packages need to be scanned
            module_names: module names need to be scanned
            exclude_libs: True/False to exclude the libs folder
            num_passes: number of passes to parsing the config
        """
        JsonObjectProcessor.__init__(self)

        if not isinstance(num_passes, int):
            raise TypeError(f"num_passes must be int but got {num_passes}")
        if not num_passes > 0:
            raise ValueError(f"num_passes must > 0 but got {num_passes}")

        if isinstance(config_file_name, str):
            config_files = [config_file_name]
        elif isinstance(config_file_name, list):
            config_files = config_file_name
        else:
            raise TypeError(f"config_file_name must be str or list of strs but got {type(config_file_name)}")

        for f in config_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"config_file_name {f} does not exist")
            if not os.path.isfile(f):
                raise FileNotFoundError(f"config_file_name {f} is not a valid file")

        self.config_file_name = config_files
        self.num_passes = num_passes
        self.module_scanner = ModuleScanner(base_pkgs, module_names, exclude_libs)
        self.config_ctx = None

        config_data = {}
        for f in config_files:
            with open(f) as file:
                try:
                    data = json.load(file)
                    augment(to_dict=config_data, from_dict=data, from_override_to=False)
                except Exception as e:
                    print("Error processing config file {}: {}".format(file, secure_format_exception(e)))
                    raise e

        self.config_data = config_data
        self.json_scanner = JsonScanner(config_data, config_files)
        self.build_auth_func = None
        self.build_auth_kwargs = None

    def set_component_build_authorizer(self, func, **kwargs):
        if not callable(func):
            raise ValueError("authorizer func is not callable")
        self.build_auth_func = func
        self.build_auth_kwargs = kwargs

    def authorize_and_build_component(self, config_dict, config_ctx: ConfigContext, node: Node):
        if self.build_auth_func is not None:
            err = self.build_auth_func(config_dict, config_ctx, node, **self.build_auth_kwargs)
            if err:
                raise ComponentNotAuthorized(f"component not authorized: {err}")
        return self.build_component(config_dict)

    def get_module_scanner(self):
        return self.module_scanner

    def _do_configure(self):
        config_ctx = ConfigContext()
        config_ctx.config_json = self.config_data
        self.config_ctx = config_ctx

        all_vars = extract_first_level_primitive(self.config_data)
        self.json_scanner.scan(_EnvUpdater(all_vars))

        self.start_config(self.config_ctx)

        # scan the config to create components
        for i in range(self.num_passes):
            self.config_ctx.pass_num = i + 1
            self.json_scanner.scan(self)

        # finalize configuration
        self.finalize_config(self.config_ctx)

    def configure(self):
        try:
            self._do_configure()
        except Exception as e:
            print("Error processing config {}: {}".format(self.config_file_name, secure_format_exception(e)))
            raise e

    def process_element(self, node: Node):
        self.process_config_element(self.config_ctx, node)

    def is_configured_subclass(self, config_dict, base_class):
        return issubclass(get_class(self.get_class_path(config_dict)), base_class)

    def start_config(self, config_ctx: ConfigContext):
        pass

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        pass

    def finalize_config(self, config_ctx: ConfigContext):
        pass


def get_component_refs(component):
    if "name" in component:
        name = component["name"]
        key = "name"
    elif "path" in component:
        name = component["path"]
        key = "path"
    else:
        raise ConfigError('component has no "name" or "path')

    parts = name.split("#")
    component[key] = parts[0]
    return parts
