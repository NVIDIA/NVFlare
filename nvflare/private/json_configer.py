# Copyright (c) 2021, NVIDIA CORPORATION.
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

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.class_utils import ModuleScanner, get_class
from nvflare.fuel.utils.component_builder import ComponentBuilder
from nvflare.fuel.utils.dict_utils import extract_first_level_primitive
from nvflare.fuel.utils.json_scanner import JsonObjectProcessor, JsonScanner, Node
from nvflare.fuel.utils.wfconf import _EnvUpdater


class ConfigContext(object):
    def __init__(self):
        self.config_json = None
        self.pass_num = 0


class JsonConfigurator(JsonObjectProcessor, ComponentBuilder):
    def __init__(
        self,
        config_file_name: str,
        base_pkgs: [str],
        module_names: [str],
        exclude_libs=True,
        num_passes=1,
    ):
        JsonObjectProcessor.__init__(self)

        assert isinstance(num_passes, int), "num_passes must be int"
        assert num_passes > 0, "num_passes must > 0"

        assert isinstance(config_file_name, str), "config_file_name must be str"
        assert os.path.isfile(config_file_name), "config_file_name {} is not a valid file".format(config_file_name)
        assert os.path.exists(config_file_name), "config_file_name {} does not exist".format(config_file_name)

        self.config_file_name = config_file_name
        self.num_passes = num_passes
        self.module_scanner = ModuleScanner(base_pkgs, module_names, exclude_libs)
        self.config_ctx = None

        with open(config_file_name) as file:
            self.config_data = json.load(file)

        self.json_scanner = JsonScanner(self.config_data, config_file_name)

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
        except ConfigError as ex:
            raise ConfigError("Config error in {}: {}".format(self.config_file_name, ex))
        except Exception as ex:
            print("Error processing config {}: {}".format(self.config_file_name, ex))
            raise ex

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
