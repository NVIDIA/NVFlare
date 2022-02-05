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

import copy
import json
import os
import re
from typing import List

from nvflare.fuel.common.excepts import ConfigError

from .class_utils import ModuleScanner, get_class, instantiate_class
from .dict_utils import extract_first_level_primitive, merge_dict
from .json_scanner import JsonObjectProcessor, JsonScanner, Node


class ConfigContext(object):
    def __init__(self):
        """Object containing configuration context."""
        self.app_root = ""
        self.vars = None
        self.config_json = None
        self.pass_num = 0


class _EnvUpdater(JsonObjectProcessor):
    def __init__(self, vs, element_filter=None):
        JsonObjectProcessor.__init__(self)
        self.vars = vs
        if element_filter is not None and not callable(element_filter):
            raise ValueError("element_filter must be a callable function but got {}.".format(type(element_filter)))
        self.element_filter = element_filter

    def process_element(self, node: Node):
        element = node.element
        if isinstance(element, str):
            if self.element_filter is not None and not self.element_filter(element):
                return
            element = self.substitute(element)
            parent_element = node.parent_element()
            if node.position > 0:
                # parent is a list
                parent_element[node.position - 1] = element
            else:
                # parent is a dict
                parent_element[node.key] = element

    def substitute(self, element: str):
        a = re.split("{|}", element)
        if len(a) == 3 and a[0] == "" and a[2] == "":
            element = self.vars.get(a[1], None)
        else:
            element = element.format(**self.vars)
        return element


class Configurator(JsonObjectProcessor):
    def __init__(
        self,
        app_root: str,
        cmd_vars: dict,
        env_config: dict,
        wf_config_file_name: str,
        base_pkgs: List[str],
        module_names: List[str],
        exclude_libs=True,
        default_vars=None,
        num_passes=1,
        element_filter=None,
        var_processor=None,
    ):
        """Base class of Configurator to parse JSON configuration.

        Args:
            app_root: app root
            cmd_vars: command vars
            env_config: environment configuration
            wf_config_file_name: config file name
            base_pkgs: base packages
            module_names: module names
            exclude_libs: whether to exclude libs
            default_vars: default vars
            num_passes: number of passes
            element_filter: element filter
            var_processor: variable processor
        """
        JsonObjectProcessor.__init__(self)

        assert isinstance(app_root, str), "app_root must be str but got {}.".format(type(app_root))

        assert isinstance(num_passes, int), "num_passes must be int but got {}.".format(type(num_passes))
        assert num_passes > 0, "num_passes must > 0"

        if cmd_vars:
            assert isinstance(cmd_vars, dict), "cmd_vars must be dict but got {}.".format(type(cmd_vars))

        if env_config:
            assert isinstance(env_config, dict), "env_config must be dict but got {}.".format(type(env_config))

        assert isinstance(wf_config_file_name, str), "wf_config_file_name must be str but got {}.".format(
            type(wf_config_file_name)
        )
        assert os.path.isfile(wf_config_file_name), "wf_config_file_name {} is not a valid file".format(
            wf_config_file_name
        )
        assert os.path.exists(wf_config_file_name), "wf_config_file_name {} does not exist".format(wf_config_file_name)

        if default_vars is not None:
            assert isinstance(default_vars, dict), "default_vars must be dict but got {}.".format(type(default_vars))
        else:
            default_vars = {}

        self.cmd_vars = cmd_vars
        self.default_vars = default_vars
        self.app_root = app_root
        self.env_config = env_config
        self.wf_config_file_name = wf_config_file_name
        self.num_passes = num_passes
        self.element_filter = element_filter

        self.module_scanner = ModuleScanner(base_pkgs, module_names, exclude_libs)
        self.all_vars = None
        self.vars_from_cmd = None
        self.vars_from_env_config = None
        self.vars_from_wf_config = None
        self.config_ctx = None
        self.var_processor = var_processor

        with open(wf_config_file_name) as file:
            self.wf_config_data = json.load(file)

        self.json_scanner = JsonScanner(self.wf_config_data, wf_config_file_name)

    def _do_configure(self):
        vars_from_cmd = {}
        if self.cmd_vars:
            vars_from_cmd = copy.copy(self.cmd_vars)
            for key, value in vars_from_cmd.items():
                if key.startswith("APP_") and value != "":
                    vars_from_cmd[key] = os.path.join(self.app_root, value)

        vars_from_env_config = {}
        if self.env_config:
            vars_from_env_config = copy.copy(self.env_config)
            for key, value in vars_from_env_config.items():
                if key.startswith("APP_") and value != "":
                    vars_from_env_config[key] = os.path.join(self.app_root, value)

        vars_from_wf_conf = extract_first_level_primitive(self.wf_config_data)

        if "determinism" in self.wf_config_data:
            vars_from_wf_conf["determinism"] = self.wf_config_data["determinism"]

        # precedence of vars (high to low):
        #   vars_from_cmd, vars_from_config, vars_from_wf_conf
        # func merge_dict(d1, d2) gives d2 higher precedence for the same key
        all_vars = merge_dict(self.default_vars, vars_from_wf_conf)
        all_vars = merge_dict(all_vars, vars_from_env_config)
        all_vars = merge_dict(all_vars, vars_from_cmd)

        # update the wf_config with vars
        self.all_vars = all_vars
        self.vars_from_cmd = vars_from_cmd
        self.vars_from_env_config = vars_from_env_config
        self.vars_from_wf_config = vars_from_wf_conf

        if self.var_processor:
            self.var_processor.process(self.all_vars, app_root=self.app_root)

        self.json_scanner.scan(_EnvUpdater(all_vars, self.element_filter))

        config_ctx = ConfigContext()
        config_ctx.vars = self.all_vars
        config_ctx.app_root = self.app_root
        config_ctx.config_json = self.wf_config_data
        self.config_ctx = config_ctx

        self.start_config(self.config_ctx)

        # scan the wf_config again to create components
        for i in range(self.num_passes):
            self.config_ctx.pass_num = i + 1
            self.json_scanner.scan(self)

        # finalize configuration
        self.finalize_config(self.config_ctx)

    def configure(self):
        try:
            self._do_configure()
        except ConfigError as ex:
            raise ConfigError("Config error in {}: {}".format(self.wf_config_file_name, ex))
        except Exception as ex:
            print("Error processing config {}: {}".format(self.wf_config_file_name, ex))
            raise ex

    def process_element(self, node: Node):
        self.process_config_element(self.config_ctx, node)

    def process_args(self, args: dict):
        return args

    def build_component(self, config_dict):
        if not config_dict:
            return None

        if not isinstance(config_dict, dict):
            raise ConfigError("component config must be dict but got {}.".format(type(config_dict)))

        if config_dict.get("disabled") is True:
            return None

        class_args = config_dict.get("args", dict())
        class_args = self.process_args(class_args)

        class_path = self.get_class_path(config_dict)

        # Handle the special case, if config pass in the class_attributes, use the user defined class attributes
        # parameters directly.
        if "class_attributes" in class_args:
            class_args = class_args["class_attributes"]

        return instantiate_class(class_path, class_args)

    def get_class_path(self, config_dict):
        if "path" in config_dict.keys():
            path_spec = config_dict["path"]
            if not isinstance(path_spec, str):
                raise ConfigError("path spec must be str but got {}.".format(type(path_spec)))

            if len(path_spec) <= 0:
                raise ConfigError("path spec must not be empty")

            class_path = format(path_spec)
            parts = class_path.split(".")
            if len(parts) < 2:
                raise ConfigError("invalid class path '{}': missing module name".format(class_path))
        else:
            if "name" not in config_dict:
                raise ConfigError("class name or path must be specified")

            class_name = config_dict["name"]

            if not isinstance(class_name, str):
                raise ConfigError("class name must be str")

            if len(class_name) <= 0:
                raise ConfigError("class name must not be empty")
            module_name = self.module_scanner.get_module_name(class_name)
            if module_name is None:
                raise ConfigError('Cannot find component class "{}"'.format(class_name))
            class_path = module_name + ".{}".format(class_name)

        return class_path

    def is_configured_subclass(self, config_dict, base_class):
        return issubclass(get_class(self.get_class_path(config_dict)), base_class)

    def start_config(self, config_ctx: ConfigContext):
        pass

    def process_config_element(self, config_ctx: ConfigContext, node: Node):
        pass

    def finalize_config(self, config_ctx: ConfigContext):
        pass


def get_component_refs(component):
    """Get component reference.

    Args:
        component: string for component

    Returns: list of component and reference

    """
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
