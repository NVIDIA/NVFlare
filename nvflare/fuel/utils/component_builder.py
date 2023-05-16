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

from abc import abstractmethod

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.class_utils import instantiate_class
from nvflare.security.logging import secure_format_exception


class ConfigType:
    COMPONENT = "component"
    DICT = "dict"


class ComponentBuilder:
    @abstractmethod
    def get_module_scanner(self):
        """Provide the package module scanner.

        Returns: module_scanner

        """
        pass

    def is_class_config(self, config_dict: dict) -> bool:
        def has_valid_class_path():
            try:
                _ = self.get_class_path(config_dict)
                # we have valid class path
                return True
            except ConfigError:
                # this is not a valid class path
                return False

        # use config_type to distinguish between components and regular dictionaries
        config_type = config_dict.get("config_type", ConfigType.COMPONENT)
        if config_type != ConfigType.COMPONENT:
            return False

        # regardless it has args or not. if path/name and valid class path, very likely we have
        # class config.
        if ("path" in config_dict or "name" in config_dict) and has_valid_class_path():
            return True
        else:
            return False

    def build_component(self, config_dict):
        if not config_dict:
            return None

        if not isinstance(config_dict, dict):
            raise ConfigError("component config must be dict but got {}.".format(type(config_dict)))

        if config_dict.get("disabled") is True:
            return None

        class_args = config_dict.get("args", dict())
        for k, v in class_args.items():
            if isinstance(v, dict) and self.is_class_config(v):
                # try to replace the arg with a component
                try:
                    t = self.build_component(v)
                    class_args[k] = t
                except Exception as e:
                    raise ValueError(f"failed to instantiate class: {secure_format_exception(e)} ")

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
                raise ConfigError("class name must be str but got {}.".format(type(class_name)))

            if len(class_name) <= 0:
                raise ConfigError("class name must not be empty")
            module_name = self.get_module_scanner().get_module_name(class_name)
            if module_name is None:
                raise ConfigError('Cannot find component class "{}"'.format(class_name))
            class_path = module_name + ".{}".format(class_name)

        return class_path
