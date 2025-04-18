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
import importlib
import inspect
import pkgutil
from typing import Dict, List, Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.class_loader import load_class
from nvflare.fuel.utils.components_utils import create_classes_table_static
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.security.logging import secure_format_exception

DEPRECATED_PACKAGES = ["nvflare.app_common.pt", "nvflare.app_common.homomorphic_encryption"]


def instantiate_class(class_path, init_params):
    """Method for creating an instance for the class.

    Args:
        class_path: full path of the class
        init_params: A dictionary that contains the name of the transform and constructor input
        arguments. The transform name will be appended to `medical.common.transforms` to make a
        full name of the transform to be built.
    """
    c = load_class(class_path)
    try:
        if init_params:
            instance = c(**init_params)
        else:
            instance = c()
    except TypeError as e:
        raise ValueError(f"Class {class_path} has parameters error: {secure_format_exception(e)}.")

    return instance


class ModuleScanner:
    def __init__(self, base_pkgs: List[str], module_names: List[str], exclude_libs=True):
        """Loads specified modules from base packages and then constructs a class to module name mapping.

        Args:
            base_pkgs: base packages to look for modules in
            module_names: module names to load
            exclude_libs: excludes modules containing .libs if True. Defaults to True.
        """
        self.base_pkgs = base_pkgs
        self.module_names = module_names
        self.exclude_libs = exclude_libs

        self._logger = get_obj_logger(self)
        self._class_table = create_classes_table_static()

    def create_classes_table(self):
        class_table: Dict[str, list[str]] = {}
        for base in self.base_pkgs:
            package = importlib.import_module(base)

            for module_info in pkgutil.walk_packages(path=package.__path__, prefix=package.__name__ + "."):
                module_name = module_info.name
                if any(module_name.startswith(deprecated_package) for deprecated_package in DEPRECATED_PACKAGES):
                    continue
                if module_name.startswith(base):
                    if not self.exclude_libs or (".libs" not in module_name):
                        if any(module_name.startswith(base + "." + name + ".") for name in self.module_names):
                            try:
                                module = importlib.import_module(module_name)
                                for name, obj in inspect.getmembers(module):
                                    if (
                                        not name.startswith("_")
                                        and inspect.isclass(obj)
                                        and obj.__module__ == module_name
                                        and issubclass(obj, FLComponent)
                                    ):
                                        if name in class_table:
                                            class_table[name].append(module_name)
                                        else:
                                            class_table[name] = [module_name]
                            except (ModuleNotFoundError, RuntimeError, AttributeError) as e:
                                self._logger.error(
                                    f"Try to import module {module_name}, but failed: {secure_format_exception(e)}. "
                                    f"Can't use name in config to refer to classes in module: {module_name}."
                                )
                                pass
        return class_table

    def get_module_name(self, class_name) -> Optional[str]:
        """Gets the name of the module that contains this class.

        Args:
            class_name: The name of the class

        Returns:
            The module name if found.
        """
        if class_name not in self._class_table:
            raise ConfigError(
                f"Cannot find class '{class_name}'. Please check its spelling. If the spelling is correct, "
                "specify the class using its full path."
            )

        modules = self._class_table.get(class_name, None)
        if modules and len(modules) > 1:
            raise ConfigError(
                f"Multiple modules have the class '{class_name}': {modules}. "
                f"Please specify the class using its full path."
            )
        else:
            return modules[0]


def _retrieve_parameters(class__, parameters):
    constructor = class__.__init__
    constructor__parameters = inspect.signature(constructor).parameters
    parameters.update(constructor__parameters)
    if "args" in constructor__parameters.keys() and "kwargs" in constructor__parameters.keys():
        for item in class__.__bases__:
            parameters.update(_retrieve_parameters(item, parameters))
    return parameters


def get_component_init_parameters(component):
    """To retrieve the initialize parameters of an object from the class constructor.

    Args:
        component: a class instance

    Returns:

    """
    class__ = component.__class__
    parameters = {}
    _retrieve_parameters(class__, parameters)
    return parameters
