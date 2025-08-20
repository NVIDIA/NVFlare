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


def _retrieve_parameters(class__, parameters, visited=None):
    if visited is None:
        visited = set()

    # Prevent infinite recursion in case of circular inheritance
    if class__ in visited:
        return parameters
    visited.add(class__)

    constructor = class__.__init__
    constructor__parameters = inspect.signature(constructor).parameters

    # Only add parameters that don't already exist (child class takes precedence)
    for param_name, param_obj in constructor__parameters.items():
        if param_name not in parameters:
            parameters[param_name] = param_obj

    # Check if this constructor has *args and **kwargs (indicating it passes arguments to parent)
    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in constructor__parameters.values())
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in constructor__parameters.values())

    if has_var_positional and has_var_keyword:
        for base_class in class__.__bases__:
            # Only traverse classes that have __init__ methods (not object)
            if hasattr(base_class, "__init__") and base_class is not object:
                _retrieve_parameters(base_class, parameters, visited)

    return parameters


def get_component_init_parameters(component):
    """To retrieve the initialize parameters of an object from the class constructor.

    Args:
        component: a class instance

    Returns:
        Dict: A dictionary containing parameter names as keys and Parameter objects as values

    """
    class__ = component.__class__
    parameters = {}
    _retrieve_parameters(class__, parameters)
    return parameters


def resolve_component_attribute_key(component, param_name):
    """Resolve the correct attribute key for a parameter, handling underscore prefixes and ambiguity.

    Args:
        component: The component instance to check
        param_name: The parameter name to resolve

    Returns:
        str: The correct attribute key to use, or None if neither exists

    Raises:
        ValueError: If there's ambiguous attribute naming (both param and _param exist)
                   or if there's ambiguity between property and instance variable
    """
    has_param = hasattr(component, param_name)
    has_underscore_param = hasattr(component, "_" + param_name)

    if has_param and has_underscore_param:
        raise ValueError(
            f"Ambiguous attribute naming in {component.__class__.__name__}: "
            f"both '{param_name}' and '_{param_name}' exist. Use only one."
        )

    # Check for property vs instance variable ambiguity
    if has_param:
        is_property = isinstance(getattr(component.__class__, param_name, None), property)
        is_in_instance_dict = param_name in getattr(component, "__dict__", {})

        if is_property and is_in_instance_dict:
            raise ValueError(
                f"Ambiguous attribute access in {component.__class__.__name__}: "
                f"'{param_name}' exists as both property and instance variable."
            )

        return param_name
    elif has_underscore_param:
        return "_" + param_name
    else:
        return None  # Neither attribute exists
