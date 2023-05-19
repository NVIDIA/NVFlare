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
import logging
import pkgutil
from typing import Dict, List, Optional

from nvflare.security.logging import secure_format_exception

DEPRECATED_PACKAGES = ["nvflare.app_common.pt", "nvflare.app_common.homomorphic_encryption"]


def get_class(class_path):
    module_name, class_name = class_path.rsplit(".", 1)

    try:
        module_ = importlib.import_module(module_name)

        try:
            class_ = getattr(module_, class_name)
        except AttributeError:
            raise ValueError("Class {} does not exist".format(class_path))
    except AttributeError:
        raise ValueError("Module {} does not exist".format(class_path))

    return class_


def instantiate_class(class_path, init_params):
    """Method for creating an instance for the class.

    Args:
        class_path: full path of the class
        init_params: A dictionary that contains the name of the transform and constructor input
        arguments. The transform name will be appended to `medical.common.transforms` to make a
        full name of the transform to be built.
    """
    c = get_class(class_path)
    try:
        if init_params:
            instance = c(**init_params)
        else:
            instance = c()
    except TypeError as e:
        raise ValueError(f"Class {class_path} has parameters error: {secure_format_exception(e)}.")

    return instance


class _ModuleScanResult:
    """Data class for ModuleScanner."""

    def __init__(self, class_name: str, module_name: str):
        self.class_name = class_name
        self.module_name = module_name

    def __str__(self):
        return f"{self.class_name}:{self.module_name}"


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

        self._logger = logging.getLogger(self.__class__.__name__)
        self._class_table: Dict[str, str] = {}
        self._create_classes_table()

    def _create_classes_table(self):
        scan_result_table = {}
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
                                    ):
                                        # same class name exists in multiple modules
                                        if name in scan_result_table:
                                            scan_result = scan_result_table[name]
                                            if name in self._class_table:
                                                self._class_table.pop(name)
                                                self._class_table[f"{scan_result.module_name}.{name}"] = module_name
                                            self._class_table[f"{module_name}.{name}"] = module_name
                                        else:
                                            scan_result = _ModuleScanResult(class_name=name, module_name=module_name)
                                            scan_result_table[name] = scan_result
                                            self._class_table[name] = module_name
                            except (ModuleNotFoundError, RuntimeError) as e:
                                self._logger.debug(
                                    f"Try to import module {module_name}, but failed: {secure_format_exception(e)}. "
                                    f"Can't use name in config to refer to classes in module: {module_name}."
                                )
                                pass

    def get_module_name(self, class_name) -> Optional[str]:
        """Gets the name of the module that contains this class.

        Args:
            class_name: The name of the class

        Returns:
            The module name if found.
        """
        return self._class_table.get(class_name, None)
