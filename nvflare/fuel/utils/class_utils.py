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

import importlib
import inspect
import pkgutil
from typing import List


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
        raise ValueError("Class {} has parameters error.".format(class_path), str(e))

    return instance


def get_object_method(obj, method_name):
    op = getattr(obj, method_name, None)
    if op is None or not callable(op):
        return None
    return op


def get_instance_method(instance, method_name):
    return get_object_method(instance, method_name)


def get_config_classname(config_dict: dict):
    class_name = config_dict.get("name", None)
    if not class_name:
        class_name = config_dict.get("path", "")
    return class_name


class ModuleScanner:
    def __init__(self, base_pkgs: List[str], module_names: List[str], exclude_libs=True):
        """Scanner to look for and load specified module names.

        Args:
            base_pkgs: base packages to look for modules in
            module_names: module names to load
            exclude_libs: excludes modules containing .libs if True. Defaults to True.
        """
        self.base_pkgs = base_pkgs
        self.module_names = module_names
        self.exclude_libs = exclude_libs
        self._class_table = {}
        self._create_classes_table()

    def _create_classes_table(self):
        for base in self.base_pkgs:
            package = __import__(base)

            for importer, modname, ispkg in pkgutil.walk_packages(path=package.__path__, prefix=package.__name__ + "."):

                if modname.startswith(base):
                    if not self.exclude_libs or (".libs" not in modname):
                        if any(name in modname for name in self.module_names):
                            try:
                                module = importlib.import_module(modname)
                                for name, obj in inspect.getmembers(module):
                                    if inspect.isclass(obj) and obj.__module__ == modname:
                                        self._class_table[name] = modname
                            except ModuleNotFoundError as ex:
                                pass

    def get_module_name(self, class_name):
        return self._class_table.get(class_name, None)
