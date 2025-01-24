# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import importlib
from typing import Type


# Those functions are extracted from class_utils module to share the code
# with FOBS and to avoid circular imports
def get_class_name(cls: Type) -> str:
    """Get canonical class path or fully qualified name. The builtins module is removed
    so common builtin class can be referenced with its normal name

        Args:
            cls: The class type
        Returns:
            The canonical name
    """
    module = cls.__module__
    if module == "builtins":
        return cls.__qualname__
    return module + "." + cls.__qualname__


def load_class(class_path):
    """Load class from fully qualified class name

    Args:
        class_path: fully qualified class name
    Returns:
        The class type
    """

    try:
        if "." in class_path:
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        else:
            return getattr(builtins, class_path)
    except Exception as ex:
        raise TypeError(f"Can't load class {class_path}: {ex}")
