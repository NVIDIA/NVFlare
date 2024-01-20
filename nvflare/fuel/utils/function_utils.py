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
import importlib
import inspect
from typing import Callable, Tuple


def find_task_fn(task_fn_path: str) -> Callable:
    """
    Find and return a callable task function based on its module path.

    Args:
        task_fn_path (str): The path to the task function in the format "module_path.function_name".

    Returns:
        Callable: The callable task function.
    """
    # Split the text by the last dot
    tokens = task_fn_path.rsplit(".", 1)
    module_name = tokens[0]
    fn_name = tokens[1] if len(tokens) > 1 else ""
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


def require_arguments(func: Callable) -> Tuple[bool, int, int]:
    """
    Check if a function requires arguments and provide information about its signature.

    Args:
        func (Callable): The function to be checked.

    Returns:
        Tuple[bool, int, int]: A tuple containing three elements:
            1. A boolean indicating whether the function requires any arguments.
            2. The total number of parameters in the function's signature.
            3. The number of parameters with default values (i.e., optional parameters).
    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    req = any(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters.values())
    size = len(parameters)
    args_with_defaults = [param for param in parameters.values() if param.default != inspect.Parameter.empty]
    default_args_size = len(args_with_defaults)
    return req, size, default_args_size
