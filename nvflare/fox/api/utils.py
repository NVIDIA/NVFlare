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
import inspect
import logging
from typing import List

from .constants import CollabMethodArgName


def check_optional_args(func, kwargs, arg_names: List[str]):
    signature = inspect.signature(func)
    parameter_names = signature.parameters.keys()

    # make sure to expose the optional args if the collab method supports them
    for n in arg_names:
        if n not in parameter_names:
            kwargs.pop(n, None)


def check_context_support(func, kwargs):
    check_optional_args(func, kwargs, [CollabMethodArgName.CONTEXT])


def get_collab_object_name(target_name: str):
    """The target_name is either the site name or <site_name>.<collab_obj_name>.
    This function gets the collab object name.

    Args:
        target_name:

    Returns:

    """
    parts = target_name.split(".")
    if len(parts) == 1:
        return "_app_"
    else:
        return parts[1]


def check_call_args(func_name, func_itf, call_args, call_kwargs: dict):
    """Check call args against the function's interface.

    Args:
        func_name:
        func_itf:
        call_args:
        call_kwargs:

    Returns:

    """
    num_call_args = len(call_args) + len(call_kwargs)
    if num_call_args > len(func_itf):
        # For security, collab funcs must only have fixed args - no flexible args are allowed.
        raise RuntimeError(
            f"there are {num_call_args} call args ({len(call_args)=} {len(call_kwargs)=}), "
            f"but function '{func_name}' only supports {len(func_itf)} args ({func_itf})"
        )

    # make sure every arg in kwargs is valid
    for arg_name in call_kwargs.keys():
        if arg_name not in func_itf:
            raise RuntimeError(f"call arg {arg_name} is not supported by func '{func_name}'")


def simple_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
