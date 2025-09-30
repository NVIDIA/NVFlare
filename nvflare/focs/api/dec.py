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

from .constants import CollabMethodArgName

_ATTR_COLLAB = "_fox_is_collab"
_ATTR_SUPPORT_CTX = "_fox_supports_ctx"
_ATTR_PARAM_NAMES = "_fox_param_names"


def collab(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    signature = inspect.signature(func)
    parameter_names = list(signature.parameters.keys())
    if "self" in parameter_names:
        parameter_names.remove("self")
    setattr(wrapper, _ATTR_PARAM_NAMES, parameter_names)
    if CollabMethodArgName.CONTEXT in parameter_names:
        setattr(wrapper, _ATTR_SUPPORT_CTX, True)
    setattr(wrapper, _ATTR_COLLAB, True)
    return wrapper


def get_param_names(func):
    return getattr(func, _ATTR_PARAM_NAMES, None)


def is_collab(func):
    return hasattr(func, _ATTR_COLLAB)


def supports_context(func):
    return hasattr(func, _ATTR_SUPPORT_CTX)


def adjust_kwargs(func, kwargs):
    if not supports_context(func):
        kwargs.pop(CollabMethodArgName.CONTEXT, None)


def get_object_collab_interface(obj):
    result = {}
    for name in dir(obj):
        func = getattr(obj, name)
        if callable(func) and is_collab(func):
            result[name] = get_param_names(func)
    return result
