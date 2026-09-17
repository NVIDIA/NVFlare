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
from .publish_interface import PublishInterface

_FLAG_PUBLISH = "_collab_is_publish"
_FLAG_INIT = "_collab_is_init"
_FLAG_FINAL = "_collab_is_final"
_FLAG_MAIN = "_collab_is_main"
_FLAG_SUPPORT_CTX = "_collab_supports_ctx"
_ATTR_PARAM_NAMES = "_collab_param_names"
_ATTR_PARAM_SPECS = "_collab_param_specs"


class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_instance, owner_class):
        return self.fget(owner_class)


def _set_attrs(func, wrapper, require_fixed_args=False):
    signature = inspect.signature(func)
    parameters = [p for p in signature.parameters.values() if p.name != "self"]
    if require_fixed_args:
        flexible = [
            p.name for p in parameters if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if flexible:
            raise TypeError(f"@collab.publish does not support flexible parameters {flexible}")
    parameter_names = [p.name for p in parameters]
    parameter_specs = [
        {
            "name": p.name,
            "kind": p.kind.name,
            "required": (p.default is inspect.Parameter.empty and p.name != CollabMethodArgName.CONTEXT),
        }
        for p in parameters
    ]
    setattr(wrapper, _ATTR_PARAM_NAMES, parameter_names)
    setattr(wrapper, _ATTR_PARAM_SPECS, parameter_specs)
    if CollabMethodArgName.CONTEXT in parameter_names:
        setattr(wrapper, _FLAG_SUPPORT_CTX, True)


def publish(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper, require_fixed_args=True)
    setattr(wrapper, _FLAG_PUBLISH, True)
    return wrapper


def is_publish(func):
    return _has_flag(func, _FLAG_PUBLISH)


def get_object_publish_interface(obj) -> PublishInterface:
    result = {}
    for name in dir(obj):
        func = getattr(obj, name)
        if callable(func) and is_publish(func):
            result[name] = get_param_specs(func)
    return PublishInterface(result)


def init(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_INIT, True)
    return wrapper


def get_object_init_funcs(obj):
    return _get_object_funcs(obj, _FLAG_INIT, "init")


def final(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_FINAL, True)
    return wrapper


def get_object_final_funcs(obj):
    return _get_object_funcs(obj, _FLAG_FINAL, "final")


def main(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_MAIN, True)
    return wrapper


def get_object_main_funcs(obj):
    return _get_object_funcs(obj, _FLAG_MAIN, "main")


def get_param_names(func):
    return getattr(func, _ATTR_PARAM_NAMES, None)


def get_param_specs(func):
    return getattr(func, _ATTR_PARAM_SPECS, None)


def _has_flag(func, flag: str) -> bool:
    v = getattr(func, flag, None)
    return v is True


def _get_object_funcs(obj, flag, func_type):
    result = []
    for name in sorted(dir(obj)):
        func = getattr(obj, name)
        if callable(func) and _has_flag(func, flag):
            # print(f"found {func_type} func of object {obj.__class__.__name__}.{name}")
            result.append((name, func))
    return result


def supports_context(func):
    return _has_flag(func, _FLAG_SUPPORT_CTX)


def adjust_kwargs(func, kwargs):
    """Adjust the kwargs and remove keys that are not supported by the func.

    Args:
        func: the func to be checked
        kwargs: the kwargs to be adjusted

    Returns: the adjusted kwargs

    """
    if not supports_context(func):
        kwargs.pop(CollabMethodArgName.CONTEXT, None)
    return kwargs
