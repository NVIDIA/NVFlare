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

_FLAG_COLLAB = "_fox_is_collab"
_FLAG_INIT = "_fox_is_init"
_FLAG_ALGO = "_fox_is_algo"
_FLAG_SUPPORT_CTX = "_fox_supports_ctx"
_ATTR_PARAM_NAMES = "_fox_param_names"


class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_instance, owner_class):
        return self.fget(owner_class)


def _set_attrs(func, wrapper):
    signature = inspect.signature(func)
    parameter_names = list(signature.parameters.keys())
    if "self" in parameter_names:
        parameter_names.remove("self")
    setattr(wrapper, _ATTR_PARAM_NAMES, parameter_names)
    if CollabMethodArgName.CONTEXT in parameter_names:
        setattr(wrapper, _FLAG_SUPPORT_CTX, True)


def collab(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_COLLAB, True)
    return wrapper


def init(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_INIT, True)
    return wrapper


def algo(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_ALGO, True)
    return wrapper


def get_param_names(func):
    return getattr(func, _ATTR_PARAM_NAMES, None)


def _has_flag(func, flag: str) -> bool:
    v = getattr(func, flag, None)
    return v is True


def is_collab(func):
    return _has_flag(func, _FLAG_COLLAB)


def is_init(func):
    return _has_flag(func, _FLAG_INIT)


def is_algo(func):
    return _has_flag(func, _FLAG_ALGO)


def supports_context(func):
    return _has_flag(func, _FLAG_SUPPORT_CTX)


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


def get_object_init_funcs(obj):
    result = []
    for name in dir(obj):
        func = getattr(obj, name)
        if callable(func) and is_init(func):
            print(f"found init func of object {obj.__class__.__name__}.{name}")
            result.append(func)
    return result


def get_object_algo_funcs(obj):
    result = []
    for name in dir(obj):
        func = getattr(obj, name)
        if callable(func) and is_algo(func):
            result.append((name, func))
    return result
