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

_FLAG_PUBLISH = "_collab_is_collab"
_FLAG_INIT = "_collab_is_init"
_FLAG_FINAL = "_collab_is_final"
_FLAG_MAIN = "_collab_is_algo"
_FLAG_CALL_FILTER = "_collab_is_call_filter"
_FLAG_IN_CALL_FILTER = "_collab_is_in_call_filter"
_FLAG_OUT_CALL_FILTER = "_collab_is_out_call_filter"
_FLAG_RESULT_FILTER = "_collab_is_result_filter"
_FLAG_IN_RESULT_FILTER = "_collab_is_in_result_filter"
_FLAG_OUT_RESULT_FILTER = "_collab_is_out_result_filter"
_FLAG_SUPPORT_CTX = "_collab_supports_ctx"
_ATTR_PARAM_NAMES = "_collab_param_names"


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


def publish(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_PUBLISH, True)
    return wrapper


def is_publish(func):
    return _has_flag(func, _FLAG_PUBLISH)


def get_object_publish_interface(obj):
    result = {}
    for name in dir(obj):
        func = getattr(obj, name)
        if callable(func) and is_publish(func):
            result[name] = get_param_names(func)
    return result


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


def call_filter(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_CALL_FILTER, True)
    return wrapper


def get_object_call_filter_funcs(obj):
    return _get_object_funcs(obj, _FLAG_CALL_FILTER, "call_filter")


def in_call_filter(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_IN_CALL_FILTER, True)
    return wrapper


def get_object_in_call_filter_funcs(obj):
    return _get_object_funcs(obj, _FLAG_IN_CALL_FILTER, "in_call_filter")


def out_call_filter(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_OUT_CALL_FILTER, True)
    return wrapper


def get_object_out_call_filter_funcs(obj):
    return _get_object_funcs(obj, _FLAG_OUT_CALL_FILTER, "out_call_filter")


def result_filter(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_RESULT_FILTER, True)
    return wrapper


def get_object_result_filter_funcs(obj):
    return _get_object_funcs(obj, _FLAG_RESULT_FILTER, "result_filter")


def in_result_filter(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_IN_RESULT_FILTER, True)
    return wrapper


def get_object_in_result_filter_funcs(obj):
    return _get_object_funcs(obj, _FLAG_IN_RESULT_FILTER, "in_result_filter")


def out_result_filter(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _set_attrs(func, wrapper)
    setattr(wrapper, _FLAG_OUT_RESULT_FILTER, True)
    return wrapper


def get_object_out_result_filter_funcs(obj):
    return _get_object_funcs(obj, _FLAG_OUT_RESULT_FILTER, "out_result_filter")


def get_param_names(func):
    return getattr(func, _ATTR_PARAM_NAMES, None)


def _has_flag(func, flag: str) -> bool:
    v = getattr(func, flag, None)
    return v is True


def _get_object_funcs(obj, flag, func_type):
    result = []
    for name in dir(obj):
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
