# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import functools
import os
from inspect import Parameter, signature

from nvflare.app_common.abstract.fl_model import FLModel

from .api import PROCESS_CACHE


def _get_attr_from_fl_model(model: FLModel, query_string: str):
    if hasattr(model, query_string):
        return getattr(model, query_string)
    segments = query_string.split(".")  # meta.current_round
    if len(segments) != 2:
        raise RuntimeError(f"Invalid string: {query_string}")
    elif segments[0] != "meta":
        raise RuntimeError(f"Invalid string: {query_string}")
    elif segments[1] not in model.meta:
        raise RuntimeError(f"Can't get {segments[1]} from meta.")
    return model.meta[segments[1]]


def _replace_func_args(func, kwargs, model: FLModel, key_mapping: dict):
    # Replace the first argument
    first_params = next(iter(signature(func).parameters.values()))
    kwargs[first_params.name] = model.params

    for key in key_mapping:
        # replace kwargs
        if key in kwargs:
            kwargs[key] = _get_attr_from_fl_model(model, key_mapping[key])
        # replace default kwargs
        defaults = {p.name: p.default for p in signature(func).parameters.values() if p.default is not Parameter.empty}
        if key in defaults:
            kwargs[key] = _get_attr_from_fl_model(model, key_mapping[key])


def train(
    _func=None,
    **root_kwargs,
):
    def decorator(train_fn):
        @functools.wraps(train_fn)
        def wrapper(*args, **kwargs):
            pid = os.getpid()
            if pid not in PROCESS_CACHE:
                raise RuntimeError("needs to call init method first")
            cache = PROCESS_CACHE[pid]

            key_mapping = dict()
            key_mapping.update(root_kwargs)

            # Replace func arguments
            _replace_func_args(train_fn, kwargs, cache.input_model, key_mapping)
            return_value = train_fn(**kwargs)

            if return_value is None:
                raise RuntimeError("return value is None!")
            elif isinstance(return_value, tuple) and len(return_value) == 2:
                model, meta = return_value
                if meta is not None:
                    cache.meta.update(meta)
            else:
                model = return_value

            fl_model = cache.construct_fl_model(params=model)

            cache.model_exchanger.submit_model(fl_model)
            cache.model_exchanger.finalize(close_pipe=False)
            PROCESS_CACHE.pop(pid)

            return return_value

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def evaluate(
    _func=None,
    **root_kwargs,
):
    def decorator(eval_fn):
        @functools.wraps(eval_fn)
        def wrapper(*args, **kwargs):
            pid = os.getpid()
            if pid not in PROCESS_CACHE:
                raise RuntimeError("needs to call init method first")
            cache = PROCESS_CACHE[pid]

            key_mapping = dict()
            key_mapping.update(root_kwargs)

            _replace_func_args(eval_fn, kwargs, cache.input_model, key_mapping)
            return_value = eval_fn(**kwargs)

            if return_value is None:
                raise RuntimeError("return value is None!")
            elif isinstance(return_value, tuple) and len(return_value) == 2:
                metrics, meta = return_value
                if meta is not None:
                    cache.meta.update(meta)
            else:
                metrics = return_value
            cache.metrics = metrics

            return return_value

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
