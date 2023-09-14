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
from inspect import signature

from nvflare.app_common.abstract.fl_model import FLModel

from .api import PROCESS_MODEL_REGISTRY


def _replace_func_args(func, kwargs, model: FLModel):
    # Replace only the first argument
    first_params = next(iter(signature(func).parameters.values()))
    kwargs[first_params.name] = model


def train(
    _func=None,
    **root_kwargs,
):
    def decorator(train_fn):
        @functools.wraps(train_fn)
        def wrapper(*args, **kwargs):
            pid = os.getpid()
            if pid not in PROCESS_MODEL_REGISTRY:
                raise RuntimeError("needs to call init method first")
            cache = PROCESS_MODEL_REGISTRY[pid]
            input_model = cache.get_model()

            # Replace func arguments
            _replace_func_args(train_fn, kwargs, input_model)
            return_value = train_fn(**kwargs)

            if return_value is None:
                raise RuntimeError("return value is None!")
            elif not isinstance(return_value, FLModel):
                raise RuntimeError("return value needs to be an FLModel.")

            if cache.metrics is not None:
                return_value.metrics = cache.metrics

            cache.send(model=return_value)
            cache.model_exchanger.finalize(close_pipe=False)
            PROCESS_MODEL_REGISTRY.pop(pid)

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
            if pid not in PROCESS_MODEL_REGISTRY:
                raise RuntimeError("needs to call init method first")
            cache = PROCESS_MODEL_REGISTRY[pid]
            input_model = cache.get_model()

            _replace_func_args(eval_fn, kwargs, input_model)
            return_value = eval_fn(**kwargs)

            if return_value is None:
                raise RuntimeError("return value is None!")

            cache.metrics = return_value

            return return_value

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
