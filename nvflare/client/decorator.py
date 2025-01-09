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
from inspect import signature

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.fuel.utils.deprecated import deprecated

from .api import is_train, receive, send


def _replace_func_args(func, kwargs, model: FLModel):
    # Replace only the first argument
    first_params = next(iter(signature(func).parameters.values()))
    kwargs[first_params.name] = model


class ObjectHolder:
    def __init__(self):
        self.metrics = None


object_holder = ObjectHolder()


@deprecated("@flare.train is deprecated and will be removed in a future version." "Use flare send/receive instead.")
def train(
    _func=None,
    **root_kwargs,
):
    """A decorator to wraps the training logic.

    Note:
        FLARE will pass the model received from the server side to the first argument of the decorated method.
        The return value of the decorated training method needs to be an FLModel object.

    Usage:

        .. code-block:: python

            @nvflare.client.train
            def my_train(input_model=None, device="cuda:0"):
               ...
               return new_model

    """

    def decorator(train_fn):
        @functools.wraps(train_fn)
        def wrapper(*args, **kwargs):
            input_model = receive()
            # Replace func arguments
            _replace_func_args(train_fn, kwargs, input_model)
            return_value = train_fn(**kwargs)

            if return_value is None:
                raise RuntimeError("return value is None!")
            elif not isinstance(return_value, FLModel):
                raise RuntimeError("return value needs to be an FLModel.")

            global object_holder

            if object_holder.metrics is not None:
                return_value.metrics = object_holder.metrics
                object_holder = ObjectHolder()

            send(model=return_value)

            return return_value

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


@deprecated("@flare.evaluate is deprecated and will be removed in a future version." "Use flare send/receive instead.")
def evaluate(
    _func=None,
    **root_kwargs,
):
    """A decorator to wraps the evaluate logic.

    Note:
        FLARE will pass the model received from the server side to the first argument of the decorated method.
        The return value of the decorated method needs to be a float number metric.
        The decorated method needs to be run BEFORE the training method,
        so the metrics will be sent along with the trained output model.

    Usage:

        .. code-block:: python

            @nvflare.client.evaluate
            def my_eval(input_model, device="cuda:0"):
               ...
               return metrics

    """

    def decorator(eval_fn):
        @functools.wraps(eval_fn)
        def wrapper(*args, **kwargs):
            input_model = receive()

            _replace_func_args(eval_fn, kwargs, input_model)
            return_value = eval_fn(**kwargs)

            if return_value is None:
                raise RuntimeError("return value is None!")
            global object_holder

            if is_train():
                object_holder.metrics = return_value
            else:
                send(model=FLModel(metrics=return_value))

            return return_value

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
