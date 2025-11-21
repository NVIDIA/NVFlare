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
from typing import Any

from nvflare.fuel.utils.log_utils import get_obj_logger

from .constants import CollabMethodArgName
from .ctx import Context
from .dec import (
    get_object_call_filter_funcs,
    get_object_in_call_filter_funcs,
    get_object_in_result_filter_funcs,
    get_object_out_call_filter_funcs,
    get_object_out_result_filter_funcs,
    get_object_result_filter_funcs,
    supports_context,
)


class _Filter:

    def __init__(self, filter_type: str, impl: object = None, incoming=True):
        self.filter_type = filter_type
        self.impl = impl
        self.incoming = incoming
        self.impl_func = None
        self.logger = get_obj_logger(self)

    def get_impl_object(self):
        if self.impl:
            return self.impl
        else:
            return self

    def filter_data(self, data, context: Context):
        if self.impl_func is not None:
            if self.incoming:
                d = "incoming"
            else:
                d = "outgoing"

            name, f = self.impl_func
            self.logger.info(f"calling {d} {self.filter_type}: {name} on ctx {id(context)}")
            if supports_context(f):
                kwargs = {CollabMethodArgName.CONTEXT: context}
            else:
                kwargs = {}
            return f(data, **kwargs)
        else:
            return data


def _determine_filter_impl_func(
    obj,
    filter_type: str,
    incoming: bool,
    get_filter_f,
    get_in_filter_f,
    get_out_filter_f,
):
    if incoming:
        funcs = get_in_filter_f(obj)
        d = "in"
    else:
        funcs = get_out_filter_f(obj)
        d = "out"

    if len(funcs) > 1:
        raise ValueError(
            f"filter object {obj.__class__.__name__} must have one {d}_{filter_type} func but got {len(funcs)}"
        )

    if len(funcs) == 1:
        return funcs[0]

    funcs = get_filter_f(obj)
    if not funcs:
        raise ValueError(f"filter impl object {obj.__class__.__name__} has no {filter_type} func")

    if len(funcs) > 1:
        raise ValueError(
            f"filter object {obj.__class__.__name__} must have one {filter_type} func but got {len(funcs)}"
        )
    return funcs[0]


class CallFilter(_Filter):

    def __init__(self, impl: object = None, incoming=True):
        super().__init__("call filter", impl, incoming)
        if impl:
            self.impl_func = _determine_filter_impl_func(
                obj=impl,
                incoming=incoming,
                filter_type="call_filter",
                get_filter_f=get_object_call_filter_funcs,
                get_in_filter_f=get_object_in_call_filter_funcs,
                get_out_filter_f=get_object_out_call_filter_funcs,
            )

    def filter_call(self, func_kwargs: dict, context: Context):
        """Filter kwargs of function call.

        Args:
            func_kwargs: kwargs to be filtered
            context: call context

        Returns: filtered kwargs that will be passed to a collab func.

        """
        return self.filter_data(func_kwargs, context)


class ResultFilter(_Filter):

    def __init__(self, impl: object = None, incoming=True):
        super().__init__("result filter", impl, incoming)
        if impl:
            self.impl_func = _determine_filter_impl_func(
                obj=impl,
                filter_type="result_filter",
                incoming=incoming,
                get_filter_f=get_object_result_filter_funcs,
                get_in_filter_f=get_object_in_result_filter_funcs,
                get_out_filter_f=get_object_out_result_filter_funcs,
            )

    def filter_result(self, result: Any, context: Context):
        """Filter result produced by a collab func.

        Args:
            result: data to be filtered
            context: call context

        Returns: filtered result

        """
        return self.filter_data(result, context)


class FilterChain:

    def __init__(self, pattern, filter_type):
        if filter_type not in [ResultFilter, CallFilter]:
            raise ValueError(
                f"filter_type must be type of {ResultFilter.__name__} or {CallFilter.__name__} but got {filter_type}"
            )
        self.pattern = pattern
        self.filter_type = filter_type
        self.filters = []

    def add_filters(self, filters):
        if not filters:
            return

        if isinstance(filters, list):
            if not all(isinstance(item, self.filter_type) for item in filters):
                raise ValueError(f"some items in filters are not {self.filter_type}")
            self.filters.extend(filters)
        else:
            if not isinstance(filters, self.filter_type):
                raise ValueError(f"filter item must be {self.filter_type} but got {type(filters)}")
            self.filters.append(filters)

    def apply_filters(self, data, context: Context):
        for f in self.filters:
            if isinstance(f, ResultFilter):
                data = f.filter_result(data, context)
            else:
                data = f.filter_call(data, context)
        return data
