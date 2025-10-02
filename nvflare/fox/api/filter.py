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

from .ctx import Context


class CallFilter:

    def filter_call(self, func_kwargs: dict, context: Context):
        """Filter kwargs of function call.

        Args:
            func_kwargs: kwargs to be filtered
            context: call context

        Returns: filtered kwargs that will be passed to a collab func.

        """
        return func_kwargs


class ResultFilter:

    def filter_result(self, result: Any, context: Context):
        """Filter result produced by a collab func.

        Args:
            result: data to be filtered
            context: call context

        Returns: filtered result

        """
        return result


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
