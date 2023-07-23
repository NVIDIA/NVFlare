# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ContentBlockedException(Exception):
    """
    A filter should raise this exception when the content is to be blocked
    """

    pass


class FilterChainType(object):

    TASK_DATA_CHAIN = "task_data"
    TASK_RESULT_CHAIN = "task_result"


class FilterSource(object):

    JOB = "job"
    SITE = "site"


class FilterContextKey(object):

    SOURCE = "__source"
    CHAIN_TYPE = "__chain_type"


class Filter(FLComponent, ABC):
    @abstractmethod
    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Filter process applied to the Shareable object.

        Args:
            shareable: shareable
            fl_ctx: FLContext

        Returns:
            a Shareable object

        """
        pass

    def set_prop(self, key: str, value):
        setattr(self, key, value)

    def get_prop(self, key: str, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default
