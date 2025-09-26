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

_ATTR_COLLAB = "_is_fox_collab"
_ATTR_SUPPORT_CTX = "_supports_fox_ctx"


def collab(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    signature = inspect.signature(func)
    parameter_names = signature.parameters.keys()
    if CollabMethodArgName.CONTEXT in parameter_names:
        setattr(wrapper, _ATTR_SUPPORT_CTX, True)
    setattr(wrapper, _ATTR_COLLAB, True)
    return wrapper


def is_collab(func):
    return hasattr(func, _ATTR_COLLAB)


def supports_context(func):
    return hasattr(func, _ATTR_SUPPORT_CTX)


def adjust_kwargs(func, kwargs):
    if not supports_context(func):
        kwargs.pop(CollabMethodArgName.CONTEXT, None)
