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
from typing import List

from .constants import CollabMethodArgName


def check_optional_args(func, kwargs, arg_names: List[str]):
    signature = inspect.signature(func)
    parameter_names = signature.parameters.keys()

    # make sure to expose the optional args if the collab method supports them
    for n in arg_names:
        if n not in parameter_names:
            kwargs.pop(n, None)


def check_context_support(func, kwargs):
    check_optional_args(func, kwargs, [CollabMethodArgName.CONTEXT])
