# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional, Union

CommandArg = Union[str, list[str]]


def normalize_argv(value: Optional[CommandArg], arg_name: str, allow_none: bool = False) -> Optional[CommandArg]:
    """Validate an argument string or pre-tokenized argv and copy mutable input."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{arg_name} must be a string or list of strings, but got NoneType")
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError(f"{arg_name} must be a string or list of strings, but got {type(value).__name__}")
    if not all(isinstance(arg, str) for arg in value):
        raise ValueError(f"{arg_name} argv must contain only strings")
    return list(value)
