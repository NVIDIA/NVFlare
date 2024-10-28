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

import pathlib
from typing import Optional


def get_ext_format(ext: str) -> str:
    if ext is None or ext == "" or ext.isspace():
        return "csv"
    elif ext.startswith("."):
        return ext[1:]
    else:
        return ext


def get_file_format(input_path: str) -> str:
    ext = get_file_ext(input_path)
    return get_ext_format(ext)


def get_file_ext(input_path: str) -> Optional[str]:
    ext = pathlib.Path(input_path).suffix
    if ext.startswith("."):
        return ext[1:]
    else:
        return ext
