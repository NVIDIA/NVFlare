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

import os
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


def resolve_path_under_root(
    root_dir: str, relative_path: str, path_name: str = "path", preserve_leaf: bool = False
) -> str:
    """Resolve a relative path beneath a root, optionally retaining its final symlink name."""
    if not isinstance(root_dir, str):
        raise TypeError(f"root_dir must be str but got {type(root_dir)}")
    if not isinstance(relative_path, str):
        raise TypeError(f"{path_name} must be str but got {type(relative_path)}")
    if os.path.isabs(relative_path):
        raise ValueError(f"{path_name} {relative_path} must be relative to {root_dir}.")

    root_real = os.path.realpath(root_dir)
    candidate_path = os.path.join(root_real, relative_path)
    full_path = os.path.realpath(candidate_path)
    if os.path.commonpath([root_real, full_path]) != root_real:
        raise ValueError(f"{path_name} {relative_path} must stay inside {root_real}.")
    if preserve_leaf:
        parent_real = os.path.realpath(os.path.dirname(candidate_path))
        if os.path.commonpath([root_real, parent_real]) != root_real:
            raise ValueError(f"{path_name} {relative_path} must stay inside {root_real}.")
        return os.path.join(parent_real, os.path.basename(candidate_path))
    return full_path
