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

import os


def validate_flower_app_path(path: str) -> None:
    """Validate flower_app_path format."""
    if not path.startswith("local/custom/"):
        raise ValueError(
            f"flower_app_path must start with 'local/custom/', got '{path}'. "
            "Pre-deployed apps must be in the workspace's local/custom directory."
        )

    # Check for path traversal attempts with either Unix or Windows separators.
    normalized = path.replace("\\", "/")
    if ".." in normalized:
        raise ValueError(f"flower_app_path contains invalid path traversal: '{path}'")


def validate_flower_app_path_no_symlinks(app_dir: str) -> None:
    """Validate that the resolved flower app directory is not a symbolic link."""
    if os.path.islink(app_dir):
        raise RuntimeError(
            "flower_app_path resolves to a symbolic link. "
            "For security, pre-deployed app paths must be real directories, not links."
        )
