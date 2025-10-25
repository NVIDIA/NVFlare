# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from app.core.config import settings
from app.utils.dependencies import validate_user
from fastapi import APIRouter, Depends


def get_stats_directories(app_name: str) -> list[str]:
    # Validate app_name to prevent path traversal
    app_dir = os.path.normpath(os.path.join(settings.data_root, app_name))
    if not app_dir.startswith(settings.data_root):
        raise Exception(f"Invalid app directory: {app_dir}, not allowed.")

    if not os.path.isdir(app_dir):
        raise Exception(f"Application directory: {app_dir}, not found.")

    # Get the list of immediate & validated subdirectories
    subdirectories = []
    for name in os.listdir(app_dir):
        sub_path = os.path.normpath(os.path.join(app_dir, name))
        if os.path.isdir(sub_path) and sub_path.startswith(settings.data_root):
            subdirectories.append(name)

    return subdirectories


router = APIRouter()


@router.get("/{app_name}/", response_model=list[str])
async def get_stats_list(app_name: str, dep: None = Depends(validate_user)):
    """An API to get the list of available statistics for a given application.

    This API expects different directories to be available inside the application directory.
    The list of available statistics is just the list of different directories present inside the root directory.

    Args:
        app_name: Name of the application

    Returns:
        Returns the list of available statistics.
    """
    return get_stats_directories(app_name)
