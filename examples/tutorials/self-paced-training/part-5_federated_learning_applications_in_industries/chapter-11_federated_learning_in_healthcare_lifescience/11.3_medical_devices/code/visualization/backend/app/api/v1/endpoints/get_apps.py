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

from pathlib import Path

from app.core.config import settings
from app.utils.dependencies import validate_user
from fastapi import APIRouter, Depends, HTTPException


def get_subdirectories() -> list[str]:
    directory_path = settings.data_root
    root_dir = Path(directory_path)
    if not root_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Provided data directory root path is not a directory",
        )

    # Get the list of only immediate subdirectories
    subdirectories = [name.name for name in list(root_dir.iterdir()) if (Path(directory_path) / name).is_dir()]
    return subdirectories


router = APIRouter()


@router.get("/", response_model=list[str])
async def get_apps(dep: None = Depends(validate_user)):
    """An API to get the list of registered applications for Holoscan Federated Analytics.

    This API expects all the output from different applications to be inside the same root directory.
    It expects a separate directory with the name same as the application inside the root directory.
    The list of registered applications is just the list of directories present inside the root directory.

    Args:
        app_name: Name of the application
        timestamp: One of the available timestamps for which to return the statistics.

    Returns:
        Returns the list of registered applications for Holoscan Federated Analytics.
    """
    return get_subdirectories()
