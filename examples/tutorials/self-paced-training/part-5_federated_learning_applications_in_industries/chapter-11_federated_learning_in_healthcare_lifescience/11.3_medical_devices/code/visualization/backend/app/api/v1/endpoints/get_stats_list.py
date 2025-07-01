# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import List

from app.core.config import settings
from app.utils.dependencies import validate_user
from fastapi import APIRouter, Depends, HTTPException


def get_stats_directories(app_name: str) -> List[str]:
    app_directory_path = settings.data_root + "/" + app_name
    app_dir = Path(app_directory_path)
    if not app_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Provided application directory path is not a directory",
        )

    # Get the list of only immediate subdirectories
    subdirectories = [
        name.name for name in list(app_dir.iterdir()) if (app_dir / name).is_dir()
    ]
    return subdirectories


router = APIRouter()


@router.get("/{app_name}/", response_model=List[str])
async def get_stats_list(app_name: str, dep: None = Depends(validate_user)):
    """An API to get the list of available statistics for a given application.

    This API expects different directories to be available inside the application directory.
    The list of available statistics is just the list of different directories present inside the root directory.

    Args:
        app_name: Name of the application

    Returns:
        Returns the list of available statistics.
    """
    # Validate path inside app root
    app_root = Path(settings.data_root).resolve()
    app_dir = (app_root / app_name).resolve()

    if not str(app_dir).startswith(str(app_root)):
        raise HTTPException(
            status_code=400,
            detail="Invalid application name: outside allowed directory scope",
        )

    return get_stats_directories(app_name)
