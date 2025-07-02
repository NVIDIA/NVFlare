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


from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from app.core.config import settings
from app.utils.dependencies import validate_user
from app.utils.path_security import (
    secure_path_join,
    validate_directory_exists,
    validate_file_exists,
    validate_path_component,
    validate_timestamp_format,
    validate_path_within_root,
)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse


def json_streamer(file_path: str, chunk_size: int = 1024) -> Generator[str, None, None]:
    try:
        # validate file path exists and is secure
        validate_file_exists(Path(file_path), "Stats file")
        validate_path_within_root(Path(file_path), settings.data_root)
        
        with open(file_path, "r") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_latest_stats_dir(app_name: str) -> str:
    # Validate app_name to prevent path traversal
    validate_path_component(app_name, "app_name")
    
    # Use secure path joining
    app_dir = secure_path_join(settings.data_root, app_name)
    validate_directory_exists(app_dir, "Application directory")
    
    # Get the list of only immediate subdirectories
    # Use secure path joining to validate the path is within the allowed directory
    subdirectories = [
        name.name for name in list(app_dir.iterdir()) 
        if secure_path_join(app_dir, name).is_dir() 
        and validate_path_within_root(secure_path_join(app_dir, name), settings.data_root)
    ]

    timestamps = [
        datetime.strptime(directory, settings.timestamp_dir_format)
        for directory in subdirectories
    ]
    latest_timestamp = max(timestamps)
    latest_directory = latest_timestamp.strftime(settings.timestamp_dir_format)
    return latest_directory


def get_stats_json(app_name: str, timestamp: str) -> dict:
    # Validate app_name to prevent path traversal
    validate_path_component(app_name, "app_name")
    
    if not timestamp:
        timestamp = get_latest_stats_dir(app_name)
    else:
        # Validate timestamp format to prevent path traversal
        validate_timestamp_format(timestamp)

    # Use secure path joining
    app_directory = secure_path_join(settings.data_root, app_name)
    stats_directory = secure_path_join(app_directory, timestamp)
    validate_directory_exists(stats_directory, "Stats directory")

    stats_file_path = secure_path_join(stats_directory, settings.stats_file_name)
    validate_file_exists(stats_file_path, "Stats file")

    return json_streamer(str(stats_file_path))


router = APIRouter()


@router.get("/{app_name}/", response_class=StreamingResponse)
async def get_stats(
    app_name: str, timestamp: Optional[str] = None, dep: None = Depends(validate_user)
):
    """An API to get the statistics for the given application.

    Args:
        app_name: Name of the application
        timestamp: One of the available timestamps for which to return the statistics.

    Returns:
        Returns the statistics JSON for the given application and timestamp.
        If no timestamp is provided, it returns the latest statistics for the given application.
    """
    return StreamingResponse(
        get_stats_json(app_name, timestamp), media_type="application/json"
    )
