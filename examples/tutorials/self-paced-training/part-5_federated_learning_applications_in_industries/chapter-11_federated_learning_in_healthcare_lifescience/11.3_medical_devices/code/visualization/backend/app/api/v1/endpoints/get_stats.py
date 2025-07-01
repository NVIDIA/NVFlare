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
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse


def json_streamer(file_path: str, chunk_size: int = 1024) -> Generator[str, None, None]:
    try:
        with open(file_path, "r") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_latest_stats_dir(app_name: str) -> str:
    app_root = Path(settings.data_root).resolve()
    app_dir = (app_root / app_name).resolve()
    
    # Validate the path is within the allowed directory
    try:
        app_dir.relative_to(app_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid app directory")
    
    if not app_dir.is_dir():
        raise HTTPException(status_code=400, detail="Invalid app directory")
    
    # Get the list of only immediate subdirectories
    subdirectories = [
        name.name for name in list(app_dir.iterdir()) if (app_dir / name).is_dir()
    ]

    timestamps = [
        datetime.strptime(directory, settings.timestamp_dir_format)
        for directory in subdirectories
    ]
    latest_timestamp = max(timestamps)
    latest_directory = latest_timestamp.strftime(settings.timestamp_dir_format)
    return latest_directory


def get_stats_json(app_name: str, timestamp: str) -> dict:
    if not app_name:
        raise HTTPException(status_code=400, detail="Application name not provided")
    if not timestamp:
        timestamp = get_latest_stats_dir(app_name)

    app_root = Path(settings.data_root).resolve()
    app_dir = (app_root / app_name).resolve()
    stats_dir = (app_dir / timestamp).resolve()
    stats_file_path = (stats_dir / settings.stats_file_name).resolve()
    
    # Validate all paths are within the allowed directory
    try:
        app_dir.relative_to(app_root)
        stats_dir.relative_to(app_root)
        stats_file_path.relative_to(app_root)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid path: outside allowed directory scope",
        )
    
    if not stats_dir.is_dir():
        raise HTTPException(
            status_code=400, detail="Provided stats directory path is not a directory"
        )

    if not stats_file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Stats not available for the give application and timestamp",
        )
    else:
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
