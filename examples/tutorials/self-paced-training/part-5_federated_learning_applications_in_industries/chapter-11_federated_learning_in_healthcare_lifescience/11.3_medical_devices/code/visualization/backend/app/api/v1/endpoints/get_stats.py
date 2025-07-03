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
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from app.core.config import settings
from app.utils.dependencies import validate_user
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse


def json_streamer(file_path: str, chunk_size: int = 1024) -> Generator[str, None, None]:
    try:
        file_path = os.path.normpath(file_path)
        if not file_path.startswith(settings.data_root):
            raise Exception(f"Invalid file path: {file_path}, not allowed.")

        with open(file_path, "r") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_latest_stats_dir(app_name: str) -> str:

    # Use secure path joining
    app_dir = os.path.join(settings.data_root, app_name)

    app_dir = os.path.normpath(app_dir)
    if not app_dir.startswith(settings.data_root):
        raise Exception(f"Invalid app directory: {app_dir}, not allowed.")

    if not os.path.isdir(app_dir):
        raise Exception(f"Application directory: {app_dir}, not found.")

    # Get the list of only immediate subdirectories
    subdirectories = []
    for name in os.listdir(app_dir):
        sub_path = os.path.normpath(os.path.join(app_dir, name))
        if os.path.isdir(sub_path) and sub_path.startswith(settings.data_root):
            subdirectories.append(name)

    timestamps = [datetime.strptime(directory, settings.timestamp_dir_format) for directory in subdirectories]
    latest_timestamp = max(timestamps)
    latest_directory = latest_timestamp.strftime(settings.timestamp_dir_format)
    return latest_directory


def get_stats_json(app_name: str, timestamp: str) -> dict:
    if not app_name:
        raise HTTPException(status_code=400, detail="Application name not provided")

    # Validate app_name to prevent path traversal
    app_dir = os.path.join(settings.data_root, app_name)
    app_dir = os.path.normpath(app_dir)
    if not app_dir.startswith(settings.data_root):
        raise Exception(f"Invalid app directory: {app_dir}, not allowed.")

    if not os.path.isdir(app_dir):
        raise Exception(f"Application directory: {app_dir}, not found.")

    if not timestamp:
        timestamp = get_latest_stats_dir(app_name)

    # Validate timestamp
    stats_directory = os.path.normpath(os.path.join(app_dir, timestamp))
    if not stats_directory.startswith(settings.data_root):
        raise Exception(f"Invalid stats directory: {stats_directory}, not allowed.")
    if not os.path.isdir(stats_directory):
        raise Exception(f"Stats directory: {stats_directory}, not found.")

    stats_file_path = os.path.normpath(os.path.join(stats_directory, settings.stats_file_name))
    if not Path(stats_file_path).is_file():
        raise HTTPException(
            status_code=400,
            detail="Stats not available for the give application and timestamp",
        )
    else:
        return json_streamer(stats_file_path)


router = APIRouter()


@router.get("/{app_name}/", response_class=StreamingResponse)
async def get_stats(app_name: str, timestamp: Optional[str] = None, dep: None = Depends(validate_user)):
    """An API to get the statistics for the given application.

    Args:
        app_name: Name of the application
        timestamp: One of the available timestamps for which to return the statistics.

    Returns:
        Returns the statistics JSON for the given application and timestamp.
        If no timestamp is provided, it returns the latest statistics for the given application.
    """
    return StreamingResponse(get_stats_json(app_name, timestamp), media_type="application/json")
