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

import json
from datetime import datetime
from math import sqrt
from pathlib import Path
from typing import List

from app.core.config import settings
from app.utils.dependencies import validate_user
from app.utils.path_security import (
    secure_path_join,
    validate_directory_exists,
    validate_file_exists,
    validate_path_component,
    validate_timestamp_format,
)
from fastapi import APIRouter, Depends, HTTPException


def accumulate_stats(dst_dict: dict, src_dict: dict):
    """Accumulates the global and local statistics of a source dictionary into a destination dictionary.

    Args:
        dst_dict (dict): The dictionary to accumulate the statistics into.
        src_dict (dict): The dictionary to accumulate the statistics from.
    """
    # Define a dictionary that maps the keys to the corresponding accumulation functions
    accumulation_functions = {
        "count": lambda x, y: x + y,
        "failure_count": lambda x, y: x + y,
        "sum": lambda x, y: x + y,
        "mean": lambda x, y: (x + y) / 2,
        "min": min,
        "max": max,
        "histogram": accumulate_bins,
        "var": lambda x, y: x + y,
        "stddev": lambda x, y: sqrt(x**2 + y**2),
    }

    # Iterate over the key-value pairs of the destination dictionary
    for key, value in dst_dict.items():
        # If the key is in the source dictionary, accumulate the statistics
        if key in src_dict:
            for k, v in value.items():
                for f1, v1 in v.items():
                    v[f1] = accumulation_functions[key](v[f1], src_dict[key][k][f1])


def accumulate_bins(bins1: list, bins2: list):
    """Accumulates the values of two lists of histogram bins.

    Each bin is represented as tuple [low_value, high_value, sample_count].

    Args:
        bins1 (list): The first list of bins.
        bins2 (list): The second list of bins.

    Returns:
        list: The accumulated list of bins.
    """
    for bin1, bin2 in zip(bins1, bins2):
        # Accumulate the values of the bins
        bin1[2] += bin2[2]
    return bins1


def accumulate_dicts(dst_dict: dict, src_dict: dict) -> dict:
    """Accumulates the source dictionary into a destination dictionary.

    Args:
        dst_dict (dict): The dictionary to accumulate into.
        src_dict (dict): The dictionary to accumulate from.
    """

    for key, value in src_dict.items():
        if key == "Global" or key == "Local":
            if key not in dst_dict:
                dst_dict[key] = value
            else:
                accumulate_stats(dst_dict[key], value)
        elif isinstance(value, dict):
            if key not in dst_dict:
                dst_dict[key] = value
            else:
                accumulate_dicts(dst_dict[key], value)
        elif isinstance(value, list):
            if key not in dst_dict:
                dst_dict[key] = value
            else:
                for item1, item2 in zip(dst_dict[key], value):
                    accumulate_dicts(item1, item2)
        else:
            continue


def get_eligible_stats_directories(app_name: str, start: str, end: str) -> List[str]:
    """Get the list of timestamp directories in a given dates range.
    
    Args:
        app_name: Name of the application
        start: Start timestamp
        end: End timestamp

    Returns:
        Returns the list of timestamp directories in a given dates range.
    """
    # Validate all inputs to prevent path traversal
    validate_path_component(app_name, "app_name")
    validate_timestamp_format(start)
    validate_timestamp_format(end)
    
    # Use secure path joining
    app_dir = secure_path_join(settings.data_root, app_name)
    validate_directory_exists(app_dir, "Application directory")

    # Get the list of eligible immediate subdirectories
    start_timestamp = datetime.strptime(start, settings.timestamp_dir_format)
    end_timestamp = datetime.strptime(end, settings.timestamp_dir_format)

    # Use secure path joining to validate the path is within the allowed directory
    subdirectories = [
        name.name
        for name in list(app_dir.iterdir())
        if secure_path_join(app_dir, name).is_dir()
        and datetime.strptime(name.name, settings.timestamp_dir_format) >= start_timestamp
        and datetime.strptime(name.name, settings.timestamp_dir_format) <= end_timestamp
    ]

    return subdirectories


def get_stats_json(app_name: str, start: str, end: str):
    """Get the accumulated statistics for the given range.
    
    Args:
        app_name: Name of the application
        start: Start timestamp
        end: End timestamp

    Returns:
        Returns the accumulated statistics JSON for the given application.
    """
    stats_dirs = get_eligible_stats_directories(app_name, start, end)
    accumulated_stats = {}
    
    # Use secure path joining for base directories
    app_directory = secure_path_join(settings.data_root, app_name)
    
    for stats in stats_dirs:
        # Use secure path joining for each stats directory and file
        stats_directory = secure_path_join(app_directory, stats)
        file_path = secure_path_join(stats_directory, settings.stats_file_name)
        
        # Validate file exists and file path is secure before processing
        try:
            validate_file_exists(file_path, f"Stats file for {stats}")
            validate_path_component(file_path, "app_name")
            
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                if not isinstance(data, dict):
                    data = json.loads(data)
                accumulate_dicts(accumulated_stats, data)
        except HTTPException:
            continue  # Skip missing files
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return accumulated_stats


router = APIRouter()


@router.get("/{app_name}/{start}/{end}/")
async def get_range_stats(
    app_name: str, start: str, end: str, dep: None = Depends(validate_user)
):
    """An API to get the accumulated statistics for the given range.

    Args:
        app_name: Name of the application
        start: Start timestamp
        end: End timestamp

    Returns:
        Returns the accumulated statistics JSON for the given application.
    """
    return get_stats_json(app_name, start, end)
