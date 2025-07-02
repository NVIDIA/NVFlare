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

import os
import re
from pathlib import Path
from typing import Union

from fastapi import HTTPException


def validate_path_component(component: str, component_name: str) -> str:
    """
    Validate a path component to prevent path traversal attacks.
    
    Args:
        component: The path component to validate
        component_name: Name of the component for error messages
        
    Returns:
        The validated component
        
    Raises:
        HTTPException: If the component contains invalid characters
    """
    if not component:
        raise HTTPException(
            status_code=400, 
            detail=f"{component_name} cannot be empty"
        )
    
    # Check for path traversal attempts
    if ".." in component or "/" in component or "\\" in component:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {component_name}: path traversal characters not allowed"
        )
    
    # Check for null bytes
    if "\x00" in component:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {component_name}: null bytes not allowed"
        )
    
    # Only allow alphanumeric, hyphens, underscores, and dots (but not .. sequences)
    if not re.match(r'^[a-zA-Z0-9._-]+$', component):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {component_name}: only alphanumeric characters, dots, hyphens, and underscores allowed"
        )
    
    return component


def validate_timestamp_format(timestamp: str) -> str:
    """
    Validate timestamp format to ensure it matches expected pattern.
    
    Args:
        timestamp: The timestamp string to validate
        
    Returns:
        The validated timestamp
        
    Raises:
        HTTPException: If the timestamp format is invalid
    """
    # Expected format: YYYYMMDD_HHMMSS
    timestamp_pattern = r'^\d{8}_\d{6}$'
    
    if not re.match(timestamp_pattern, timestamp):
        raise HTTPException(
            status_code=400,
            detail="Invalid timestamp format. Expected format: YYYYMMDD_HHMMSS"
        )
    
    return timestamp


def secure_path_join(base_path: Union[str, Path], *components: str) -> Path:
    """
    Securely join path components and ensure the result is within the base path.
    
    Args:
        base_path: The base directory path
        *components: Path components to join
        
    Returns:
        The resolved secure path
        
    Raises:
        HTTPException: If the resulting path would be outside the base path
    """
    base = Path(base_path).resolve()
    
    # Join all components
    result_path = base
    for component in components:
        result_path = result_path / component
    
    # Resolve the final path
    resolved_path = result_path.resolve()
    
    # Check if the resolved path is within the base path
    try:
        resolved_path.relative_to(base)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid path: access outside base directory not allowed"
        )
    
    return resolved_path


def validate_file_exists(file_path: Path, file_description: str = "File") -> Path:
    """
    Validate that a file exists and is a regular file.
    
    Args:
        file_path: The file path to validate
        file_description: Description of the file for error messages
        
    Returns:
        The validated file path
        
    Raises:
        HTTPException: If the file doesn't exist or is not a regular file
    """
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{file_description} not found"
        )
    
    if not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"{file_description} is not a valid file"
        )
    
    return file_path


def validate_directory_exists(dir_path: Path, dir_description: str = "Directory") -> Path:
    """
    Validate that a directory exists and is a directory.
    
    Args:
        dir_path: The directory path to validate
        dir_description: Description of the directory for error messages
        
    Returns:
        The validated directory path
        
    Raises:
        HTTPException: If the directory doesn't exist or is not a directory
    """
    if not dir_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{dir_description} not found"
        )
    
    if not dir_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"{dir_description} is not a valid directory"
        )
    
    return dir_path


def validate_path_within_root(file_path: Union[str, Path], app_root: Union[str, Path], raise_exception: bool = True) -> bool:
    """
    Validate that a file path is within the app_root directory (subdirectory check).
    
    This function prevents path traversal attacks by ensuring that the resolved file_path
    is within the boundaries of the app_root directory.
    
    Args:
        file_path: The file/directory path to validate
        app_root: The root directory that should contain the file_path
        raise_exception: Whether to raise HTTPException on validation failure (default: True)
        
    Returns:
        bool: True if the path is within app_root, False otherwise (only if raise_exception=False)
        
    Raises:
        HTTPException: If the path is outside app_root and raise_exception=True
        
    Examples:
        >>> validate_path_within_root("/app/data/user_file.txt", "/app/data")  # ✅ Valid
        >>> validate_path_within_root("/app/data/../../../etc/passwd", "/app/data")  # ❌ Invalid
        >>> validate_path_within_root("/app/data/subdir/../file.txt", "/app/data")  # ✅ Valid (resolves to /app/data/file.txt)
    """
    try:
        # Convert to Path objects and resolve to handle symbolic links and relative paths
        resolved_file_path = Path(file_path).resolve()
        resolved_app_root = Path(app_root).resolve()
        
        # Check if the file path is within the app root
        # This will raise ValueError if file_path is not relative to app_root
        resolved_file_path.relative_to(resolved_app_root)
        
        return True
        
    except ValueError:
        # Path is outside the app_root directory
        if raise_exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid path: '{file_path}' is outside the allowed directory '{app_root}'"
            )
        return False
    except Exception as e:
        # Handle other potential errors (permission issues, non-existent paths, etc.)
        if raise_exception:
            raise HTTPException(
                status_code=400,
                detail=f"Path validation error: {str(e)}"
            )
        return False