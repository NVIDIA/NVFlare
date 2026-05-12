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

"""Shared file and name validation helpers for distributed provisioning commands."""

import os
import re
from typing import Optional, Tuple
from collections.abc import Callable

_PROJECT_NAME_HINT = "Project names must match [A-Za-z0-9][A-Za-z0-9._-]* and must not contain path separators."
_PROJECT_NAME_MAX_LENGTH_HINT = "Project names must be 64 characters or fewer."
_SAFE_PROJECT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def safe_project_name_error(project_name: str, *, field_label: str = "Project") -> tuple[str, str] | None:
    if not isinstance(project_name, str) or not project_name.strip():
        return f"{field_label} must not be empty or whitespace only.", _PROJECT_NAME_HINT
    if len(project_name) > 64:
        return f"{field_label} must be 64 characters or fewer.", _PROJECT_NAME_MAX_LENGTH_HINT
    if os.sep in project_name or (os.altsep and os.altsep in project_name) or project_name.startswith("."):
        return f"{field_label} must not contain path separators or start with '.'.", _PROJECT_NAME_HINT
    if not _SAFE_PROJECT_NAME_PATTERN.fullmatch(project_name):
        return f"{field_label} must match [A-Za-z0-9][A-Za-z0-9._-]*.", _PROJECT_NAME_HINT
    return None


def write_file_nofollow(path: str, content: bytes, mode: int = 0o644) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            fd = -1  # ownership transferred to f
            f.write(content)
    except Exception:
        if fd != -1:
            os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def read_file_nofollow(
    path: str,
    max_size: int,
    *,
    too_large_error_factory: Callable[[str], Exception] | None = None,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        with os.fdopen(fd, "rb") as f:
            fd = -1  # ownership transferred to f
            content = f.read(max_size + 1)
    except BaseException:
        if fd != -1:
            os.close(fd)
        raise
    if len(content) > max_size:
        if too_large_error_factory:
            raise too_large_error_factory(path)
        raise ValueError(f"file exceeds maximum size: {path}")
    return content
