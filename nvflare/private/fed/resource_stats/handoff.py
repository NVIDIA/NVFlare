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

"""Publish and read the private terminal handoff for one launch.

Measurements are self-reported. The parent supplies public identity and checks
launch generation to exclude stale attempts. Directory-relative operations and
bounded regular-file reads prevent job-owned paths from redirecting parent I/O.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from .measurements import canonical_json_bytes

MAX_TERMINAL_HANDOFF_BYTES = 1024 * 1024

RESOURCE_STATS_DIR = "resource_stats"
STAGING_DIR = "staging"
TERMINAL_HANDOFF_FILE = "terminal_handoff.json"
LAUNCH_GENERATION_OPTION = "resource_launch_generation"
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
_SECURE_HANDOFF_DIR_FD = (
    hasattr(os, "O_NOFOLLOW")
    and bool(getattr(os, "O_DIRECTORY", 0))
    and os.open in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.rmdir in os.supports_dir_fd
)
_SECURE_HANDOFF_WRITE_DIR_FD = (
    _SECURE_HANDOFF_DIR_FD and os.mkdir in os.supports_dir_fd and os.rename in os.supports_dir_fd
)


class InvalidTerminalHandoff(ValueError):
    """Raised when the child-to-parent handoff is malformed or oversized."""


def launch_generation_from_options(options: list[str], *, option_name: str = LAUNCH_GENERATION_OPTION) -> str:
    """Capture the parent-owned token before generic option parsing changes its type."""

    prefix = f"{option_name}="
    for option in reversed(options):
        if option.startswith(prefix):
            return option[len(prefix) :]
    return ""


def _handoff_filename(generation: str | None) -> str:
    if generation is None:
        return TERMINAL_HANDOFF_FILE  # Unbound diagnostic helpers only.
    if (
        not isinstance(generation, str)
        or not 1 <= len(generation) <= 128
        or any(not (char.isascii() and (char.isalnum() or char in "_-")) for char in generation)
    ):
        raise InvalidTerminalHandoff("invalid launch generation")
    return f"terminal_handoff.{generation}.json"


def terminal_handoff_path(run_dir: str | Path, *, launch_generation: str | None = None) -> Path:
    return Path(run_dir) / RESOURCE_STATS_DIR / STAGING_DIR / _handoff_filename(launch_generation)


def _open_handoff_directory(run_dir: str | Path, *, include_staging: bool, create: bool = False) -> int:
    """Open every parent component without following an untrusted symlink."""

    root = Path(run_dir)
    components = (RESOURCE_STATS_DIR, STAGING_DIR) if include_staging else (RESOURCE_STATS_DIR,)
    if not _SECURE_HANDOFF_DIR_FD or (create and not _SECURE_HANDOFF_WRITE_DIR_FD):
        raise OSError("resource handoff requires directory-relative filesystem operations")
    if create:
        root.mkdir(parents=True, exist_ok=True)

    current_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in components:
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
            next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def write_terminal_handoff(
    run_dir: str | Path, handoff: Mapping[str, Any], *, launch_generation: str | None = None
) -> Path:
    """Atomically write the child handoff under its existing job workspace."""

    filename = _handoff_filename(launch_generation)
    if launch_generation is not None:
        handoff = {**handoff, "launch_generation": launch_generation}
    data = canonical_json_bytes(handoff)
    if len(data) > MAX_TERMINAL_HANDOFF_BYTES:
        raise InvalidTerminalHandoff("terminal handoff exceeds its size limit")
    path = terminal_handoff_path(run_dir, launch_generation=launch_generation)
    directory_fd = _open_handoff_directory(run_dir, include_staging=True, create=True)
    temp_name = f".terminal-{secrets.token_hex(16)}.tmp"
    fd = None
    try:
        fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o600, dir_fd=directory_fd)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
        os.replace(temp_name, filename, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        return path
    finally:
        try:
            if fd is not None:
                os.close(fd)
                try:
                    os.unlink(temp_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        finally:
            os.close(directory_fd)


def read_terminal_handoff(run_dir: str | Path, *, expected_generation: str | None = None) -> Optional[dict[str, Any]]:
    """Read only this launch's handoff; return ``None`` if absent or malformed."""

    try:
        filename = _handoff_filename(expected_generation)
    except InvalidTerminalHandoff:
        return None

    directory_fd = None
    fd = None
    try:
        directory_fd = _open_handoff_directory(run_dir, include_staging=True)
        # The job process owns this self-report, so treat the path as untrusted:
        # do not follow any path-component symlink and do not let a FIFO/device
        # block the parent.
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(filename, flags, dir_fd=directory_fd)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_TERMINAL_HANDOFF_BYTES:
            return None
        with os.fdopen(fd, "rb") as stream:
            fd = None
            data = stream.read(MAX_TERMINAL_HANDOFF_BYTES + 1)
    except OSError:
        return None
    finally:
        if fd is not None:
            os.close(fd)
        if directory_fd is not None:
            os.close(directory_fd)
    if len(data) > MAX_TERMINAL_HANDOFF_BYTES:
        return None

    try:
        value = json.loads(data)
        if not isinstance(value, dict):
            return None
        if expected_generation is not None and value.get("launch_generation") != expected_generation:
            return None
        return value
    except (ValueError, RecursionError):
        return None


def remove_terminal_handoff(run_dir: str | Path, *, expected_generation: str | None = None) -> None:
    """Remove only the owned generation; never consume a replacement's report."""

    try:
        filename = _handoff_filename(expected_generation)
    except InvalidTerminalHandoff:
        return

    directory_fd = None
    try:
        directory_fd = _open_handoff_directory(run_dir, include_staging=True)
        os.unlink(filename, dir_fd=directory_fd)
    except FileNotFoundError:
        pass
    except OSError:
        # Unsafe job-owned paths are left untouched.
        return
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

    stats_fd = None
    try:
        stats_fd = _open_handoff_directory(run_dir, include_staging=False)
        os.rmdir(STAGING_DIR, dir_fd=stats_fd)
    except OSError:
        pass
    finally:
        if stats_fd is not None:
            os.close(stats_fd)
