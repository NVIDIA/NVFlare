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

"""Checked read/write of the private, job-process-owned terminal handoff.

The child writes capacity, CPU work, file sizes, and traffic observations.
The parent validates each measurement while reading the handoff. Identity
comes from parent-owned lifecycle state, never from the child file.
Directory-relative operations protect against path redirection on supported
platforms. Other platforms reject pre-existing directory symlinks but do not
claim race-proof traversal.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from .contract import (
    MAX_PARTICIPANT_SUMMARY_BYTES,
    ContractError,
    canonical_json_bytes,
    normalize_duration,
    normalize_measurement,
)

# The private handoff carries startup capacity and accumulated execution time.
INTERNAL_HANDOFF_VERSION = "4"
INTERNAL_HANDOFF_KIND = "nvflare.resource_stats.internal.terminal_handoff"
MAX_TERMINAL_HANDOFF_BYTES = MAX_PARTICIPANT_SUMMARY_BYTES

RESOURCE_STATS_DIR = "resource_stats"
STAGING_DIR = "staging"
TERMINAL_HANDOFF_FILE = "terminal_handoff.json"
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


def terminal_handoff_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / RESOURCE_STATS_DIR / STAGING_DIR / TERMINAL_HANDOFF_FILE


def _open_handoff_directory(
    run_dir: str | Path, *, include_staging: bool, create: bool = False
) -> tuple[int | None, Path]:
    """Open every parent component without following an untrusted symlink."""

    root = Path(run_dir)
    components = (RESOURCE_STATS_DIR, STAGING_DIR) if include_staging else (RESOURCE_STATS_DIR,)
    target = root.joinpath(*components)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not _SECURE_HANDOFF_DIR_FD or (create and not _SECURE_HANDOFF_WRITE_DIR_FD):
        current = root
        for component in (None, *components):
            if component is not None:
                current /= component
                if create:
                    current.mkdir(exist_ok=True)
            mode = current.lstat().st_mode
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                raise OSError(f"resource handoff path is not a plain directory: {current}")
        return None, target

    opened = []
    try:
        current_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
        opened.append(current_fd)
        for component in components:
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
            next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            opened.append(next_fd)
            current_fd = next_fd
        result_fd = opened.pop()
        return result_fd, target
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _validate_handoff(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "internal_version",
        "kind",
        "resource_capacity",
        "resource_time",
        "resource_utilization",
        "workspace_filesystem",
    }
    if not isinstance(value, Mapping) or not required <= set(value) or set(value) - required - {"duration_seconds"}:
        raise InvalidTerminalHandoff(f"terminal handoff fields must be {sorted(required)}")
    if value["internal_version"] != INTERNAL_HANDOFF_VERSION or value["kind"] != INTERNAL_HANDOFF_KIND:
        raise InvalidTerminalHandoff("terminal handoff has an unsupported version or kind")
    result = {"internal_version": INTERNAL_HANDOFF_VERSION, "kind": INTERNAL_HANDOFF_KIND}
    if "duration_seconds" in value:
        try:
            result["duration_seconds"] = normalize_duration(value["duration_seconds"])
        except ContractError:
            pass  # Preserve untimed quantities; never invent an elapsed duration.
    for name in ("resource_capacity", "resource_time", "workspace_filesystem"):
        try:
            result[name] = normalize_measurement(name, value[name])
        except ContractError:
            result[name] = {"status": "unavailable", "issues": ["malformed_source"]}
    utilization = value["resource_utilization"]
    fields = {"cpu_consumed", "retained_content", "message_traffic"}
    if not isinstance(utilization, Mapping) or set(utilization) != fields:
        raise InvalidTerminalHandoff(
            "resource_utilization must contain CPU work, retained content, and message traffic"
        )
    result["resource_utilization"] = {}
    for name in sorted(fields):
        try:
            result["resource_utilization"][name] = normalize_measurement(name, utilization[name])
        except ContractError:
            result["resource_utilization"][name] = {"status": "unavailable", "issues": ["malformed_source"]}
    return result


def write_terminal_handoff(run_dir: str | Path, handoff: Mapping[str, Any]) -> Path:
    """Atomically write the child handoff under its existing job workspace."""

    validated = _validate_handoff(handoff)
    data = canonical_json_bytes(validated)
    if len(data) > MAX_TERMINAL_HANDOFF_BYTES:
        raise InvalidTerminalHandoff("terminal handoff exceeds its size limit")
    path = terminal_handoff_path(run_dir)
    directory_fd, directory_path = _open_handoff_directory(run_dir, include_staging=True, create=True)
    temp_path = None
    try:
        if directory_fd is None:
            # On platforms without directory-relative operations, reject
            # existing directory symlinks above, then retain best-effort atomic
            # publication. This fallback does not claim protection from races.
            stream = tempfile.NamedTemporaryFile(dir=directory_path, prefix=".terminal-", suffix=".tmp", delete=False)
            temp_path = Path(stream.name)
        else:
            for _ in range(8):
                temp_name = f".terminal-{secrets.token_hex(8)}.tmp"
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
                try:
                    fd = os.open(temp_name, flags, mode=0o600, dir_fd=directory_fd)
                except FileExistsError:
                    continue
                temp_path = temp_name
                try:
                    stream = os.fdopen(fd, "wb")
                except BaseException:
                    os.close(fd)
                    raise
                break
            else:
                raise OSError("could not create a private resource handoff temporary file")
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if directory_fd is None:
            os.replace(temp_path, path)
        else:
            os.replace(temp_path, TERMINAL_HANDOFF_FILE, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        return path
    finally:
        try:
            if temp_path is not None:
                try:
                    if directory_fd is None:
                        temp_path.unlink()
                    else:
                        os.unlink(temp_path, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        finally:
            if directory_fd is not None:
                os.close(directory_fd)


def read_terminal_handoff(run_dir: str | Path) -> Optional[dict[str, Any]]:
    """Read and validate the fixed child handoff; return ``None`` if absent/bad."""

    directory_fd = None
    fd = None
    try:
        directory_fd, directory_path = _open_handoff_directory(run_dir, include_staging=True)
        # The job process owns this self-report, so treat the path as untrusted:
        # do not follow any path-component symlink and do not let a FIFO/device
        # block the parent.
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if directory_fd is None:
            path = directory_path / TERMINAL_HANDOFF_FILE
            if not hasattr(os, "O_NOFOLLOW") and path.is_symlink():
                return None
            fd = os.open(path, flags)
        else:
            fd = os.open(TERMINAL_HANDOFF_FILE, flags, dir_fd=directory_fd)
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

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidTerminalHandoff(f"duplicate terminal handoff key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(InvalidTerminalHandoff(value)),
        )
        return _validate_handoff(value)
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidTerminalHandoff, RecursionError, TypeError):
        return None


def remove_terminal_handoff(run_dir: str | Path) -> None:
    """Remove private staging data before the finalized workspace is archived."""

    directory_fd = None
    try:
        directory_fd, directory_path = _open_handoff_directory(run_dir, include_staging=True)
        if directory_fd is None:
            (directory_path / TERMINAL_HANDOFF_FILE).unlink()
        else:
            os.unlink(TERMINAL_HANDOFF_FILE, dir_fd=directory_fd)
    except FileNotFoundError:
        pass
    except OSError:
        # Unsafe job-owned paths are left untouched.  The server's
        # final inventory check will reject any remaining staging entry.
        return
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

    stats_fd = None
    try:
        stats_fd, stats_path = _open_handoff_directory(run_dir, include_staging=False)
        if stats_fd is None:
            (stats_path / STAGING_DIR).rmdir()
        else:
            os.rmdir(STAGING_DIR, dir_fd=stats_fd)
    except OSError:
        pass
    finally:
        if stats_fd is not None:
            os.close(stats_fd)
