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

"""Resolve-beneath and atomic-write helpers for untrusted result bundles."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path, PurePosixPath


class UnsafeArtifactPath(ValueError):
    """Raised when a serialized artifact path escapes its trusted root."""


def relative_posix_path(value: object, *, label: str, required_prefix: str | None = None) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise UnsafeArtifactPath(f"{label} must be a non-empty relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise UnsafeArtifactPath(f"{label} must stay beneath the result root")
    if required_prefix is not None and pure.parts[0] != required_prefix:
        raise UnsafeArtifactPath(f"{label} must start with {required_prefix!r}")
    return Path(*pure.parts)


def reject_symlink_components(path: Path, *, include_leaf: bool = True) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    current = Path(parts[0])
    stop = len(parts) if include_leaf else max(1, len(parts) - 1)
    for part in parts[1:stop]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise UnsafeArtifactPath(f"could not inspect path component {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise UnsafeArtifactPath(f"symlink path component is not allowed: {current}")


def path_beneath(
    root: Path,
    value: object,
    *,
    label: str,
    required_prefix: str | None = None,
) -> Path:
    relative = relative_posix_path(value, label=label, required_prefix=required_prefix)
    reject_symlink_components(root)
    candidate = root / relative
    reject_symlink_components(candidate)
    return candidate


def read_regular_file_beneath(root: Path, value: object, *, label: str, max_bytes: int) -> bytes | None:
    relative = relative_posix_path(value, label=label)
    reject_symlink_components(root)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current_fd = os.open(root, directory_flags | nofollow)
        descriptors.append(current_fd)
        for part in relative.parts[:-1]:
            current_fd = os.open(part, directory_flags | nofollow, dir_fd=current_fd)
            descriptors.append(current_fd)
        file_fd = os.open(relative.parts[-1], os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow, dir_fd=current_fd)
        descriptors.append(file_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > max_bytes:
            return None
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        return data if len(data) <= max_bytes else None
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return None
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def atomic_replace_bytes(target: Path, data: bytes, *, mode: int = 0o600) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(target.parent)
    descriptor = None
    temporary = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary = Path(temporary_name)
        os.fchmod(descriptor, mode)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, target)
        temporary = None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
