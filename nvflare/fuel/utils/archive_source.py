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

"""Descriptor-relative input for filesystem archives."""

import os
import stat
from contextlib import contextmanager

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_OPEN_FLAGS = os.O_RDONLY | _NOFOLLOW | _NONBLOCK


@contextmanager
def open_archive_entry(path: str, *, dir_fd=None, directory=False):
    """Open a regular file or directory without following its final path component.

    Absolute source paths come from the caller's trusted parent directories.
    Descendants are opened by name relative to a held directory descriptor.
    """
    if not (_NOFOLLOW and _DIRECTORY and _NONBLOCK):
        raise OSError("workspace archives require no-follow filesystem operations")
    flags = _OPEN_FLAGS | _DIRECTORY if directory else _OPEN_FLAGS
    fd = os.open(path, flags, dir_fd=dir_fd)
    try:
        info = os.fstat(fd)
        valid_type = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if not valid_type:
            raise OSError(f"unexpected archive source type: {path!r}")
        yield fd, info
    finally:
        os.close(fd)


@contextmanager
def _open_relative_directory(root_fd: int, relative: str):
    if not relative:
        yield root_fd
        return

    # Reopen each component from the held root so directory depth does not
    # determine how many descriptors stay open.
    fd = os.dup(root_fd)
    try:
        for name in relative.split(os.sep):
            child_fd = os.open(name, _OPEN_FLAGS | _DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child_fd
        yield fd
    finally:
        os.close(fd)


def walk_archive_directory(root_fd: int, *, skip_unreadable_children=False):
    """Yield each directory with its immediate subdirectories and regular files.

    Symlinks and special files are omitted. A symlink swapped in for a directory
    causes the walk to fail. The whole-workspace archive can keep the directory
    entry for an unreadable child and omit its contents.
    """
    pending = [""]
    while pending:
        relative = pending.pop()
        try:
            with _open_relative_directory(root_fd, relative) as dir_fd:
                dirs = {}
                files = {}
                with os.scandir(dir_fd) as entries:
                    for entry in entries:
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            dirs[entry.name] = info
                        elif stat.S_ISREG(info.st_mode):
                            files[entry.name] = info
                yield dir_fd, relative, dirs, files
        except PermissionError:
            if not relative or not skip_unreadable_children:
                raise
            continue
        pending.extend(os.path.join(relative, name) for name in reversed(dirs))
