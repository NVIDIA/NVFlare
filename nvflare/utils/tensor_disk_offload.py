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

"""Ownership records for job-worker tensor disk-offload roots."""

import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from nvflare.apis.utils.format_check import check_job_id

TENSOR_DISK_OFFLOAD_ROOT_PREFIX = "nvflare_tensor_offload_"
_OWNER_FILE = ".nvflare_tensor_offload_owner.json"
_OWNER_VERSION = 1
_MAX_OWNER_RECORD_BYTES = 4096
_CLEANUP_ROOT_PREFIX = ".nvflare_tensor_offload_cleanup_"
_SECURE_CLEANUP_SUPPORTED = (
    all(func in os.supports_dir_fd for func in (os.open, os.stat, os.rename, os.unlink, os.rmdir))
    and os.scandir in os.supports_fd
)
_UNSUPPORTED_CLEANUP_ERROR = "tensor disk offload requires secure directory-fd cleanup support"


@dataclass
class TensorDiskOffloadCleanupResult:
    removed: List[str] = field(default_factory=list)
    failures: Dict[str, str] = field(default_factory=dict)


def create_tensor_disk_offload_root(job_id: str, temp_root: Optional[str] = None) -> str:
    """Create a root whose ownership can be verified by the surviving job parent."""
    check_job_id(job_id)
    if not _SECURE_CLEANUP_SUPPORTED:
        # Do not create temporary data that the surviving parent cannot safely
        # reclaim. In particular, Python's dir_fd APIs are unavailable on Windows.
        raise RuntimeError(_UNSUPPORTED_CLEANUP_ERROR)
    root_dir = tempfile.mkdtemp(prefix=f"{TENSOR_DISK_OFFLOAD_ROOT_PREFIX}{job_id}_", dir=temp_root)
    try:
        root_stat = os.stat(root_dir, follow_symlinks=False)
        owner = {
            "version": _OWNER_VERSION,
            "job_id": job_id,
            "creator_pid": os.getpid(),
            "parent_pid": os.getppid(),
            "root_device": root_stat.st_dev,
            "root_inode": root_stat.st_ino,
        }
        marker_path = os.path.join(root_dir, _OWNER_FILE)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(marker_path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as marker:
            json.dump(owner, marker, sort_keys=True)
            marker.flush()
            os.fsync(marker.fileno())
        return root_dir
    except BaseException:
        shutil.rmtree(root_dir, ignore_errors=True)
        raise


def cleanup_owned_tensor_disk_offload_roots(
    job_id: str, owner_parent_pid: Optional[int] = None, temp_root: Optional[str] = None
) -> TensorDiskOffloadCleanupResult:
    """Remove roots created by a dead job worker owned by this client parent.

    Directory names are only a discovery hint. The root and marker must be real
    directories/files, and the record must exactly match the job, parent, device,
    and inode before deletion is attempted.
    """
    check_job_id(job_id)
    if owner_parent_pid is None:
        owner_parent_pid = os.getpid()
    if not isinstance(owner_parent_pid, int) or owner_parent_pid <= 0:
        raise ValueError("owner_parent_pid must be a positive int")

    result = TensorDiskOffloadCleanupResult()
    search_root = temp_root or tempfile.gettempdir()
    if not _SECURE_CLEANUP_SUPPORTED:
        result.failures[search_root] = _UNSUPPORTED_CLEANUP_ERROR
        return result
    job_root_prefix = f"{TENSOR_DISK_OFFLOAD_ROOT_PREFIX}{job_id}_"
    try:
        entries = list(os.scandir(search_root))
    except OSError as e:
        result.failures[search_root] = str(e)
        return result

    for entry in entries:
        if not entry.name.startswith(job_root_prefix):
            continue
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
            root_stat = entry.stat(follow_symlinks=False)
            owner = _read_owner_record(entry.path)
            if not owner:
                continue
            if (
                owner.get("version") != _OWNER_VERSION
                or owner.get("job_id") != job_id
                or owner.get("parent_pid") != owner_parent_pid
                or owner.get("root_device") != root_stat.st_dev
                or owner.get("root_inode") != root_stat.st_ino
            ):
                continue
            # Close the marker before deletion and revalidate the root identity.
            final_stat = os.stat(entry.path, follow_symlinks=False)
            if not stat.S_ISDIR(final_stat.st_mode):
                continue
            if (final_stat.st_dev, final_stat.st_ino) != (root_stat.st_dev, root_stat.st_ino):
                continue
            _quarantine_and_remove_root(entry.path, root_stat, search_root)
            result.removed.append(entry.path)
        except FileNotFoundError:
            continue
        except Exception as e:
            result.failures[entry.path] = str(e)
    return result


def _quarantine_and_remove_root(root_dir: str, expected_stat: os.stat_result, search_root: str) -> None:
    """Atomically detach a root from the shared namespace before deleting it.

    Revalidating a pathname immediately before ``shutil.rmtree`` still leaves a
    window in which another process can replace that pathname. Move the candidate
    into a new private directory first, then verify that the moved directory is
    the inode that was approved. Recursive deletion is performed relative to an
    open directory descriptor so it never resolves the original shared pathname.
    """
    cleanup_dir = tempfile.mkdtemp(prefix=_CLEANUP_ROOT_PREFIX, dir=search_root)
    cleanup_stat = os.stat(cleanup_dir, follow_symlinks=False)
    cleanup_fd = -1
    root_quarantined = False
    remove_cleanup_dir = False
    try:
        cleanup_fd = _open_directory(cleanup_dir)
        opened_cleanup_stat = os.fstat(cleanup_fd)
        if (opened_cleanup_stat.st_dev, opened_cleanup_stat.st_ino) != (cleanup_stat.st_dev, cleanup_stat.st_ino):
            raise RuntimeError("tensor disk-offload quarantine identity changed before use")
        root_name = os.path.basename(root_dir)
        os.rename(root_dir, root_name, dst_dir_fd=cleanup_fd)
        root_quarantined = True

        quarantined_stat = os.stat(root_name, dir_fd=cleanup_fd, follow_symlinks=False)
        expected_identity = (expected_stat.st_dev, expected_stat.st_ino)
        if (
            not stat.S_ISDIR(quarantined_stat.st_mode)
            or (
                quarantined_stat.st_dev,
                quarantined_stat.st_ino,
            )
            != expected_identity
        ):
            raise RuntimeError(
                f"tensor disk-offload root identity changed while quarantining; preserved under {cleanup_dir}"
            )

        _remove_tree_at(cleanup_fd, root_name, expected_identity)
        remove_cleanup_dir = True
    finally:
        if cleanup_fd >= 0:
            os.close(cleanup_fd)
        if remove_cleanup_dir or not root_quarantined:
            # This removes only the now-empty quarantine directory. It is never
            # used for recursive deletion, so a replacement cannot expose an
            # unrelated tree to cleanup.
            os.rmdir(cleanup_dir)


def _open_directory(path: str, dir_fd: Optional[int] = None) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, dir_fd=dir_fd)


def _remove_tree_at(parent_fd: int, name: str, expected_identity: tuple) -> None:
    """Remove one directory tree relative to a private, already-open parent."""
    root_fd = _open_directory(name, dir_fd=parent_fd)
    try:
        root_stat = os.fstat(root_fd)
        if (root_stat.st_dev, root_stat.st_ino) != expected_identity:
            raise RuntimeError("tensor disk-offload root identity changed inside quarantine")

        with os.scandir(root_fd) as entries:
            children = list(entries)
        for child in children:
            try:
                if child.is_dir(follow_symlinks=False):
                    child_stat = child.stat(follow_symlinks=False)
                    _remove_tree_at(root_fd, child.name, (child_stat.st_dev, child_stat.st_ino))
                else:
                    os.unlink(child.name, dir_fd=root_fd)
            except FileNotFoundError:
                continue

        final_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (final_stat.st_dev, final_stat.st_ino) != expected_identity:
            raise RuntimeError("tensor disk-offload root identity changed before removal")
        os.rmdir(name, dir_fd=parent_fd)
    finally:
        os.close(root_fd)


def _read_owner_record(root_dir: str) -> Optional[dict]:
    marker_path = os.path.join(root_dir, _OWNER_FILE)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(marker_path, flags)
    except (FileNotFoundError, OSError):
        return None

    try:
        marker_stat = os.fstat(fd)
        if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_size > _MAX_OWNER_RECORD_BYTES:
            return None
        with os.fdopen(fd, "r", encoding="utf-8") as marker:
            fd = -1
            owner = json.load(marker)
        return owner if isinstance(owner, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    finally:
        if fd >= 0:
            os.close(fd)
