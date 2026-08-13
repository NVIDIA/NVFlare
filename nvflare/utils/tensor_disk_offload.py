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


@dataclass
class TensorDiskOffloadCleanupResult:
    removed: List[str] = field(default_factory=list)
    failures: Dict[str, str] = field(default_factory=dict)


def create_tensor_disk_offload_root(job_id: str, temp_root: Optional[str] = None) -> str:
    """Create a root whose ownership can be verified by the surviving job parent."""
    check_job_id(job_id)
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
            # Close the marker before deletion and revalidate the root identity to
            # avoid following a replacement at the final mutation boundary.
            final_stat = os.stat(entry.path, follow_symlinks=False)
            if not stat.S_ISDIR(final_stat.st_mode):
                continue
            if (final_stat.st_dev, final_stat.st_ino) != (root_stat.st_dev, root_stat.st_ino):
                continue
            shutil.rmtree(entry.path)
            result.removed.append(entry.path)
        except FileNotFoundError:
            continue
        except Exception as e:
            result.failures[entry.path] = str(e)
    return result


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
