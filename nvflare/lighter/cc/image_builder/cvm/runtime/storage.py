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

"""Locate guest disks by their launch-assigned identities."""

import time
from pathlib import Path

from ..common.contracts import DISK_ROLES
from ..common.errors import require


def disk_device(role, *, wait=False):
    """Select a disk by its launch-assigned serial, never Linux probe order."""
    require(role in DISK_ROLES, "Unknown disk role")
    path = Path("/dev/disk/by-id") / ("scsi-0QEMU_QEMU_HARDDISK_cvm-" + role)
    if wait:
        deadline = time.monotonic() + 30
        while not path.is_block_device():
            require(time.monotonic() < deadline, "Required CVM disk is missing")
            time.sleep(0.1)
    return str(path)
