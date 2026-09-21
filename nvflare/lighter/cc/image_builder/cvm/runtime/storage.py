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

"""Locate guest disks and release the authenticated vault mapping."""

import time
from pathlib import Path

from ..common.contracts import DISK_ROLES
from ..common.errors import require
from ..common.linux import run


def close_vault():
    """Release a complete or partial unlock; any unconfirmed cleanup is fatal."""
    mount = Path("/vault")

    def active():
        # An interrupted cryptsetup can leave a kernel mapping before udev has
        # created its /dev/mapper link. Inspect kernel state, not that link.
        names = run(["dmsetup", "info", "--columns", "--noheadings", "--options", "name"], timeout=10)
        return b"vault" in (name.strip() for name in names.splitlines())

    if mount.is_mount():
        run(["umount", str(mount)], timeout=60)
    if active():
        run(["cryptsetup", "close", "vault"], timeout=60)
    require(not mount.is_mount() and not active(), "Vault cleanup could not be confirmed")


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
