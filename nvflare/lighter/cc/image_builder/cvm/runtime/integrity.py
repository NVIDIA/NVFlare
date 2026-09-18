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

"""Monitor guest storage integrity and fail closed on errors."""

import errno
import os
import re
import selectors
from pathlib import Path

from ..common.errors import BuildError, require
from ..common.linux import run
from ..common.luks import validate_mapping
from .systemd import notify


def healthy_status(status):
    fields = status.split()
    # Kernel dm-integrity status: start length integrity failures provided_sectors recalc_sector.
    require(len(fields) >= 6 and fields[2] == "integrity" and fields[3].isdigit(), "Unrecognized integrity status")
    require(int(fields[3]) == 0, "Vault authentication failure")


def watch():
    major, minor = validate_mapping("vault").split(":")
    fd = os.open("/dev/kmsg", os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        os.lseek(fd, 0, os.SEEK_END)
        selector = selectors.DefaultSelector()
        selector.register(fd, selectors.EVENT_READ)

        def check():
            healthy_status(run(["dmsetup", "status", "-j", major, "-m", minor], timeout=10).decode())
            require(Path("/dev/mapper/vault").exists(), "Encrypted vault mapping disappeared")

        check()
        notify("READY=1\nSTATUS=Watching authenticated vault")
        previous = None
        while True:
            for _, _ in selector.select(timeout=1):
                while True:
                    try:
                        event = os.read(fd, 16384).decode(errors="replace")
                    except BlockingIOError:
                        break
                    except OSError as exc:
                        if exc.errno == errno.EPIPE:
                            raise BuildError("Kernel integrity event stream lost records") from None
                        raise
                    require(event, "Kernel integrity event stream closed")
                    prefix, message = event.split(";", 1)
                    sequence = int(prefix.split(",")[1])
                    require(previous is None or sequence == previous + 1, "Kernel event sequence gap")
                    previous = sequence
                    # Fail closed on any vault authentication error even
                    # before a device-specific counter update.
                    require(
                        not re.search(
                            r"(?i)(integrity.*(mismatch|failure|failed)|"
                            r"crypt.*(auth.*fail|integrity)|I/O error, dev dm-)",
                            message,
                        ),
                        "Kernel reported a vault integrity error",
                    )
            check()
            notify("WATCHDOG=1")
    finally:
        os.close(fd)


if __name__ == "__main__":
    try:
        watch()
    except Exception:
        # No token, key, payload or untrusted kernel text is sent to the journal.
        raise SystemExit("Vault integrity supervision failed") from None
