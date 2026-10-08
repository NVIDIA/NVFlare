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

"""Notify PID 1 from the guest supervisor and integrity monitor."""

import math
import os
import socket

from ..common.errors import require


def notify(message):
    address = os.environ.get("NOTIFY_SOCKET")
    require(address, "CVM supervisor requires systemd notification")
    if address.startswith("@"):
        address = "\0" + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as connection:
        connection.settimeout(1)
        connection.connect(address)
        connection.sendall(message.encode())


def watchdog(seconds):
    """Arm PID 1's independent deadline for the next bounded supervisor phase."""
    require(0 < seconds <= 3600, "Invalid supervisor watchdog deadline")
    notify(f"WATCHDOG_USEC={math.ceil(seconds * 1000000)}\nWATCHDOG=1")
