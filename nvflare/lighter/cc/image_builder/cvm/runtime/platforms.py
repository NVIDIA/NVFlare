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

"""Read guest TEE devices and verify the local vault binding."""

import ctypes
import fcntl
import os
import struct
from pathlib import Path

from ..common.errors import require
from ..common.measurements import parse_snp_report, parse_tdx_report


def guest_platform(dev=Path("/dev")):
    found = [
        name for name, device in [("amd_sev_snp", "sev-guest"), ("intel_tdx", "tdx_guest")] if (dev / device).exists()
    ]
    require(len(found) == 1, "No unique supported guest TEE device; refusing fallback")
    return found[0]


def local_report(platform):
    """Read the local hardware report. KBS separately authenticates remote evidence.

    UAPI: linux/sev-guest.h and linux/tdx-guest.h. No untrusted subprocess
    output, sidecar or command-line value can stand in for a local report.
    """
    nonce = os.urandom(64)
    if platform == "intel_tdx":
        request = bytearray(nonce + bytes(1024))
        with open("/dev/tdx_guest", "rb", buffering=0) as device:
            # _IOWR('T', 1, struct tdx_report_req[1088])
            fcntl.ioctl(device.fileno(), 0xC4405401, request, True)
        report = bytes(request[64:])
        parse_tdx_report(report, nonce)
        return report, nonce
    require(platform == "amd_sev_snp", "Unsupported local report adapter")

    class Request(ctypes.Structure):
        _fields_ = [
            ("version", ctypes.c_uint8),
            ("request", ctypes.c_uint64),
            ("response", ctypes.c_uint64),
            ("error", ctypes.c_uint64),
        ]

    request = ctypes.create_string_buffer(nonce + bytes(32), 96)
    response = ctypes.create_string_buffer(4000)
    ioctl = Request(1, ctypes.addressof(request), ctypes.addressof(response), 0)
    with open("/dev/sev-guest", "rb", buffering=0) as device:
        libc = ctypes.CDLL(None, use_errno=True)
        require(
            libc.ioctl(device.fileno(), ctypes.c_ulong(0xC0205300), ctypes.byref(ioctl)) == 0,
            "SNP local report ioctl failed",
        )
    require(ioctl.error == 0, "SNP firmware rejected local report")
    status, size = struct.unpack_from("<II", response.raw)
    require(status == 0 and size == 1184, "Invalid SNP firmware report response")
    report = response.raw[32 : 32 + size]
    parse_snp_report(report, nonce)
    return report, nonce


def local_binding(platform):
    report, nonce = local_report(platform)
    return parse_tdx_report(report, nonce) if platform == "intel_tdx" else parse_snp_report(report, nonce)


def verify_local_binding(platform, digest):
    expected = digest + (bytes(16) if platform == "intel_tdx" else b"")
    require(local_binding(platform) == expected, "Attached vault does not match local TEE binding")
