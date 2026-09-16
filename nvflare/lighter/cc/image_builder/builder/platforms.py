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

"""Host selection and local report adapters. No vendor-only or vTPM fallback."""

import ctypes
import fcntl
import os
import struct
from pathlib import Path

from .common import PLATFORMS, require


def host_capabilities(proc=Path("/proc"), sys=Path("/sys")):
    info = (proc / "cpuinfo").read_text() if (proc / "cpuinfo").exists() else ""
    result = set()
    for name, vendor, flag, module in [
        ("amd_sev_snp", "AuthenticAMD", "sev_snp", "kvm_amd/parameters/sev_snp"),
        ("intel_tdx", "GenuineIntel", "tdx", "kvm_intel/parameters/tdx"),
    ]:
        capability = sys / "module" / module
        enabled = capability.exists() and capability.read_text().strip().lower() in ("1", "y", "yes")
        flags = set(info.replace(":", " ").split())
        # A loaded KVM module explicitly disabling the feature overrides CPU
        # enumeration. A CPU flag alone cannot turn a disabled host into a TEE.
        available = enabled if capability.exists() else flag in flags
        if vendor in info and available:
            result.add(name)
    return result


def select_platform(profile, explicit=None, capabilities=None):
    enabled = {k for k, v in profile["platforms"].items() if v.get("enabled", True)}
    if explicit is not None:
        require(explicit in PLATFORMS and explicit in enabled, "-p must name a supported, enabled platform")
        return explicit
    found = (host_capabilities() if capabilities is None else capabilities) & enabled
    require(len(found) == 1, "Cannot unambiguously auto-detect an enabled SNP/TDX host; specify -p <platform>")
    return next(iter(found))


def guest_platform(dev=Path("/dev")):
    found = [
        name for name, device in [("amd_sev_snp", "sev-guest"), ("intel_tdx", "tdx_guest")] if (dev / device).exists()
    ]
    require(len(found) == 1, "No unique supported guest TEE device; refusing fallback")
    return found[0]


def parse_snp_report(report, nonce):
    require(len(report) == 1184, "Unexpected SNP report length")
    require(struct.unpack_from("<I", report)[0] in (2, 3, 4, 5), "Unsupported SNP report version")
    require(report[80:144] == nonce, "SNP local report nonce mismatch")
    return report[192:224]


def parse_tdx_report(report, nonce):
    require(len(report) == 1024 and report[0] == 0x81, "Unsupported TDREPORT type or length")
    require(report[128:192] == nonce, "TDX local report nonce mismatch")
    return report[576:624]


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


def measurements(platform, report):
    import base64

    if platform == "intel_tdx":
        require(len(report) == 1024, "Invalid TDREPORT")
        return {
            "mr_td": report[528:576].hex(),
            "rtmr_0": report[720:768].hex(),
            "rtmr_1": report[768:816].hex(),
            "rtmr_2": report[816:864].hex(),
        }
    require(platform == "amd_sev_snp" and len(report) == 1184, "Invalid SNP report")
    return {"snp.measurement": base64.b64encode(report[144:192]).decode()}


def verify_local_binding(platform, digest):
    expected = digest + (bytes(16) if platform == "intel_tdx" else b"")
    require(local_binding(platform) == expected, "Attached vault does not match local TEE binding")
