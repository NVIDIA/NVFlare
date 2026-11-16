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

"""Frame serial evidence and validate collected reference reports."""

import base64
import json
import re
import struct
import zlib

from .errors import require
from .io import canonical
from .measurements import measurements, parse_snp_report, parse_tdx_report, validate_measurements
from .references import SNP_BOOLS, SNP_INTS, SNP_LISTS, snp_guest_policy, validate_snp_policy


def verify_reference(platform, evidence):
    require(evidence["platform"] == platform, "Reference platform mismatch")
    report = base64.b64decode(evidence["report"], validate=True)
    nonce = base64.b64decode(evidence["nonce"], validate=True)
    require(len(nonce) == 64, "Invalid reference nonce")
    if platform == "intel_tdx":
        parse_tdx_report(report, nonce)
        require(base64.b64decode(evidence["ccel"], validate=True), "Missing reference CCEL")
    else:
        parse_snp_report(report, nonce)
    require(evidence["measurements"] == measurements(platform, report), "Reference evidence measurements mismatch")
    validate_measurements(platform, evidence["measurements"])


def tdx_tcb(evidence):
    """Return the TDX TCB identity from structurally verified local evidence."""
    verify_reference("intel_tdx", evidence)
    report = base64.b64decode(evidence["report"], validate=True)
    return {
        "mr_seam": report[280:328].hex(),
        "tcb_svn": report[264:280].hex(),
        "xfam": report[520:528].hex(),
    }


def require_approved_tdx_tcb(evidence, references):
    """Reject a reference boot whose platform TCB is absent from approved inputs."""
    for name, value in tdx_tcb(evidence).items():
        approved = references.get(name)
        require(
            isinstance(approved, list) and value in approved,
            f"Reference boot TDX {name} is not approved; run cvmctl inspect-tcb, review and update the approved "
            "TCB references, increment profile_version, and rebuild",
        )


def snp_tcb(evidence):
    """Return the SNP TCB and launch configuration from verified local evidence."""
    verify_reference("amd_sev_snp", evidence)
    report = base64.b64decode(evidence["report"], validate=True)
    version = struct.unpack_from("<I", report, 0)[0]
    require(
        version >= 3,
        "SNP report v2 does not identify the CPU generation; use a v3 or later report for TCB approval",
    )
    require(
        report[0x188] != 0x1A,
        "AMD Turin SNP is not supported by the pinned Trustee v0.22 policy because it does not expose FMC SPL",
    )
    policy = struct.unpack_from("<Q", report, 8)[0]
    platform_info = struct.unpack_from("<Q", report, 64)[0]
    reported_tcb = struct.unpack_from("<Q", report, 384)[0]
    validate_snp_policy(policy)
    return {
        "snp_bootloader": reported_tcb & 0xFF,
        "snp_tee_svn": (reported_tcb >> 8) & 0xFF,
        "snp_snp_svn": (reported_tcb >> 48) & 0xFF,
        "snp_microcode": (reported_tcb >> 56) & 0xFF,
        "snp_smt_enabled": bool(platform_info & 1),
        "snp_tsme_enabled": bool(platform_info & 2),
        "snp_guest_abi_minor": policy & 0xFF,
        "snp_guest_abi_major": (policy >> 8) & 0xFF,
        "snp_smt_allowed": bool(policy & (1 << 16)),
        "snp_single_socket": bool(policy & (1 << 20)),
    }


def require_approved_snp_tcb(evidence, references):
    """Reject a reference boot whose SNP TCB/configuration is not approved."""
    candidate = snp_tcb(evidence)
    for name in SNP_LISTS:
        require(
            candidate[name] in references.get(name, ()),
            f"Reference boot SNP {name} is not approved; inspect the signed report, review and update the approved "
            "TCB references, increment profile_version, and rebuild",
        )
    for name in SNP_BOOLS | SNP_INTS:
        require(
            candidate[name] == references.get(name),
            f"Reference boot SNP {name} differs from the approved profile; review the reference and rebuild",
        )
    report = base64.b64decode(evidence["report"], validate=True)
    require(
        struct.unpack_from("<Q", report, 8)[0] == snp_guest_policy(references),
        "Reference boot SNP guest policy differs from the approved profile; review the reference and rebuild",
    )


def serial_frames(evidence):
    data = base64.b64encode(zlib.compress(canonical(evidence))).decode()
    chunks = [data[i : i + 1024] for i in range(0, len(data), 1024)]
    return [f"CVM_REFERENCE_V2 {i + 1}/{len(chunks)} {part}" for i, part in enumerate(chunks)]


def serial_evidence(text):
    frames = {}
    count = None
    # A serial log may be read while the guest is still writing its final frame.
    # Require the line terminator emitted by print() before treating a frame as
    # complete; otherwise a valid Base64 prefix can be decoded prematurely.
    # journal+console prefixes service output with a monotonic timestamp and
    # process identity; direct serial output has no prefix.
    prefix = r"(?:\[\s*\d+(?:\.\d+)?\]\s+[^\s\[\]:]+\[\d+\]:\s+)?"
    for match in re.finditer(r"(?m)^" + prefix + r"CVM_REFERENCE_V2 (\d+)/(\d+) ([A-Za-z0-9+/=]+)\r?\n", text):
        index, total = int(match[1]), int(match[2])
        require(1 <= index <= total <= 16384 and (count is None or total == count), "Invalid reference frame count")
        require(index not in frames or frames[index] == match[3], "Conflicting reference frame")
        frames[index] = match[3]
        count = total
    if count is None or len(frames) != count:
        return None
    packed = base64.b64decode("".join(frames[i] for i in range(1, count + 1)), validate=True)
    inflater = zlib.decompressobj()
    data = inflater.decompress(packed, 16 * 1024**2)
    require(inflater.eof and not inflater.unused_data, "Invalid or oversized reference evidence")
    return json.loads(data)
