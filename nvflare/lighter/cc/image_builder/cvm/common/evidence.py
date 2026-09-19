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
import zlib

from .errors import require
from .io import canonical
from .measurements import measurements, parse_snp_report, parse_tdx_report, validate_measurements


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
