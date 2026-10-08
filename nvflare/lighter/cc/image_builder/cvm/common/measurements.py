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

"""Parse and validate TDX and SNP measurement formats."""

import re
import struct

from .contracts import PLATFORMS
from .errors import require


def parse_snp_report(report, nonce):
    require(len(report) == 1184, "Unexpected SNP report length")
    require(struct.unpack_from("<I", report)[0] in (2, 3, 4, 5), "Unsupported SNP report version")
    require(report[80:144] == nonce, "SNP local report nonce mismatch")
    return report[192:224]


def parse_tdx_report(report, nonce):
    require(len(report) == 1024 and report[0] == 0x81, "Unsupported TDREPORT type or length")
    require(report[128:192] == nonce, "TDX local report nonce mismatch")
    return report[576:624]


def measurements(platform, report):
    if platform == "intel_tdx":
        require(len(report) == 1024, "Invalid TDREPORT")
        return {
            "mr_td": report[528:576].hex(),
            "rtmr_0": report[720:768].hex(),
            "rtmr_1": report[768:816].hex(),
            "rtmr_2": report[816:864].hex(),
        }
    require(platform == "amd_sev_snp" and len(report) == 1184, "Invalid SNP report")
    return {"snp.measurement": report[144:192].hex()}


def validate_measurements(platform, values):
    require(platform in PLATFORMS and isinstance(values, dict), "Invalid reference measurements")
    if platform == "amd_sev_snp":
        require(set(values) == {"snp.measurement"}, "SNP requires exactly its launch measurement")
        value = values["snp.measurement"]
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{96}", value), "Invalid SNP measurement encoding")
    else:
        require(
            set(values) == {"mr_td", "rtmr_0", "rtmr_1", "rtmr_2"},
            "TDX requires MRTD and RTMR0/1/2",
        )
        require(
            all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{96}", x) for x in values.values()),
            "Invalid TDX measurement encoding",
        )
