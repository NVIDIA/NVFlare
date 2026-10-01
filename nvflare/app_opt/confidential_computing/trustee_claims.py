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

"""Reviewed successful EAR claims, shared with assembled role kits."""

import re

TRUST_VECTOR = {
    "executables": 3,
    "hardware": 2,
    "configuration": 3,
    "file-system": 0,
    "instance-identity": 0,
    "runtime-opaque": 0,
    "storage-opaque": 0,
    "sourced-data": 0,
}

# Exact policy contracts, not thresholds over AR4SI values. The TDX policy
# appraises an approved non-debug configuration as 2 rather than SNP's 3.
CPU_TRUST_VECTORS = {
    "snp": dict(TRUST_VECTOR),
    "tdx": {**TRUST_VECTOR, "configuration": 2},
}


def cpu_evidence_type(evidence):
    """Identify the platform in already authenticated Trustee EAR evidence.

    This function does not verify the EAR signature. Callers must first verify
    it against their independently authenticated Attestation Service key.
    """
    if not isinstance(evidence, dict):
        raise ValueError("Missing CPU annotated evidence")
    cpu_types = set(evidence) - {"report_data", "runtime_data_claims", "init_data", "init_data_claims"}
    if len(cpu_types) != 1 or not cpu_types <= CPU_TRUST_VECTORS.keys():
        raise ValueError("Expected exactly one supported SNP or TDX CPU evidence type")
    cpu_type = next(iter(cpu_types))
    if not isinstance(evidence[cpu_type], dict) or not evidence[cpu_type]:
        raise ValueError("Malformed CPU hardware evidence")
    return cpu_type


def normalized_init_data(evidence):
    """Read an attested SHA-256 InitData binding, never arbitrary prefix bytes.

    Call only on AS-signature-verified evidence. Pinned Trustee emits SNP's
    32-byte host_data directly. For TDX it emits the entire 48-byte MRCONFIGID,
    which Kata binds as SHA-256 followed by 16 zero bytes. The duplicated quote
    field must agree, and unsupported widths, encodings and padding fail closed.
    """
    cpu_type = cpu_evidence_type(evidence)
    value = evidence.get("init_data")
    width = 64 if cpu_type == "snp" else 96
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(width) + "}", value):
        raise ValueError("Invalid attested InitData encoding")
    if cpu_type == "tdx":
        quote = evidence["tdx"].get("quote")
        body = quote.get("body") if isinstance(quote, dict) else None
        if not isinstance(body, dict) or body.get("mr_config_id") != value or value[64:] != "0" * 32:
            raise ValueError("Invalid TDX InitData binding or padding")
    return value[:64]
