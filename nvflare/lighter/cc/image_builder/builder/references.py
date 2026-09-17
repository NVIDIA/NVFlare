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

"""Validate administrator-approved TCB inputs and isolate backend profiles."""

import datetime

from .common import require

SNP_LISTS = {"snp_bootloader", "snp_microcode", "snp_snp_svn", "snp_tee_svn"}
SNP_BOOLS = {"snp_smt_enabled", "snp_tsme_enabled", "snp_single_socket", "snp_smt_allowed"}
SNP_INTS = {"snp_guest_abi_major", "snp_guest_abi_minor"}
TDX_HEX = {"mr_seam": 96, "tcb_svn": 32, "xfam": 16}
GPU_NAMES = {"gpu_driver_versions", "gpu_vbios_versions"}
TCB_NAMES = GPU_NAMES | SNP_LISTS | SNP_BOOLS | SNP_INTS | set(TDX_HEX) | {"allowed_advisory_ids"}
MEASUREMENT_NAMES = {"snp_launch_measurement", "mr_td", "rtmr_0", "rtmr_1", "rtmr_2"}


def validate_references(values, platforms=(), *, finalized=False, gpu=False):
    import re

    require(isinstance(values, dict), "Approved TCB references must be a JSON object")
    allowed = TCB_NAMES | (MEASUREMENT_NAMES if finalized else set())
    unknown = sorted(set(values) - allowed)
    require(not unknown, "Unknown approved TCB reference keys: " + ", ".join(unknown))
    needed = set(GPU_NAMES) if gpu else set()
    if "amd_sev_snp" in platforms:
        needed |= SNP_LISTS | SNP_BOOLS | SNP_INTS
    if "intel_tdx" in platforms:
        needed |= set(TDX_HEX) | {"allowed_advisory_ids"}
    require(needed <= set(values), "Missing approved TCB reference keys: " + ", ".join(sorted(needed - set(values))))
    for name, value in values.items():
        if name in SNP_BOOLS:
            valid = type(value) is bool
        elif name in SNP_INTS:
            valid = type(value) is int and 0 <= value <= 255
        elif name in SNP_LISTS:
            valid = isinstance(value, list) and value and all(type(v) is int and 0 <= v <= 255 for v in value)
        elif name in TDX_HEX:
            valid = (
                isinstance(value, list)
                and value
                and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{%d}" % TDX_HEX[name], v) for v in value)
            )
        elif name == "allowed_advisory_ids":
            valid = isinstance(value, list) and all(
                isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in value
            )
        else:
            valid = isinstance(value, list) and value and all(isinstance(v, str) for v in value)
        require(valid, "Invalid approved TCB reference: " + name)
    return values


def profile_identity(manifest):
    return {"profile_version": manifest["profile_version"], "contract": manifest["contract"]}


def check_profile(existing, manifest):
    require(
        existing == profile_identity(manifest),
        "Trustee instance belongs to another security profile; use a separate instance",
    )


def merge_records(records, incoming, expiration):
    """Add new names only; an import cannot broaden or renew an existing approval."""
    by_name = {entry["name"]: entry for entry in records}
    require(len(by_name) == len(records), "Duplicate RVPS reference names")
    validate_references(incoming, finalized=True)
    expires = datetime.datetime.fromisoformat(expiration.replace("Z", "+00:00"))
    require(
        expires.tzinfo is not None and expires > datetime.datetime.now(datetime.timezone.utc),
        "Reference expiry must be in the future",
    )
    for name, value in incoming.items():
        if name in by_name:
            require(by_name[name]["value"] == value, "Conflicting approved reference: " + name)
            # Preserve the original expiry, including an already-expired value.
            continue
        by_name[name] = {"version": "0.1.0", "name": name, "expiration": expiration, "value": value}
    return list(by_name.values())
