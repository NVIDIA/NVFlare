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

"""Shared approved-reference schemas and profile-scoped policy definitions."""

import hashlib
import json
import re

from .errors import require
from .io import canonical

SNP_LISTS = {"snp_bootloader", "snp_microcode", "snp_snp_svn", "snp_tee_svn"}


SNP_BOOLS = {"snp_smt_enabled", "snp_tsme_enabled", "snp_single_socket", "snp_smt_allowed"}


SNP_INTS = {"snp_guest_abi_major", "snp_guest_abi_minor"}


TDX_HEX = {"mr_seam": 96, "tcb_svn": 32, "xfam": 16}


GPU_NAMES = {"gpu_driver_versions", "gpu_vbios_versions"}


TCB_NAMES = GPU_NAMES | SNP_LISTS | SNP_BOOLS | SNP_INTS | set(TDX_HEX) | {"allowed_advisory_ids"}


MEASUREMENT_NAMES = {"snp_launch_measurement", "mr_td", "rtmr_0", "rtmr_1", "rtmr_2"}


REFERENCE_RECORD_PLACEHOLDER = "__CVM_PROFILE_REFERENCE_RECORD__"


# SEV-SNP guest policy bits (AMD SEV-SNP ABI specification, GUEST_POLICY).
SNP_POLICY_SMT = 1 << 16
SNP_POLICY_RESERVED = 1 << 17
SNP_POLICY_MIGRATE_MA = 1 << 18
SNP_POLICY_DEBUG = 1 << 19
SNP_POLICY_SINGLE_SOCKET = 1 << 20
SNP_POLICY_KNOWN = (1 << 21) - 1


# Trustee returns reference values without checking their metadata expiry. Keep
# every profile's values and deadlines in one uniquely named RVPS record so a
# CPU and GPU profile can safely share the same Trustee instance.
def reference_record_name(profile_version, contract):
    require(isinstance(profile_version, str) and profile_version, "Invalid profile version")
    require(isinstance(contract, dict), "Invalid profile contract")
    digest = hashlib.sha256(canonical({"profile_version": profile_version, "contract": contract})).hexdigest()
    return "cvm_profile_" + digest[:32]


def reference_rego(record=REFERENCE_RECORD_PLACEHOLDER):
    require(
        record == REFERENCE_RECORD_PLACEHOLDER
        or (isinstance(record, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", record)),
        "Invalid RVPS record",
    )
    encoded = json.dumps(record)
    return f"""profile_references := query_reference_value({encoded})

reference(name) := value if {{
    expiry := profile_references["expirations"][name]
    is_number(expiry)
    time.now_ns() < expiry * 1000000000
    value := profile_references["values"][name]
}}
"""


REFERENCE_REGO = reference_rego()


def render_reference_policy(policy, record):
    require(isinstance(policy, str), "Invalid appraisal policy")
    require(policy.count(REFERENCE_RECORD_PLACEHOLDER) == 1, "Appraisal policy must contain one RVPS placeholder")
    reference_rego(record)  # Validate before replacing the reviewed placeholder.
    return policy.replace(REFERENCE_RECORD_PLACEHOLDER, record)


def validate_references(values, platforms=(), *, finalized=False, gpu=False):

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


def snp_guest_policy(values):
    """Derive the SEV-SNP launch policy word from the approved configuration references.

    The AS policy compares the guest's reported policy fields with these same
    references, so the launcher must request exactly the approved values instead
    of a hard-coded constant. Debug and migration are never enabled.
    """
    for name in ("snp_smt_allowed", "snp_single_socket"):
        require(type(values.get(name)) is bool, "Approved SNP references must state " + name)
    for name in ("snp_guest_abi_major", "snp_guest_abi_minor"):
        require(
            type(values.get(name)) is int and 0 <= values[name] <= 255, "Approved SNP references must state " + name
        )
    policy = SNP_POLICY_RESERVED | (values["snp_guest_abi_major"] << 8) | values["snp_guest_abi_minor"]
    if values["snp_smt_allowed"]:
        policy |= SNP_POLICY_SMT
    if values["snp_single_socket"]:
        policy |= SNP_POLICY_SINGLE_SOCKET
    return policy


def validate_snp_policy(policy):
    """Accept only a measured launch policy without debug, migration or unknown bits."""
    require(type(policy) is int and 0 < policy <= SNP_POLICY_KNOWN, "Invalid SNP guest policy")
    require(policy & SNP_POLICY_RESERVED, "SNP guest policy must set the reserved bit")
    require(not policy & (SNP_POLICY_DEBUG | SNP_POLICY_MIGRATE_MA), "SNP guest policy enables debug or migration")
    return policy
