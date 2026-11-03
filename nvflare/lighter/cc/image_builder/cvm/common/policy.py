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

"""Generate CVM resource rules without replacing a shared CoCo policy."""

import json
import re

from .contracts import identifier
from .errors import require
from .gpu_policy import resource_conditions
from .measurements import validate_measurements

PRELUDE = """package policy
import rego.v1
default allow := false
cpu := input["submods"]["cpu0"]
ev := cpu["ear.veraison.annotated-evidence"]
tv := cpu["ear.trustworthiness-vector"]
fresh_token if {
    now := time.now_ns() / 1000000000
    is_number(input.iat)
    is_number(input.exp)
    input.iat <= now + 5
    input.iat >= now - 300
    input.exp > now
    input.exp > input.iat
    input.exp - input.iat <= 300
    nbf := object.get(input, "nbf", 0)
    is_number(nbf)
    nbf <= now + 5
}
approved_cpu(policy_id) if {
    fresh_token
    cpu["ear.appraisal-policy-id"] == policy_id
    cpu["ear.status"] == "affirming"
    tv["executables"] == 3
    tv["hardware"] == 2
    tv["configuration"] == 2
}
"""

MANAGED_BEGIN = "# BEGIN NVFLARE CVM MANAGED POLICY v1"
MANAGED_END = "# END NVFLARE CVM MANAGED POLICY v1"

FRAGMENT_PRELUDE = """cvm_cpu := input["submods"]["cpu0"]
cvm_ev := cvm_cpu["ear.veraison.annotated-evidence"]
cvm_tv := cvm_cpu["ear.trustworthiness-vector"]
cvm_fresh_token if {
    now := time.now_ns() / 1000000000
    is_number(input.iat)
    is_number(input.exp)
    input.iat <= now + 5
    input.iat >= now - 300
    input.exp > now
    input.exp > input.iat
    input.exp - input.iat <= 300
    nbf := object.get(input, "nbf", 0)
    is_number(nbf)
    nbf <= now + 5
}
cvm_approved_cpu(policy_id) if {
    cvm_fresh_token
    cvm_cpu["ear.appraisal-policy-id"] == policy_id
    cvm_cpu["ear.status"] == "affirming"
    cvm_tv["executables"] == 3
    cvm_tv["hardware"] == 2
    cvm_tv["configuration"] == 2
}
"""


def bundle_rule(manifest, prefix=""):
    platform = manifest["platform"]
    values = manifest["measurements"]
    validate_measurements(platform, values)
    build = json.dumps(identifier(manifest["build_id"]))
    policy = json.dumps(identifier(manifest["attestation_policy_id"]))
    cpu = prefix + "cpu"
    ev = prefix + "ev"
    approved = prefix + "approved_cpu"
    lines = ["allow if {", f"    {approved}({policy})", f'    is_string({ev}["init_data"])']
    contract = manifest.get("contract", {})
    if contract.get("token_issuer"):
        lines.append(f'    input.iss == {json.dumps(contract["token_issuer"])}')
    if contract.get("gpu") == "nvidia_cc":

        lines += resource_conditions(contract["gpu_count"], policy)
    if platform == "amd_sev_snp":
        lines += [
            f'    {ev}["snp"]["measurement"] == {json.dumps(values["snp.measurement"])}',
            f'    {ev}["snp"]["policy_debug_allowed"] == false',
            f'    {ev}["snp"]["policy_migrate_ma"] == false',
            f'    regex.match("^[0-9a-f]{{64}}$", {ev}["init_data"])',
            f'    binding_id := {ev}["init_data"]',
        ]
    else:
        lines += [
            f'    {ev}["tdx"]["quote"]["body"]["{key}"] == {json.dumps(values[key])}'
            for key in ("mr_td", "rtmr_0", "rtmr_1", "rtmr_2")
        ]
        lines += [
            f'    {ev}["tdx"]["td_attributes"]["debug"] == false',
            f'    regex.match("^[0-9a-f]{{64}}0{{32}}$", {ev}["init_data"])',
            f'    binding_id := {ev}["init_data"]',
        ]
    lines += [
        '    data.plugin == "resource"',
        f'    data["resource-path"] == ["keys", {build}, binding_id]',
        "}",
        "",
    ]
    return "\n".join(lines)


def compose(manifests):
    ids = [m["build_id"] for m in manifests]
    require(len(ids) == len(set(ids)), "Duplicate bundle identity")
    return PRELUDE + "\n" + "\n".join(bundle_rule(m) for m in sorted(manifests, key=lambda m: m["build_id"]))


def fragment(manifests):
    ids = [m["build_id"] for m in manifests]
    require(len(ids) == len(set(ids)), "Duplicate bundle identity")
    rules = "\n".join(bundle_rule(m, "cvm_") for m in sorted(manifests, key=lambda m: m["build_id"]))
    return MANAGED_BEGIN + "\n" + FRAGMENT_PRELUDE + "\n" + rules + MANAGED_END + "\n"


def merge(existing, manifests, previous=()):
    """Replace only the CVM-managed fragment in a shared global policy."""
    require(isinstance(existing, str) and "\x00" not in existing, "Invalid active resource policy")
    require(existing.count(MANAGED_BEGIN) == existing.count(MANAGED_END) <= 1, "Invalid CVM policy markers")
    replacement = None
    if MANAGED_BEGIN in existing:
        start = existing.index(MANAGED_BEGIN)
        require(existing.find(MANAGED_END) > start, "Invalid CVM policy marker order")
        end = existing.index(MANAGED_END, start) + len(MANAGED_END)
        if end < len(existing) and existing[end] == "\n":
            end += 1
        before, after = existing[:start], existing[end:]
        base = before + after
        replacement = (before, after)
    elif previous and existing.strip() == compose(previous).strip():
        # Upgrade an instance previously owned exclusively by CVM administration.
        base = "package policy\nimport rego.v1\ndefault allow := false\n"
    else:
        base = existing
    require(len(re.findall(r"^package policy\s*$", base, re.MULTILINE)) == 1, "Shared policy needs one package")
    require(len(re.findall(r"^import rego\.v1\s*$", base, re.MULTILINE)) == 1, "Shared policy needs Rego v1")
    require(
        len(re.findall(r"^default allow\s*:=\s*false\s*$", base, re.MULTILINE)) == 1,
        "Shared policy must be default-deny",
    )
    require(
        not re.search(r"\bcvm_(?:cpu|ev|tv|fresh_token|approved_cpu)\b", base),
        "Shared policy collides with the reserved CVM namespace",
    )
    managed = fragment(manifests)
    if replacement is not None:
        return replacement[0] + managed + replacement[1]
    separator = "" if base.endswith("\n\n") else ("\n" if base.endswith("\n") else "\n\n")
    return base + separator + managed
