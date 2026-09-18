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

"""Generate pure resource-authorization rules for approved CVM bundles."""

import json

from .contracts import identifier
from .errors import require
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
    object.get(input, "nbf", 0) <= now + 5
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


def bundle_rule(manifest):
    platform = manifest["platform"]
    values = manifest["measurements"]
    validate_measurements(platform, values)
    build = json.dumps(identifier(manifest["build_id"]))
    policy = json.dumps(identifier(manifest["attestation_policy_id"]))
    lines = ["allow if {", f"    approved_cpu({policy})", '    is_string(ev["init_data"])']
    contract = manifest.get("contract", {})
    if contract.get("token_issuer"):
        lines.append(f'    input.iss == {json.dumps(contract["token_issuer"])}')
    if contract.get("gpu") == "nvidia_cc":
        from .gpu_policy import resource_conditions

        lines += resource_conditions(contract["gpu_count"], policy)
    if platform == "amd_sev_snp":
        lines += [
            f'    ev["snp"]["measurement"] == {json.dumps(values["snp.measurement"])}',
            '    ev["snp"]["policy_debug_allowed"] == false',
            '    ev["snp"]["policy_migrate_ma"] == false',
            '    regex.match("^[A-Za-z0-9+/]{42}[AEIMQUYcgkosw048]=$", ev["init_data"])',
            '    binding_id := replace(replace(trim_suffix(ev["init_data"], "="), "+", "-"), "/", "_")',
        ]
    else:
        lines += [
            f'    ev["tdx"]["quote"]["body"]["{key}"] == {json.dumps(values[key])}'
            for key in ("mr_td", "rtmr_0", "rtmr_1", "rtmr_2")
        ]
        lines += [
            '    ev["tdx"]["td_attributes"]["debug"] == false',
            '    regex.match("^[0-9a-f]{64}0{32}$", ev["init_data"])',
            '    binding_id := ev["init_data"]',
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
