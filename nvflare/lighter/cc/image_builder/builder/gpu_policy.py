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

"""Render AS appraisal and KBS resource conditions for composite NVIDIA evidence."""

import json

from .common import require

VECTOR = {"executables": 3, "hardware": 2, "configuration": 2}
VERIFIER = "nras-v3"


def validate_submods(submods, count, policy):
    require(type(count) is int and 1 <= count <= 8, "Invalid GPU count")
    expected = {f"gpu{i}" for i in range(count)}
    require({name for name in submods if name.startswith("gpu")} == expected, "GPU appraisal count mismatch")
    identities = set()
    for name in sorted(expected):
        item = submods[name]
        require(item["ear.appraisal-policy-id"] == policy, "Wrong GPU appraisal policy")
        require(item["ear.status"] == "affirming", "GPU appraisal denied")
        vector = item["ear.trustworthiness-vector"]
        require(
            all(type(vector.get(k)) is int and vector[k] == v for k, v in VECTOR.items()), "GPU trust vector denied"
        )
        nvidia = item["ear.veraison.annotated-evidence"]["nvidia"]
        require(nvidia["verifier"] == VERIFIER and nvidia["x-nvidia-device-type"] == "gpu", "Non-NVIDIA GPU appraisal")
        identity = nvidia["ueid"]
        require(isinstance(identity, str) and identity and identity not in identities, "Duplicate GPU identity")
        identities.add(identity)


def resource_conditions(count, policy):
    require(type(count) is int and 1 <= count <= 8, "Invalid GPU count")
    lines = [f'    count({{name | input["submods"][name]; startswith(name, "gpu")}}) == {count}']
    for i in range(count):
        item = f'input["submods"]["gpu{i}"]'
        nvidia = item + '["ear.veraison.annotated-evidence"]["nvidia"]'
        lines += [
            f'    {item}["ear.appraisal-policy-id"] == {policy}',
            f'    {item}["ear.status"] == "affirming"',
            *[f'    {item}["ear.trustworthiness-vector"]["{key}"] == {value}' for key, value in VECTOR.items()],
            f'    {nvidia}["verifier"] == "{VERIFIER}"',
            f'    {nvidia}["x-nvidia-device-type"] == "gpu"',
            f'    is_string({nvidia}["ueid"])',
            f'    {nvidia}["ueid"] != ""',
        ]
        for j in range(i):
            other = f'input["submods"]["gpu{j}"]["ear.veraison.annotated-evidence"]["nvidia"]["ueid"]'
            lines.append(f'    {nvidia}["ueid"] != {other}')
    return lines


def render(policy):
    """Only a completely matching, backend-verified NVIDIA claim set affirms."""
    lines = [
        "# Generated from the profile's strict gpu_policy.json. Do not edit.",
        "package policy",
        "import rego.v1",
        "",
        "default approved := false",
        "default executables := 33",
        "default hardware := 97",
        "default configuration := 36",
        "executables := 3 if approved",
        "hardware := 2 if approved",
        "configuration := 2 if approved",
        "approved if {",
        "    n := input.nvidia",
        f'    n.verifier == "{VERIFIER}"',
        '    n.arch in {"HOPPER", "BLACKWELL"}',
        '    n["x-nvidia-overall-att-result"] == true',
        '    n["x-nvidia-ver"] == "3.0"',
        '    n["x-nvidia-gpu-driver-version"] in data.reference.gpu_driver_versions',
        '    n["x-nvidia-gpu-vbios-version"] in data.reference.gpu_vbios_versions',
    ]

    def constraints(values, path):
        for key, value in sorted(values.items()):
            child = path + "[" + json.dumps(key) + "]"
            if isinstance(value, dict):
                constraints(value, child)
            else:
                lines.append(f"    {child} == {json.dumps(value)}")

    constraints(policy["required-claims"], "n")
    # An absent optional claim is permitted, but a supplied false value denies.
    for key, value in sorted(policy["claims-if-present"].items()):
        lines.append(f"    object.get(n, {json.dumps(key)}, {json.dumps(value)}) == {json.dumps(value)}")
    return "\n".join(lines + ["}", ""])
