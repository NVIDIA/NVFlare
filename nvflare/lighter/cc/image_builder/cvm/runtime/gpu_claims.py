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

"""Validate composite GPU claims in an authenticated attestation token."""

from ..common.errors import require
from ..common.gpu_claims import NRAS_CLAIMS, VECTOR


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
        require(all(nvidia.get(name) is True for name in NRAS_CLAIMS), "Non-NVIDIA GPU appraisal")
        identity = nvidia["ueid"]
        require(isinstance(identity, str) and identity and identity not in identities, "Duplicate GPU identity")
        identities.add(identity)
