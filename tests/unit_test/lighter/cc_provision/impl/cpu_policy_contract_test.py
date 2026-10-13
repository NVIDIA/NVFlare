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

"""Keep the shipped Trustee CPU policy aligned with NVFlare's EAR contract."""

import re
from pathlib import Path

from nvflare.app_opt.confidential_computing.trustee_claims import TRUST_VECTOR

ROOT = Path(__file__).resolve().parents[5]


def _trust_claims(policy):
    match = re.search(r"(?ms)^trust_claims := \{\n(?P<body>.*?)^\}", policy)
    assert match
    return re.findall(r'^\s*"([^"]+)":\s*([^,\n]+),?\s*$', match.group("body"), re.MULTILINE)


def test_cpu_policy_emits_exact_authorizer_trust_vector():
    policy = (ROOT / "examples/devops/coco/service/policies/default_cpu.rego").read_text()
    entries = _trust_claims(policy)
    names = [name for name, _ in entries]

    assert len(names) == len(set(names))
    assert set(names) == set(TRUST_VECTOR)
    assert dict(entries) == {name: name for name in TRUST_VECTOR}


def test_trust_claim_parser_includes_numeric_padding():
    assert _trust_claims('trust_claims := {\n    "storage-opaque": 0,\n}\n') == [("storage-opaque", "0")]
