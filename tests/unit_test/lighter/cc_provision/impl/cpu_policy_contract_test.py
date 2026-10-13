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


def test_cpu_policy_emits_exact_authorizer_trust_vector():
    policy = (ROOT / "examples/devops/coco/service/policies/default_cpu.rego").read_text()
    match = re.search(r"(?ms)^trust_claims := \{\n(?P<body>.*?)^\}", policy)
    assert match
    claims = dict(re.findall(r'^\s*"([^"]+)":\s*([a-z][a-z0-9_-]*),?$', match.group("body"), re.MULTILINE))

    assert claims == {name: name for name in TRUST_VECTOR}
