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

"""Optional real OPA checks, using the already pinned service CLI (no Python extras)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
OPA = os.environ.get("COCO_TEST_OPA") or shutil.which("opa")
pytestmark = pytest.mark.skipif(not OPA, reason="set COCO_TEST_OPA to the service-pinned OPA 1.8.0 executable")
FIELDS = ("bootloader", "tee", "snp", "microcode")


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [None, True, "255", [], {}, -1, 0])
def test_reported_tcb_must_be_numeric_and_meet_floor(tmp_path, field, value):
    source = ROOT / "examples/devops/coco/service/policies/default_cpu.rego"
    policy = tmp_path / "cpu.rego"
    # Stand in only for Trustee's builtin reference lookup; evaluate the actual policy.
    policy.write_text(source.read_text() + "\nquery_reference_value(name) := data.references[name]\n")
    references = {"snp_launch_measurement": ["approved"]}
    references.update({"snp_min_reported_tcb_" + name: 1 for name in FIELDS})
    data = tmp_path / "refs.json"
    data.write_text(json.dumps({"references": references}))
    snp = {"measurement": "approved", "policy_debug_allowed": False, "policy_migrate_ma": False}
    snp.update({"reported_tcb_" + name: 1 for name in FIELDS})

    def evaluate(evidence):
        result = subprocess.run(
            [
                OPA,
                "eval",
                "--format",
                "raw",
                "--stdin-input",
                "-d",
                str(policy),
                "-d",
                str(data),
                "data.policy.hardware",
            ],
            input=json.dumps({"snp": evidence}),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    assert evaluate(snp) == 2
    snp["reported_tcb_" + field] = value
    assert evaluate(snp) == 97
