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

"""Composite GPU policy, guest EAR validation, and AS reference-value regressions."""

import copy
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import test_contracts
from builder.attestation import validate_token
from builder.common import BuildError
from builder.config import SOURCE
from builder.gpu_policy import render

TESTS = Path(__file__).resolve().parent
ENGINE = Path(os.environ.get("CVM_POLICY_EVAL", str(TESTS / "policy_engine/target/release/cvm-policy-eval")))


def gpu_submod(policy, index=0):
    return {
        "ear.appraisal-policy-id": policy,
        "ear.status": "affirming",
        "ear.trustworthiness-vector": {"executables": 3, "hardware": 2, "configuration": 2},
        "ear.veraison.annotated-evidence": {
            "nvidia": {
                "x-nvidia-overall-att-result": True,
                "x-nvidia-gpu-attestation-report-nonce-match": True,
                "x-nvidia-gpu-arch-check": True,
                "ueid": f"GPU-{index}",
            }
        },
    }


def invalid_submods(policy, count):
    good = {f"gpu{i}": gpu_submod(policy, i) for i in range(count)}
    yield {}
    yield {"gpu1": gpu_submod(policy)}
    yield dict(good, gpu_extra=gpu_submod(policy, 8))
    yield dict(good, **{f"gpu{count}": gpu_submod(policy, count)})
    for key, value in (
        ("ear.appraisal-policy-id", "other-profile"),
        ("ear.status", "contraindicated"),
        ("ear.trustworthiness-vector", {"executables": 3, "hardware": 2, "configuration": True}),
        ("ear.trustworthiness-vector", {"executables": 3, "hardware": 2}),
        ("ear.veraison.annotated-evidence", {"sample_device": {"verifier": "nras-v3"}}),
        (
            "ear.veraison.annotated-evidence",
            {"nvidia": {"verifier": "sample", "x-nvidia-device-type": "gpu", "ueid": "0"}},
        ),
    ):
        bad = copy.deepcopy(good)
        bad["gpu0"][key] = value
        yield bad
    if count > 1:
        bad = copy.deepcopy(good)
        bad["gpu1"] = copy.deepcopy(bad["gpu0"])
        yield bad


class CompositeGuestTests(unittest.TestCase):
    def test_signed_ear_rejects_each_incomplete_gpu_appraisal(self):
        fixture = test_contracts.TokenTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config.update(gpu="nvidia_cc", gpu_count=2)
        cpu = fixture.claims["submods"]["cpu0"]
        fixture.claims["submods"].update({f"gpu{i}": gpu_submod("cvm-test", i) for i in range(2)})
        validate_token(fixture.token(), fixture.config, bytes(32), now=1001)
        for submods in invalid_submods("cvm-test", 2):
            fixture.claims["submods"] = dict(submods, cpu0=cpu)
            with self.subTest(submods=submods), self.assertRaises(BuildError):
                validate_token(fixture.token(), fixture.config, bytes(32), now=1001)
            fixture.config["gpu"] = "none"
            validate_token(fixture.token(), fixture.config, bytes(32), now=1001)
            fixture.config["gpu"] = "nvidia_cc"


@unittest.skipUnless(ENGINE.is_file(), "Build the pinned Rego engine first")
class GpuAppraisalTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((SOURCE / "config/gpu_policy.json").read_text())
        # Keep the NRAS wire fixture independent of the policy being tested.
        self.nvidia = json.loads((TESTS / "fixtures/nras_gpu_v3.json").read_text())
        self.nvidia.update(
            {
                "x-nvidia-overall-att-result": True,
            }
        )
        self.refs = {"gpu_driver_versions": ["575.28"], "gpu_vbios_versions": ["96.00.AF.00.01"]}

    def evaluate(self, claims, refs, *, expiry=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "policy.rego").write_text(render(self.policy))
            (root / "input.json").write_text(json.dumps({"nvidia": claims}))
            expiry = {name: time.time() + 300 for name in refs} if expiry is None else expiry
            (root / "data.json").write_text(json.dumps({"reference": dict(refs, cvm_reference_expiry=expiry)}))
            result = subprocess.run(
                [
                    str(ENGINE),
                    str(root / "policy.rego"),
                    str(root / "input.json"),
                    str(root / "data.json"),
                    "data.policy.approved",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip() == "true"

    def test_every_required_claim_is_enforced(self):
        self.assertTrue(self.evaluate(self.nvidia, self.refs))
        for name in self.policy["required-claims"]:
            bad = copy.deepcopy(self.nvidia)
            del bad[name]
            with self.subTest(name=name):
                self.assertFalse(self.evaluate(bad, self.refs))
        for name in self.policy["claims-if-present"]:
            for bad_value in (False, 1, "true", None):
                bad = dict(self.nvidia, **{name: bad_value})
                self.assertFalse(self.evaluate(bad, self.refs))
        for bad in (
            dict(self.nvidia, **{"x-nvidia-gpu-arch-check": False}),
            dict(self.nvidia, **{"x-nvidia-gpu-attestation-report-nonce-match": False}),
            dict(self.nvidia, **{"x-nvidia-overall-att-result": False}),
        ):
            self.assertFalse(self.evaluate(bad, self.refs))

    def test_unapproved_driver_or_vbios_is_denied(self):
        self.assertFalse(self.evaluate(self.nvidia, {}))
        for name in self.refs:
            bad = dict(self.refs, **{name: ["unapproved"]})
            self.assertFalse(self.evaluate(self.nvidia, bad))

    def test_missing_or_expired_reference_deadlines_deny_appraisal(self):
        self.assertFalse(self.evaluate(self.nvidia, self.refs, expiry={}))
        for name in self.refs:
            expires = {key: time.time() + 300 for key in self.refs}
            expires[name] = time.time() - 1
            self.assertFalse(self.evaluate(self.nvidia, self.refs, expiry=expires))
