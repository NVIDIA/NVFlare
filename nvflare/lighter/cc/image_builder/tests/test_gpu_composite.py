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
import unittest
from pathlib import Path

import test_contracts
from builder.attestation import validate_token
from builder.common import BuildError
from builder.gpu_policy import render

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("CVM_POLICY_EVAL", str(ROOT / "tests/policy_engine/target/release/cvm-policy-eval")))


def gpu_submod(policy, index=0):
    return {
        "ear.appraisal-policy-id": policy,
        "ear.status": "affirming",
        "ear.trustworthiness-vector": {"executables": 3, "hardware": 2, "configuration": 2},
        "ear.veraison.annotated-evidence": {
            "nvidia": {
                "verifier": "nras-v3",
                "x-nvidia-device-type": "gpu",
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
        self.policy = json.loads((ROOT / "config/gpu_policy.json").read_text())
        self.nvidia = copy.deepcopy(self.policy["required-claims"])
        self.nvidia.update(
            {
                "verifier": "nras-v3",
                "arch": "HOPPER",
                "x-nvidia-ver": "3.0",
                "x-nvidia-overall-att-result": True,
                "x-nvidia-gpu-driver-version": "fixture-driver",
                "x-nvidia-gpu-vbios-version": "fixture-vbios",
            }
        )
        self.refs = {"gpu_driver_versions": ["fixture-driver"], "gpu_vbios_versions": ["fixture-vbios"]}

    def evaluate(self, claims, refs):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "policy.rego").write_text(render(self.policy))
            (root / "input.json").write_text(json.dumps({"nvidia": claims}))
            (root / "data.json").write_text(json.dumps({"reference": refs}))
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
            dict(self.nvidia, arch="AMPERE"),
            dict(self.nvidia, verifier="sample"),
            dict(self.nvidia, **{"x-nvidia-overall-att-result": False}),
        ):
            self.assertFalse(self.evaluate(bad, self.refs))

    def test_unapproved_driver_or_vbios_is_denied(self):
        self.assertFalse(self.evaluate(self.nvidia, {}))
        for name in self.refs:
            bad = dict(self.refs, **{name: ["unapproved"]})
            self.assertFalse(self.evaluate(self.nvidia, bad))
