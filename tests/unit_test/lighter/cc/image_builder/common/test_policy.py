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

"""Evaluate real policy fixtures using the exact pinned Trustee Rego engine."""

import base64
import copy
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from cvm.common.contracts import resource_path
from cvm.common.policy import compose

ENGINE = Path(
    os.environ.get("CVM_POLICY_EVAL", str(Path(__file__).parents[1] / "policy_engine/target/release/cvm-policy-eval"))
)


@unittest.skipUnless(ENGINE.is_file(), "Build tests/unit_test/lighter/cc/image_builder/policy_engine with Cargo first")
class ResourcePolicyTests(unittest.TestCase):
    def fixture(self, platform):
        digest = bytes.fromhex("fbff" * 16)
        measurements = (
            {"snp.measurement": bytes(48).hex()}
            if platform == "amd_sev_snp"
            else {"mr_td": "1" * 96, "rtmr_0": "0" * 96, "rtmr_1": "2" * 96, "rtmr_2": "3" * 96}
        )
        manifest = {
            "build_id": "bundle-1",
            "platform": platform,
            "measurements": measurements,
            "attestation_policy_id": "cvm-test",
        }
        evidence = {"init_data": digest.hex() if platform == "amd_sev_snp" else (digest + bytes(16)).hex()}
        if platform == "amd_sev_snp":
            evidence["snp"] = {
                "measurement": measurements["snp.measurement"],
                "policy_debug_allowed": False,
                "policy_migrate_ma": False,
            }
        else:
            evidence["tdx"] = {"quote": {"body": measurements.copy()}, "td_attributes": {"debug": False}}
        claims = {
            "iat": int(time.time()),
            "exp": int(time.time()) + 120,
            "submods": {
                "cpu0": {
                    "ear.appraisal-policy-id": "cvm-test",
                    "ear.status": "affirming",
                    "ear.trustworthiness-vector": {"executables": 3, "hardware": 2, "configuration": 2},
                    "ear.veraison.annotated-evidence": evidence,
                }
            },
        }
        path = "resource/" + resource_path("bundle-1", platform, digest)
        return manifest, claims, path

    def evaluate(self, manifest, claims, path):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "policy.rego").write_text(compose([manifest]))
            (root / "input.json").write_text(json.dumps(claims))
            (root / "data.json").write_text(
                json.dumps({"plugin": path.split("/")[0], "resource-path": path.split("/")[1:]})
            )
            result = subprocess.run(
                [
                    str(ENGINE),
                    str(root / "policy.rego"),
                    str(root / "input.json"),
                    str(root / "data.json"),
                    "data.policy.allow",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip() == "true"

    def test_positive_platform_claims(self):
        for platform in ("amd_sev_snp", "intel_tdx"):
            with self.subTest(platform=platform):
                self.assertTrue(self.evaluate(*self.fixture(platform)))

    def test_genuine_upstream_snp_ear_matches_hex_references_and_resource_path(self):
        fixture = json.loads((Path(__file__).parents[1] / "fixtures/snp_trustee_v022.json").read_text())
        encoded = fixture["token"].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        # Only refresh token timestamps for the policy's real-clock evaluation.
        # Signature validation at capture time is tested separately.
        claims.update(iat=int(time.time()), exp=int(time.time()) + 120)
        manifest = {
            "build_id": "upstream-snp-fixture",
            "platform": "amd_sev_snp",
            "measurements": {"snp.measurement": fixture["references"]["snp_launch_measurement"][0]},
            "attestation_policy_id": "default",
        }
        evidence = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
        path = "resource/" + resource_path(manifest["build_id"], manifest["platform"], bytes(32))
        self.assertTrue(self.evaluate(manifest, claims, path))
        evidence["init_data"] = base64.b64encode(bytes(32)).decode()
        self.assertFalse(self.evaluate(manifest, claims, path))

    def test_stale_future_and_overlong_tokens_are_denied(self):
        manifest, claims, path = self.fixture("intel_tdx")
        now = int(time.time())
        for fields in (
            {"iat": now - 301},
            {"iat": now + 60},
            {"exp": now - 1},
            {"exp": now + 301},
            {"nbf": now + 60},
            {"nbf": True},
            {"nbf": None},
            {"nbf": str(now)},
            {"iat": now + 4, "exp": now + 4},
            {"iat": now + 4, "exp": now + 3},
            {"iat": True},
            {"iat": "invalid"},
            {"exp": None},
        ):
            with self.subTest(fields=fields):
                self.assertFalse(self.evaluate(manifest, dict(claims, **fields), path))
        for name in ("iat", "exp"):
            missing = dict(claims)
            del missing[name]
            self.assertFalse(self.evaluate(manifest, missing, path))

    def test_cross_vault_and_bundle_denied(self):
        for platform in ("amd_sev_snp", "intel_tdx"):
            manifest, claims, path = self.fixture(platform)
            for other in (
                "resource/" + resource_path("bundle-1", platform, bytes(32)),
                path.replace("bundle-1", "bundle-2"),
                path + "/extra",
                path.replace("resource/keys", "keys"),
                path.replace("resource/keys", "resource/legacy"),
            ):
                with self.subTest(platform=platform, path=other):
                    self.assertFalse(self.evaluate(manifest, claims, other))

    def test_composite_gpu_rules_and_cpu_only_compatibility(self):
        from common.test_gpu_composite import gpu_submod, invalid_submods

        for platform in ("amd_sev_snp", "intel_tdx"):
            manifest, claims, path = self.fixture(platform)
            original = compose([manifest])
            manifest["contract"] = {"gpu": "none", "gpu_count": 2}
            self.assertEqual(compose([manifest]), original)
            for submods in invalid_submods("cvm-test", 2):
                cpu_only = copy.deepcopy(claims)
                cpu_only["submods"].update(submods)
                self.assertTrue(self.evaluate(manifest, cpu_only, path))
            manifest["contract"] = {"gpu": "nvidia_cc", "gpu_count": 2}
            good = copy.deepcopy(claims)
            good["submods"].update({f"gpu{i}": gpu_submod("cvm-test", i) for i in range(2)})
            self.assertTrue(self.evaluate(manifest, good, path))
            for submods in invalid_submods("cvm-test", 2):
                bad = copy.deepcopy(claims)
                bad["submods"].update(submods)
                with self.subTest(platform=platform, submods=submods):
                    self.assertFalse(self.evaluate(manifest, bad, path))

    def test_negative_or_incomplete_appraisal_denied(self):
        for platform in ("amd_sev_snp", "intel_tdx"):
            manifest, original, path = self.fixture(platform)
            for field, value in (
                ("ear.status", "contraindicated"),
                ("ear.appraisal-policy-id", "default"),
                ("ear.trustworthiness-vector", {"executables": 3, "hardware": 2}),
                ("ear.trustworthiness-vector", {"executables": 3, "hardware": 2, "configuration": True}),
            ):
                claims = copy.deepcopy(original)
                claims["submods"]["cpu0"][field] = value
                with self.subTest(platform=platform, field=field):
                    self.assertFalse(self.evaluate(manifest, claims, path))

    def test_bad_binding_encodings_denied(self):
        for platform in ("amd_sev_snp", "intel_tdx"):
            manifest, original, path = self.fixture(platform)
            init = original["submods"]["cpu0"]["ear.veraison.annotated-evidence"]["init_data"]
            invalid = [
                None,
                1,
                [],
                init[:-1],
                init + "=",
                init.upper(),
                base64.b64encode(bytes.fromhex(init)).decode(),
            ]
            invalid.append(init[:-1] + "1")
            for value in invalid:
                claims = copy.deepcopy(original)
                claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]["init_data"] = value
                with self.subTest(platform=platform, value=value):
                    self.assertFalse(self.evaluate(manifest, claims, path))

    def test_debug_migration_and_measurement_denied(self):
        for platform in ("amd_sev_snp", "intel_tdx"):
            manifest, original, path = self.fixture(platform)
            claims = copy.deepcopy(original)
            ev = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
            if platform == "amd_sev_snp":
                ev["snp"]["policy_debug_allowed"] = True
            else:
                ev["tdx"]["td_attributes"]["debug"] = True
            self.assertFalse(self.evaluate(manifest, claims, path))
            claims = copy.deepcopy(original)
            ev = claims["submods"]["cpu0"]["ear.veraison.annotated-evidence"]
            if platform == "amd_sev_snp":
                ev["snp"]["policy_migrate_ma"] = True
            else:
                ev["tdx"]["quote"]["body"]["rtmr_2"] = "f" * 96
            self.assertFalse(self.evaluate(manifest, claims, path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
