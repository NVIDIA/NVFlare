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

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TRUSTED = Path(__file__).resolve().parents[2] / "trusted_system"


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.p = {
            "runtime_class": "kata-qemu-nvidia-gpu-snp",
            "pod_resources": {"limits": {"nvidia.com/pgpu": "1"}, "requests": {"nvidia.com/pgpu": "1"}},
            "kata_config_sha256": "a" * 64,
            "runtime_default_vcpus": 1,
            "runtime_default_memory_mib": 8192,
        }
        inputs = {"smp": "1,cores=1", "memory": "8192M"}
        self.a = {
            "pod_resources": [self.p["pod_resources"]],
            "artifacts": {"kata_config": {"sha256": "a" * 64}},
            "launch_inputs": inputs,
            "launch_inputs_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
        }
        (self.root / "rehearsal-collector-build").mkdir()
        self.pp = self.root / "approved-launch-profile.json"
        self.ap = self.root / "rehearsal-collector-build/actual-launch.json"
        self.write_inputs()

    def write_inputs(self):
        self.pp.write_text(json.dumps(self.p))
        self.ap.write_text(json.dumps(self.a))
        self.ph = hashlib.sha256(self.pp.read_bytes()).hexdigest()
        self.ah = hashlib.sha256(self.ap.read_bytes()).hexdigest()

    def command(self):
        return [
            sys.executable,
            str(TRUSTED / "export-workload-launch-profile.py"),
            str(self.root),
            "test-profile",
            "3.29.0",
            self.p["runtime_class"],
            "quay.io/kata-containers/kata-deploy@sha256:" + "b" * 64,
            self.ph,
            self.ah,
        ]

    def test_export_is_minimal(self):
        result = subprocess.run(self.command(), capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        self.assertEqual(data["pod_constraints"]["gpu_count"], 1)
        self.assertNotIn("cpu", data)
        self.assertNotIn("artifacts", data)
        self.assertNotIn("snp_launch_measurement", data)
        self.assertNotIn(str(self.root), result.stdout)

    def test_missing_finalization_binding(self):
        cmd = self.command()
        cmd[-1] = ""
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

    def test_changed_evidence(self):
        self.pp.write_text(self.pp.read_text() + " ")
        self.assertNotEqual(subprocess.run(self.command(), capture_output=True).returncode, 0)

    def test_changed_actual_defaults(self):
        self.p["runtime_default_vcpus"] = 4
        self.write_inputs()
        self.assertNotEqual(subprocess.run(self.command(), capture_output=True).returncode, 0)

    def test_unapproved_resource_shape(self):
        self.p["pod_resources"]["requests"]["cpu"] = "1"
        self.write_inputs()
        self.assertNotEqual(subprocess.run(self.command(), capture_output=True).returncode, 0)

    def test_invalid_export_produces_neither_output(self):
        # Stage 10 should validate admin evidence before writing even secure-services JSON.
        profile = self.root / "test-profile"
        profile.mkdir()
        env = profile / "platform-reference.final.env"
        env.write_text(
            f'PLATFORM_WORK_ROOT="{self.root}"\nPLATFORM_PROFILE=test-profile\n'
            "KATA_VERSION=3.29.0\nRUNTIME_CLASS=kata-qemu-nvidia-gpu-snp\n"
            "KATA_DEPLOY_AMD64=quay.io/kata@sha256:" + "b" * 64 + "\n"
        )
        services, admin = self.root / "services.json", self.root / "admin.json"
        result = subprocess.run(
            ["bash", str(TRUSTED / "10-export-platform-reference-values.sh"), str(env), str(services), str(admin)],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(services.exists())
        self.assertFalse(admin.exists())


if __name__ == "__main__":
    unittest.main()
