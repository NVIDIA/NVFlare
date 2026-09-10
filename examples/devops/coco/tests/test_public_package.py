# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Regression tests for sanitized configuration and vendored dependency boundaries."""

import ast
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class PublicPackageTests(unittest.TestCase):
    def bash(self, command, *args):
        return subprocess.run(["bash", "-c", command, "test", *args], capture_output=True, text=True)

    def test_reject_placeholder_or_unsafe_service_dns(self):
        for host in (
            "secure-services.example.com",
            "example.net",
            "x.invalid",
            "x.test",
            "bad;name.local",
            "ok.local\nother",
            "-bad.local",
            "127.0.0.1",
            "ok.local.",
        ):
            with self.subTest(host=host):
                result = self.bash(
                    'source "$1"; validate_service_host "$2"', str(ROOT / "admin/lib/validate-config.sh"), host
                )
                self.assertNotEqual(result.returncode, 0)

    def test_allow_deployment_dns(self):
        result = self.bash(
            'source "$1"; validate_service_host "$2"', str(ROOT / "admin/lib/validate-config.sh"), "secure.unit.local"
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_target_host_requires_explicit_match(self):
        validator = str(ROOT / "admin/lib/validate-config.sh")
        for value in ("", "not-the-intended-host.invalid"):
            self.assertNotEqual(
                self.bash('source "$1"; EXPECTED_HOSTNAME=$2; validate_target_host', validator, value).returncode, 0
            )
        hostname = subprocess.check_output(["hostname", "-f"], text=True).strip()
        self.assertEqual(
            self.bash('source "$1"; EXPECTED_HOSTNAME=$2; validate_target_host', validator, hostname).returncode, 0
        )

    def test_all_roles_use_same_validator(self):
        expected = (ROOT / "admin/lib/validate-config.sh").read_bytes()
        for role in ("service", "coco", "trusted_system"):
            self.assertEqual((ROOT / role / "lib/validate-config.sh").read_bytes(), expected)

    def test_missing_private_configuration_fails_before_install(self):
        for role, helper in (("admin", "platform.sh"), ("service", "common.sh"), ("coco", "common.sh")):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temp:
                kit = Path(temp) / role
                shutil.copytree(ROOT / role / "lib", kit / "lib", ignore=shutil.ignore_patterns("__pycache__"))
                result = self.bash('source "$1"', str(kit / "lib" / helper))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("platform.env.example", result.stderr)
                self.assertEqual(sorted(p.name for p in kit.iterdir()), ["lib"])

    def test_templates_cannot_authorize_a_workload(self):
        admin = (ROOT / "admin/platform.env.example").read_text()
        service = (ROOT / "service/platform.env.example").read_text()
        self.assertIn('WORKLOAD_LAUNCH_PROFILE_SHA256=""', admin)
        for field in (
            "SNP_LAUNCH_MEASUREMENT",
            "SNP_MIN_REPORTED_TCB_BOOTLOADER",
            "SNP_MIN_REPORTED_TCB_TEE",
            "SNP_MIN_REPORTED_TCB_SNP",
            "SNP_MIN_REPORTED_TCB_MICROCODE",
        ):
            self.assertIn(f'{field}=""', service)

    def test_launch_shape_template_has_one_gpu_and_no_sizing_override(self):
        import yaml

        pod = yaml.safe_load((ROOT / "trusted_system/workload-source.yaml.example").read_text())
        spec = pod["spec"]
        self.assertEqual(spec["runtimeClassName"], "kata-qemu-nvidia-gpu-snp")
        self.assertEqual(len(spec["containers"]), 1)
        self.assertEqual(spec["containers"][0]["resources"], {"limits": {"nvidia.com/pgpu": 1}})
        self.assertFalse(spec["automountServiceAccountToken"])
        self.assertFalse(spec["enableServiceLinks"])

    def test_registry_match_is_configured_and_literal(self):
        script = (ROOT / "service/12-install-trusted-service-handoff.sh").read_text()
        block = re.search(r"<<'PY'\n(.*?)\nPY\n", script, re.S)[1]
        assignment = next(
            n
            for n in ast.parse(block).body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "image_match" for t in n.targets)
        )
        expression = compile(ast.Expression(assignment.value), "registry-check", "eval")
        registry = "secure.unit.local:5000"
        for host, accepted in (
            (registry, True),
            ("secureXunitXlocal:5000", False),
            ("other.unit.local:5000", False),
            ("secure.unit.local:5001", False),
        ):
            image = f"{host}/workloads/demo@sha256:" + "a" * 64
            match = eval(
                expression, {"re": re, "sys": SimpleNamespace(argv=["test", "handoff", registry]), "image": image}
            )
            self.assertEqual(bool(match), accepted)

    def test_cluster_has_only_vendored_bootstrap(self):
        entry = (ROOT / "coco/00-fetch-pinned-workflow.sh").read_text()
        self.assertNotIn("git clone", entry)
        for name in ("00-verify-host.sh", "10-install-kubernetes.sh", "20-install-coco-gpu.sh"):
            self.assertTrue((ROOT / "coco/bootstrap" / name).is_file())
        runtime = (ROOT / "coco/bootstrap/20-install-coco-gpu.sh").read_text()
        self.assertIn('"image.reference=$KATA_DEPLOY_AMD64"', runtime)
        self.assertIn('"$KATA_CHART_TGZ_SHA256" "$chart"', runtime)
        self.assertFalse((ROOT / "coco/34-verify-unpack-platform-bundle.sh").exists())

    def test_bootstrap_rejects_checksum_bypass_before_state_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.env"
            hostname = subprocess.check_output(["hostname", "-f"], text=True).strip()
            config.write_text(
                f'EXPECTED_HOSTNAME="{hostname}"\nTEE_PLATFORM=snp\n'
                "RUNTIME_CLASS=kata-qemu-nvidia-gpu-snp\nIGNORE_CHECKSUM_MISMATCH=1\n"
            )
            for role in ("trusted_system", "coco"):
                helper = str(ROOT / role / "bootstrap/lib/common.sh")
                result = self.bash(
                    'COCO_CONFIG=$1; COCO_STATE_DIR=$2; source "$3"; load_config',
                    str(config),
                    str(Path(temp) / "state"),
                    helper,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Checksum bypass", result.stderr)
                self.assertFalse((Path(temp) / "state").exists())


if __name__ == "__main__":
    unittest.main()
