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

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ADMIN = Path(__file__).resolve().parents[1]
HELPER = ADMIN / "lib/workload-launch-profile.py"
spec = importlib.util.spec_from_file_location("launch_profile", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def profile_fixture():
    return {
        "schema": "coco-approved-workload-launch/v1",
        "profile_id": "test-profile",
        "runtime_class": "kata-qemu-nvidia-gpu-snp",
        "kata_version": "3.29.0",
        "kata_deploy_image": "quay.io/kata-containers/kata-deploy@sha256:" + "a" * 64,
        "kata_config_sha256": "b" * 64,
        "launch_inputs_sha256": "c" * 64,
        "vm_defaults": {"vcpus": 1, "memory_mib": 8192},
        "pod_constraints": {
            "container_count": 1,
            "gpu_resource": "nvidia.com/pgpu",
            "gpu_count": 1,
            "cpu_memory_resources": "omitted",
            "host_namespaces": False,
            "allowed_annotations": ["io.katacontainers.config.hypervisor.cc_init_data"],
        },
    }


def pod_fixture():
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": "test"},
        "spec": {
            "runtimeClassName": "kata-qemu-nvidia-gpu-snp",
            "containers": [
                {
                    "name": "test",
                    "image": "registry/test@sha256:" + "e" * 64,
                    "command": ["/app"],
                    "securityContext": {"privileged": False},
                    "resources": {"limits": {"nvidia.com/pgpu": "1"}},
                }
            ],
        },
    }


class LaunchProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "profile.json"
        self.p = profile_fixture()
        self.write_profile()

    def write_profile(self, raw=None):
        self.path.write_text(raw if raw is not None else json.dumps(self.p))
        self.pin = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def load(self, **kwargs):
        return module.load_profile(
            self.path,
            kwargs.get("pin", self.pin),
            kwargs.get("runtime", self.p["runtime_class"]),
            kwargs.get("version", "3.29.0"),
        )

    def test_valid_profile_and_pod(self):
        module.validate_pod(self.load(), pod_fixture())

    def test_missing_profile(self):
        self.path.unlink()
        with self.assertRaises(OSError):
            self.load()

    def test_bad_pin(self):
        for pin in ("", "bad", "0" * 64):
            with self.subTest(pin=pin), self.assertRaises(ValueError):
                self.load(pin=pin)

    def test_modified_profile(self):
        self.path.write_text(self.path.read_text() + " ")
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_keys(self):
        self.write_profile(self.path.read_text()[:-1] + ',"schema":"coco-approved-workload-launch/v1"}')
        with self.assertRaises(ValueError):
            self.load()

    def test_wrong_runtime_or_version(self):
        for args in ({"runtime": "runc"}, {"version": "3.30.0"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.load(**args)

    def test_unsupported_constraint(self):
        self.p["pod_constraints"]["container_count"] = 2
        self.write_profile()
        with self.assertRaises(ValueError):
            self.load()

    def test_defaulted_gpu_request(self):
        pod = pod_fixture()
        pod["spec"]["containers"][0]["resources"]["requests"] = {"nvidia.com/pgpu": 1}
        module.validate_pod(self.load(), pod)

    def test_reject_resource_changes(self):
        cases = [
            {"limits": {"nvidia.com/pgpu": 2}},
            {"limits": {"nvidia.com/pgpu": True}},
            {"limits": {"nvidia.com/pgpu": 1, "cpu": "1"}},
            {"limits": {"nvidia.com/pgpu": 1}, "requests": {"memory": "8Gi"}},
            {"limits": {"nvidia.com/gpu": 1}},
            {},
            {"limits": {"nvidia.com/pgpu": 1}, "claims": []},
        ]
        for resources in cases:
            pod = pod_fixture()
            pod["spec"]["containers"][0]["resources"] = resources
            with self.subTest(resources=resources), self.assertRaises(ValueError):
                module.validate_pod(self.p, pod)

    def test_reject_extra_containers(self):
        pod = pod_fixture()
        pod["spec"]["containers"] *= 2
        with self.assertRaises(ValueError):
            module.validate_pod(self.p, pod)

    def test_reject_spec_overrides(self):
        for field, value in [
            ("hostNetwork", True),
            ("hostPID", True),
            ("hostIPC", True),
            ("initContainers", []),
            ("ephemeralContainers", []),
            ("overhead", {"memory": "1Gi"}),
            ("volumes", []),
            ("runtimeClassName", "runc"),
        ]:
            pod = pod_fixture()
            pod["spec"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_pod(self.p, pod)

    def test_reject_runtime_annotation(self):
        pod = pod_fixture()
        pod["metadata"]["annotations"] = {"io.katacontainers.config.hypervisor.default_vcpus": "8"}
        with self.assertRaises(ValueError):
            module.validate_pod(self.p, pod)

    def test_allow_generated_initdata(self):
        pod = pod_fixture()
        pod["metadata"]["annotations"] = {"io.katacontainers.config.hypervisor.cc_init_data": "test"}
        module.validate_pod(self.p, pod)  # Init-data contents are checked by the existing generation logic.

    def test_reject_privileged(self):
        pod = pod_fixture()
        pod["spec"]["containers"][0]["securityContext"]["privileged"] = True
        with self.assertRaises(ValueError):
            module.validate_pod(self.p, pod)

    def test_cli_snapshot_is_exclusive(self):
        snapshot = Path(self.tmp.name) / "snapshot.json"
        cmd = [
            sys.executable,
            str(HELPER),
            str(self.path),
            self.pin,
            self.p["runtime_class"],
            "3.29.0",
            "--snapshot",
            str(snapshot),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        self.assertEqual(snapshot.read_bytes(), self.path.read_bytes())
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

    def test_cli_bad_pod_does_not_write_snapshot(self):
        pod = pod_fixture()
        pod["spec"]["containers"] *= 2
        pp = Path(self.tmp.name) / "pod.json"
        pp.write_text(json.dumps(pod))
        snapshot = Path(self.tmp.name) / "snapshot.json"
        cmd = [
            sys.executable,
            str(HELPER),
            str(self.path),
            self.pin,
            self.p["runtime_class"],
            "3.29.0",
            "--pod",
            str(pp),
            "--snapshot",
            str(snapshot),
        ]
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        self.assertFalse(snapshot.exists())

    def test_generator_and_packager_gate_before_external_actions(self):
        text = (ADMIN / "30-generate-pod-and-policies.sh").read_text()
        self.assertLess(text.index('check_launch_profile\n"${GENPOLICY}"'), text.index('"${GENPOLICY}" \\\n'))
        self.assertIn("check_launch_profile\nEXPECTED_INITDATA_HEX=", text)
        pack = (ADMIN / "40-create-handoffs.sh").read_text()
        self.assertLess(pack.index('"${WORKLOAD_PROFILE_VALIDATOR}"'), pack.index("mkdir -p"))

    def test_real_generator_fails_closed_before_tool_or_key_use(self):
        kit = Path(self.tmp.name) / "kit"
        (kit / "lib").mkdir(parents=True)
        (kit / "public").mkdir()
        for name in (
            "30-generate-pod-and-policies.sh",
            "lib/platform.sh",
            "lib/validate-config.sh",
            "lib/release.sh",
            "lib/workload-launch-profile.py",
        ):
            shutil.copyfile(ADMIN / name, kit / name)
        platform = (ADMIN / "platform.env.example").read_text()
        host = subprocess.check_output(["hostname", "-f"], text=True).strip()
        platform = platform.replace('EXPECTED_HOSTNAME=""', f'EXPECTED_HOSTNAME="{host}"')
        platform = platform.replace("secure-services.example.com", "secure.unit.local")
        platform = platform.replace(
            'WORKLOAD_LAUNCH_PROFILE_SHA256=""', 'WORKLOAD_LAUNCH_PROFILE_SHA256="' + "f" * 64 + '"'
        )
        platform = platform.replace('WORK_ROOT="${HOME}/coco-workload-owner"', 'WORK_ROOT="${HOME}/isolated-work"')
        (kit / "platform.env").write_text(platform)
        for name in ("trustee.crt", "registry-ca.crt"):
            (kit / "public" / name).write_text("test-only placeholder; no TLS operations occur")
        dockerfile = Path(self.tmp.name) / "Dockerfile"
        dockerfile.write_text("FROM scratch\n")
        owner = Path(self.tmp.name) / "workload.env"
        owner.write_text(
            f'RELEASE_NAME=profile-test\nBUILD_CONTEXT="{self.tmp.name}"\nDOCKERFILE="{dockerfile}"\n'
            "REGISTRY_REPOSITORY=test/image\nAPP_COMMAND_JSON='[\"/app\"]'\nAPP_UID=1000\nAPP_GID=1000\n"
        )
        env = dict(
            os.environ, HOME=self.tmp.name, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
        )
        cmd = ["bash", str(kit / "30-generate-pod-and-policies.sh"), str(owner)]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("approved-workload-launch-profile.json", result.stderr)
        shutil.copyfile(self.path, kit / "public/approved-workload-launch-profile.json")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256 mismatch", result.stderr)
        self.assertFalse((Path(self.tmp.name) / "isolated-work/releases/profile-test/output").exists())


if __name__ == "__main__":
    unittest.main()
