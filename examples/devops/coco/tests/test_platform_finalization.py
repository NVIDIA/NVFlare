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

"""Offline negative tests for the actual finalizer blocks and TCB helper.

The snpguest stub tests control flow only, not AMD cryptography or hardware.
"""

import copy
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRUSTED = ROOT / "trusted_system"
BLOCKS = re.findall(r"<<'PY'\n(.*?)\nPY\n", (TRUSTED / "09-finalize-platform-reference.sh").read_text(), re.S)


class PlatformFinalizationTests(unittest.TestCase):
    def test_repeat_evidence_required_and_bound_to_first_run(self):
        block = next(b for b in BLOCKS if "first, repeat =" in b)
        for failure in (None, "missing", "stale", "nonce", "same_nonce", "launch", "measurement"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                first, repeat = (Path(temp) / name for name in ("first", "repeat"))
                (first / "evidence-input").mkdir(parents=True)
                repeat.mkdir()
                nonce = b"b" * 64
                initial = bytearray(0x4A0)
                initial[0x50:0x90] = b"a" * 64
                initial[0x90:0xC0] = b"m" * 48
                report = initial.copy()
                report[0x50:0x90] = nonce
                (first / "request-data.bin").write_bytes(b"a" * 64)
                (first / "evidence-input/attestation-report.bin").write_bytes(initial)
                (repeat / "request-data.bin").write_bytes(nonce)
                summary = {
                    "first_report_sha256": hashlib.sha256(initial).hexdigest(),
                    "repeat_report_sha256": hashlib.sha256(report).hexdigest(),
                    "nonce_sha256": hashlib.sha256(nonce).hexdigest(),
                    "result": "PASS: fresh signed reports and captured launch profile match",
                }
                if failure == "stale":
                    summary["first_report_sha256"] = "stale"
                if failure == "nonce":
                    report[0x50] ^= 1
                if failure == "same_nonce":
                    (first / "request-data.bin").write_bytes(nonce)
                if failure == "measurement":
                    report[0x90] ^= 1
                (repeat / "attestation-report.bin").write_bytes(report)
                if failure != "missing":
                    (repeat / "result.json").write_text(json.dumps(summary))
                for directory in (first, repeat):
                    (directory / "actual-launch.json").write_text(
                        json.dumps(
                            {
                                "launch_inputs": (
                                    "changed" if failure == "launch" and directory == repeat else "approved"
                                ),
                                "pod_resources": [{}],
                            }
                        )
                    )
                result = subprocess.run(
                    [sys.executable, "-", str(first), str(repeat)], input=block, text=True, capture_output=True
                )
                self.assertEqual(result.returncode == 0, failure is None, result.stderr)

    def test_limits_only_rehearsal_uses_approved_resource_defaults(self):
        import yaml

        script = (TRUSTED / "07-run-snp-rehearsal.sh").read_text()
        block = next(b for b in re.findall(r"<<'PY'\n(.*?)\nPY\n", script, re.S) if "pod_path, source_path" in b)
        with tempfile.TemporaryDirectory() as temp:
            pod, source = (Path(temp) / name for name in ("pod.yaml", "source.yaml"))
            data = {
                "spec": {
                    "runtimeClassName": "kata-qemu-nvidia-gpu-snp",
                    "containers": [{"resources": {"limits": {"nvidia.com/pgpu": "1"}}}],
                }
            }
            pod.write_text(yaml.safe_dump(data))
            source.write_text(yaml.safe_dump(data))
            result = subprocess.run(
                [sys.executable, "-", str(pod), str(source)], input=block, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            resources = yaml.safe_load(pod.read_text())["spec"]["containers"][0]["resources"]
            self.assertEqual(resources["limits"], resources["requests"])

    def test_actual_boot_artifacts_and_approved_workload(self):
        block = next(b for b in BLOCKS if "profile, actual" in b)
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            workload = work / "pod.yaml"
            workload.write_text("trusted source")
            artifacts = {key: {"sha256": key} for key in ("path", "firmware", "kernel", "initrd", "image")}
            profile = {
                "artifacts": artifacts,
                "pod_resources": {},
                "kata_config_sha256": "config",
                "workload_yaml_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
            }
            actual = {
                "artifacts": {
                    **{k: v for k, v in artifacts.items() if k != "path"},
                    **{"configured_" + k: v for k, v in artifacts.items()},
                    "qemu_executable": artifacts["path"],
                    "kata_config": {"sha256": "config"},
                },
                "pod_resources": [{}],
            }
            (work / "profile.json").write_text(json.dumps(profile))

            def check(data, success):
                (work / "actual.json").write_text(json.dumps(data))
                result = subprocess.run(
                    [sys.executable, "-", str(work / "profile.json"), str(work / "actual.json"), str(workload)],
                    input=block,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode == 0, success, result.stderr)

            check(actual, True)
            for key in ("qemu_executable", "firmware", "kernel", "initrd", "image"):
                with self.subTest(key=key):
                    changed = copy.deepcopy(actual)
                    changed["artifacts"][key]["sha256"] = "unapproved"
                    check(changed, False)
            for mode in ("initrd", "image"):
                changed = copy.deepcopy(actual)
                del changed["artifacts"][mode]
                check(changed, True)
            changed = copy.deepcopy(actual)
            del changed["artifacts"]["initrd"]
            del changed["artifacts"]["image"]
            check(changed, False)
            workload.write_text("changed after approval")
            check(actual, False)

    def test_actual_rootfs_drive(self):
        spec = importlib.util.spec_from_file_location("capture", TRUSTED / "capture-running-launch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.rootfs_image(["qemu", "-drive", "file=/actual.img,if=none,id=image-abc"]), "/actual.img"
        )
        self.assertIsNone(module.rootfs_image(["qemu", "-drive", "file=/workload.img,id=data"]))
        with self.assertRaises(ValueError):
            module.rootfs_image(["qemu", "-drive", "file=/a,id=image", "-drive", "file=/b,id=image-other"])

    def test_tcb_report_does_not_define_minimums(self):
        for reported, floors, success in [
            ((3, 0, 22, 62), (3, 0, 22, 62), True),
            ((4, 0, 23, 63), (3, 0, 22, 62), True),
            ((2, 0, 22, 62), (3, 0, 22, 62), False),
            ((3, 0, 21, 62), (3, 0, 22, 62), False),
            ((3, 0, 22, 61), (3, 0, 22, 62), False),
            ((3, 0, 22, 62), ("", 0, 22, 62), False),
            ((3, 0, 22, 62), (3, 0, 22, 256), False),
        ]:
            with self.subTest(reported=reported, floors=floors), tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                profile = work / "profile"
                profile.mkdir()
                binaries = work / "bin"
                binaries.mkdir()
                (binaries / "python3").symlink_to(sys.executable)
                guest = binaries / "snpguest"
                guest.write_text("#!/bin/sh\nexit 0\n")
                guest.chmod(0o700)
                config = work / "config.env"
                config.write_text(f"PLATFORM_WORK_ROOT={shlex.quote(str(work))}\nPLATFORM_PROFILE=profile\n")
                approval = profile / "approval.env"
                baseline = "".join(
                    f'SNP_MIN_REPORTED_TCB_{key}="{value}"\n'
                    for key, value in zip(("BOOTLOADER", "TEE", "SNP", "MICROCODE"), floors)
                )
                original = baseline + 'TCB_EVIDENCE_FILE=""\n'
                approval.write_text(original)
                report = bytearray(0x4A0)
                for offset, value in zip((0x180, 0x181, 0x186, 0x187), reported):
                    report[offset] = value
                (work / "report.bin").write_bytes(report)
                (work / "certs").mkdir()
                result = subprocess.run(
                    [
                        "bash",
                        str(TRUSTED / "record-reported-tcb.sh"),
                        str(config),
                        str(approval),
                        str(work / "report.bin"),
                        str(work / "certs"),
                    ],
                    env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"]},
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode == 0, success, result.stderr)
                self.assertTrue(approval.read_text().startswith(baseline))
                if not success:
                    self.assertEqual(approval.read_text(), original)


if __name__ == "__main__":
    unittest.main()
