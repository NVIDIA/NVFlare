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

"""Test the provision wrapper with mock stages; no Docker/registry access."""

import json
import os
import pty
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ProvisionScriptTests(unittest.TestCase):
    def test_pod_generator_binds_explicit_read_only_choice(self):
        script = (ROOT / "admin/30-generate-pod-and-policies.sh").read_text()
        block = next(b for b in re.findall(r"<<'PY'\n(.*?)\nPY\n", script, re.S) if '"kind": "Pod"' in b)
        with tempfile.TemporaryDirectory() as temp:
            pod = Path(temp) / "pod.yaml"
            for value in ("true", "false", "invalid"):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-",
                        str(pod),
                        "demo",
                        "kata-qemu-nvidia-gpu-snp",
                        "secure.unit.local:5000/demo@sha256:" + "a" * 64,
                        '["/opt/nvflare/startup/sub_start.sh","--once","--verify"]',
                        "65532",
                        "65532",
                        "10.96.0.1",
                        "443",
                        value,
                    ],
                    input=block,
                    text=True,
                    capture_output=True,
                )
                if value == "invalid":
                    self.assertNotEqual(result.returncode, 0)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    data = yaml.safe_load(pod.read_text())["spec"]
                    self.assertEqual(
                        data["containers"][0]["securityContext"]["readOnlyRootFilesystem"], value == "true"
                    )
                    self.assertNotIn("volumes", data)

    def run_wrapper(self, approval="demo", fail_stage=None):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            admin = work / "admin"
            (admin / "lib").mkdir(parents=True)
            handoff = work / "handoff"
            trace = work / "trace"
            (admin / "lib/release.sh").write_text(
                'die() { echo "$*" >&2; exit 1; }\nRELEASE_NAME=demo\n' + f"HANDOFF_DIR={shlex.quote(str(handoff))}\n"
            )
            stages = [
                "10-build-plaintext.sh",
                "20-encrypt-sign-publish.sh",
                "25-verify-published-image.sh",
                "30-generate-pod-and-policies.sh",
                "40-create-handoffs.sh",
            ]
            for stage in stages:
                code = f"#!/bin/bash\nset -eu\nprintf '%s\\n' {shlex.quote(stage)} >> {shlex.quote(str(trace))}\n"
                if stage == fail_stage:
                    code += "exit 7\n"
                if stage.startswith("20"):
                    code += '[[ "$2" == --approve-reviewed-plaintext ]]\n'
                if stage.startswith("40"):
                    code += f"mkdir -p {shlex.quote(str(handoff / 'coco-it'))} {shlex.quote(str(handoff / 'trusted-service'))}\n"
                    code += f"printf fixture > {shlex.quote(str(handoff / 'coco-it/demo-pod.yaml'))}\n"
                    code += f"printf fixture > {shlex.quote(str(handoff / 'trusted-service/image_key'))}\n"
                (admin / stage).write_text(code)
                (admin / stage).chmod(0o700)
            request = work / "request.json"
            result_file = work / "result.json"
            request.write_text(
                json.dumps(
                    {
                        "schema": "nvflare-coco-build-request/v1",
                        "workload_env": str(work / "workload.env"),
                        "admin_dir": str(admin),
                        "result_file": str(result_file),
                    }
                )
            )
            master, slave = pty.openpty()
            try:
                process = subprocess.Popen(
                    ["bash", str(ROOT / "admin/build_coco_image.sh"), str(request)],
                    stdin=slave,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                os.close(slave)
                slave = None
                os.write(master, (approval + "\n").encode())
                stdout, stderr = process.communicate(timeout=15)
            finally:
                os.close(master)
                if slave is not None:
                    os.close(slave)
            executed = trace.read_text().splitlines() if trace.exists() else []
            receipt = json.loads(result_file.read_text()) if result_file.exists() else None
            return process.returncode, stdout + stderr, executed, receipt

    def test_pipeline_order_and_success_receipt(self):
        status, output, stages, receipt = self.run_wrapper()
        self.assertEqual(status, 0, output)
        self.assertEqual([s[:2] for s in stages], ["10", "20", "25", "30", "40"])
        self.assertEqual(receipt["schema"], "nvflare-coco-build-result/v1")
        self.assertEqual(receipt["release_name"], "demo")
        self.assertTrue(receipt["trusted_service"].endswith("/trusted-service"))

    def test_wrong_approval_never_publishes(self):
        status, _, stages, receipt = self.run_wrapper(approval="no")
        self.assertNotEqual(status, 0)
        self.assertEqual([s[:2] for s in stages], ["10"])
        self.assertIsNone(receipt)

    def test_pipeline_stops_on_publish_failure(self):
        status, _, stages, receipt = self.run_wrapper(fail_stage="20-encrypt-sign-publish.sh")
        self.assertEqual(status, 7)
        self.assertEqual([s[:2] for s in stages], ["10", "20"])
        self.assertIsNone(receipt)


if __name__ == "__main__":
    unittest.main()
