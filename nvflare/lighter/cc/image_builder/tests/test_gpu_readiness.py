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

"""CUDA readiness follows successful appraisal and is revoked on failure."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from builder import gpu
from builder.common import BuildError


class GpuReadinessTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((Path(__file__).parent.parent / "config/gpu_policy.json").read_text())
        self.config = {
            "gpu": "nvidia_cc",
            "gpu_count": 1,
            "gpu_policy": "/etc/cvm/gpu_policy.json",
            "gpu_attestation_url": "https://nras.example.test",
        }
        self.result = {
            "result_code": 0,
            "detached_eat": [["JWT", "overall"], {"GPU-0": "detached"}],
            "claims": [{**copy.deepcopy(self.policy["required-claims"]), "eat_nonce": "ab" * 32}],
        }
        self.commands = []
        self.devices = "0000:01:00.0\n"
        self.attestation_error = None
        self.enable_error = False
        self.reset_error = False
        self.addCleanup(patch.stopall)
        patch.object(gpu.os, "urandom", return_value=bytes.fromhex("ab" * 32)).start()
        patch.object(
            gpu, "read_json", side_effect=lambda path: self.config if path.endswith("runtime.json") else self.policy
        ).start()
        patch.object(gpu, "run", side_effect=self.run_command).start()

    def run_command(self, command, **kwargs):
        self.commands.append(command)
        if "--query-gpu=pci.bus_id" in command:
            return self.devices
        if command[0] == "/usr/bin/nvattest":
            if self.attestation_error:
                raise self.attestation_error
            return json.dumps(self.result)
        if command == ["nvidia-smi", "conf-compute", "-srs", "1"] and self.enable_error:
            raise BuildError("Driver refused readiness")
        if command == ["nvidia-smi", "conf-compute", "-srs", "0"] and self.reset_error:
            raise BuildError("Driver refused readiness reset")
        return ""

    def test_success_enables_cuda_after_policy_validation(self):
        gpu.main()
        self.assertEqual(self.commands[-1], ["nvidia-smi", "conf-compute", "-srs", "1"])
        self.assertEqual(self.commands[-2][0], "/usr/bin/nvattest")
        self.assertNotIn(["nvidia-smi", "conf-compute", "-srs", "0"], self.commands)

    def test_failed_claims_nonce_or_device_count_never_enable_cuda(self):
        original = copy.deepcopy(self.result)
        for damage in ("claim", "nonce", "count", "signature", "result"):
            with self.subTest(damage=damage):
                self.commands.clear()
                self.result = copy.deepcopy(original)
                self.devices = "0000:01:00.0\n"
                if damage == "claim":
                    self.result["claims"][0]["secboot"] = False
                elif damage == "nonce":
                    self.result["claims"][0]["eat_nonce"] = "00" * 32
                elif damage == "count":
                    self.devices += "0000:02:00.0\n"
                elif damage == "signature":
                    self.result["detached_eat"] = None
                else:
                    self.result["result_code"] = 1
                with self.assertRaises(BuildError):
                    gpu.main()
                self.assertNotIn(["nvidia-smi", "conf-compute", "-srs", "1"], self.commands)
                self.assertEqual(self.commands[-1], ["nvidia-smi", "conf-compute", "-srs", "0"])

    def test_appraisal_failure_revokes_prior_readiness(self):
        self.attestation_error = BuildError("NRAS unavailable")
        with self.assertRaisesRegex(BuildError, "NRAS unavailable"):
            gpu.main()
        self.assertEqual(self.commands[-1], ["nvidia-smi", "conf-compute", "-srs", "0"])
        self.assertNotIn(["nvidia-smi", "conf-compute", "-srs", "1"], self.commands)

    def test_failed_readiness_command_is_not_success(self):
        self.enable_error = True
        with self.assertRaisesRegex(BuildError, "Driver refused readiness"):
            gpu.main()
        self.assertEqual(self.commands[-1], ["nvidia-smi", "conf-compute", "-srs", "0"])

    def test_failed_revocation_preserves_original_appraisal_error(self):
        self.attestation_error = BuildError("NRAS unavailable")
        self.reset_error = True
        with self.assertRaisesRegex(BuildError, "NRAS unavailable"):
            gpu.main()
