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

"""CUDA readiness follows composite appraisal and failure powers off the guest."""

import contextlib
import unittest
from unittest.mock import patch

from cvm.common.errors import BuildError
from cvm.runtime import bootstrap as runtime
from cvm.runtime import gpu


class GpuReadinessTests(unittest.TestCase):
    def test_readiness_requires_expected_device_count(self):
        cfg = {"gpu": "nvidia_cc", "gpu_count": 1}
        with patch.object(gpu, "run", return_value="GPU-0\n") as command:
            gpu.readiness(cfg, True)
            self.assertEqual(command.call_args.args[0], ["nvidia-smi", "conf-compute", "-srs", "1"])
        with patch.object(gpu, "run", return_value="GPU-0\nGPU-1\n") as command:
            with self.assertRaises(BuildError):
                gpu.readiness(cfg, True)
            self.assertEqual(command.call_count, 1)
        with patch.object(gpu, "run") as command:
            gpu.readiness(cfg, False)
            command.assert_called_once_with(["nvidia-smi", "conf-compute", "-srs", "0"], timeout=10)

    def test_not_ready_readback_requires_complete_expected_inventory(self):
        for devices in (b"0000:01:00.0\n", b"00000000:01:00.0\n", b"0000:0A:00.0\n00000000:0b:00.0\n"):
            with (
                self.subTest(devices=devices),
                patch.object(
                    gpu, "run", side_effect=[devices, b"Confidential Compute GPUs Ready state: not-ready\n"]
                ) as command,
            ):
                gpu.confirm_not_ready({"gpu_count": len(devices.splitlines())})
                self.assertEqual(command.call_count, 2)
                self.assertEqual(
                    command.call_args_list[0].args[0], ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"]
                )
                self.assertEqual(command.call_args_list[1].args[0], ["nvidia-smi", "conf-compute", "-grs"])
                self.assertTrue(all(call.kwargs == {"timeout": 10} for call in command.call_args_list))

    def test_not_ready_readback_rejects_invalid_counts_without_querying(self):
        for count in (None, 0, -1, True, 1.0, "1"):
            with self.subTest(count=count), patch.object(gpu, "run") as command:
                with self.assertRaisesRegex(BuildError, "Invalid expected GPU count"):
                    gpu.confirm_not_ready({"gpu_count": count})
                command.assert_not_called()

    def test_not_ready_readback_rejects_missing_malformed_and_duplicate_inventory(self):
        for count, devices in (
            (1, None),
            (1, "0000:01:00.0\n"),
            (1, b""),
            (2, b"0000:01:00.0\n"),
            (1, b"GPU-0\n"),
            (1, b"0000:01:20.0\n"),
            (1, b"0000:01:00.8\n"),
            (1, b" 0000:01:00.0\n"),
            (2, b"0000:01:00.0\n\n"),
            (2, b"0000:01:00.0\n0000:01:00.0\n"),
            (2, b"0000:0A:00.0\n00000000:0a:00.0\n"),
        ):
            with self.subTest(count=count, devices=devices), patch.object(gpu, "run", return_value=devices) as command:
                with self.assertRaises(BuildError):
                    gpu.confirm_not_ready({"gpu_count": count})
                self.assertEqual(command.call_count, 1)

    def test_not_ready_readback_rejects_every_other_complete_output(self):
        not_ready = b"Confidential Compute GPUs Ready state: not-ready"
        for state in (
            None,
            "Confidential Compute GPUs Ready state: not-ready",
            b"",
            b"not-ready",
            b"Confidential Compute GPUs Ready state: ready",
            b"Confidential Compute GPUs Ready state: unknown",
            not_ready[:-1],
            not_ready + b"\n" + not_ready,
            not_ready + b"\nunexpected GPU error",
            b"GPU 0: not-ready\nGPU 1: ready",
        ):
            with self.subTest(state=state), patch.object(gpu, "run", side_effect=[b"0000:01:00.0\n", state]):
                with self.assertRaisesRegex(BuildError, "GPU Not Ready state is unconfirmed"):
                    gpu.confirm_not_ready({"gpu_count": 1})

    def test_not_ready_readback_propagates_inventory_and_state_command_failures(self):
        for responses in ([BuildError("command timed out")], [b"0000:01:00.0\n", BuildError("query failed")]):
            with self.subTest(responses=responses), patch.object(gpu, "run", side_effect=responses) as command:
                with self.assertRaises(BuildError):
                    gpu.confirm_not_ready({"gpu_count": 1})
                self.assertTrue(all(call.kwargs == {"timeout": 10} for call in command.call_args_list))

    def test_periodic_readiness_follows_authorized_composite_transaction(self):
        events = []
        cfg = {"gpu": "nvidia_cc", "gpu_count": 1, "platform": "intel_tdx"}

        @contextlib.contextmanager
        def authorize(*args, **kwargs):
            events.append("authorized")
            yield

        def read(path):
            if path == runtime.CONFIG:
                return cfg
            if path == runtime.STATE / "binding.json":
                return {"digest": "00" * 32}
            return {"container": {"attestation_credentials": False}}

        with (
            patch.object(runtime.Path, "exists", return_value=False),
            patch.object(runtime, "protect_process"),
            patch.object(runtime, "time_sync"),
            patch.object(runtime, "verify_local_binding"),
            patch.object(runtime, "read_json", side_effect=read),
            patch.object(runtime, "authorized_key", side_effect=authorize) as authorization,
            patch.object(runtime, "readiness", side_effect=lambda *args: events.append("ready")),
        ):
            runtime.periodic()
            self.assertEqual(events, ["authorized", "ready"])
            authorization.assert_called_once_with(cfg, bytes(32))
            events.clear()
            authorization.side_effect = BuildError("GPU appraisal denied by KBS")
            with self.assertRaises(BuildError):
                runtime.periodic()
            self.assertEqual(events, [])
