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

    def test_periodic_readiness_follows_authorized_composite_transaction(self):
        events = []
        cfg = {"gpu": "nvidia_cc", "gpu_count": 1, "platform": "intel_tdx"}

        @contextlib.contextmanager
        def authorize(*args):
            events.append("authorized")
            yield

        with (
            patch.object(runtime.Path, "exists", return_value=False),
            patch.object(runtime, "protect_process"),
            patch.object(runtime, "time_sync"),
            patch.object(runtime, "verify_local_binding"),
            patch.object(
                runtime, "read_json", side_effect=lambda path: cfg if path == runtime.CONFIG else {"digest": "00" * 32}
            ),
            patch.object(runtime, "authorized_key", side_effect=authorize) as authorization,
            patch.object(gpu, "readiness", side_effect=lambda *args: events.append("ready")),
        ):
            runtime.periodic()
            self.assertEqual(events, ["authorized", "ready"])
            events.clear()
            authorization.side_effect = BuildError("GPU appraisal denied by KBS")
            with self.assertRaises(BuildError):
                runtime.periodic()
            self.assertEqual(events, [])
