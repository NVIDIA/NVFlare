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

"""Exercise GPU acceptance lifetime checks without hardware or firewall changes."""

import contextlib
import importlib.util
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

LAB = Path(__file__).resolve().parents[4] / "integration_test/lighter/cc/image_builder"


def load_hardware_tests():
    spec = importlib.util.spec_from_file_location("gpu_acceptance_hardware_tests", LAB / "test_hardware.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GPUQuarantineAcceptanceTests(unittest.TestCase):
    def exercise(self, exit_seconds, *, workload_stop_seconds=0, cleanup_seconds=0, hardware=None):
        hardware = hardware or load_hardware_tests()
        clock = SimpleNamespace(now=100.0)
        started = clock.now
        case = hardware.HardwareTests("test_periodic_gpu_denial_powers_off")
        case.manifest = {"contract": {"gpu": "nvidia_cc", "gpu_attestation_url": "https://nras.example.test"}}
        case.boot = Mock()
        case.ready = Mock()
        case.result = Mock()
        case.assert_powered_off = Mock()
        case.process = Mock()
        case.process.poll.side_effect = lambda: 0 if clock.now - started >= exit_seconds else None

        def wait(timeout):
            remaining = max(0, started + exit_seconds - clock.now)
            clock.now += min(timeout, remaining)
            if remaining > timeout:
                raise subprocess.TimeoutExpired("guest", timeout)
            return 0

        def sleep(seconds):
            clock.now += seconds

        def read_log(**kwargs):
            if clock.now - started < exit_seconds:
                return ""
            return "Power down\n"

        def request(path="/state", method="GET"):
            if method != "POST" and clock.now - started >= workload_stop_seconds:
                raise OSError("workload is stopped")
            return b"{}"

        @contextlib.contextmanager
        def drop_egress(*args):
            try:
                yield
            finally:
                clock.now += cleanup_seconds

        case.process.wait.side_effect = wait
        case.request = Mock(side_effect=request)
        case.drop_egress = drop_egress
        case.logs = [Mock(read_text=Mock(side_effect=read_log))]
        method = hardware.HardwareTests.test_periodic_gpu_denial_powers_off
        while hasattr(method, "__wrapped__"):
            method = method.__wrapped__
        with (
            patch.object(hardware.time, "monotonic", side_effect=lambda: clock.now),
            patch.object(hardware.time, "sleep", side_effect=sleep),
            patch.object(hardware.socket, "getaddrinfo", return_value=[(None, None, None, None, ("192.0.2.1", 443))]),
            patch.object(hardware, "periodic_timeout", return_value=240),
        ):
            method(case)
        case.assert_powered_off.assert_called_once_with()
        case.result.assert_called_once_with(
            periodic_gpu_denial=True,
            backend_nras_drop=True,
            workload_denied_while_isolated=True,
            terminal_gpu_quarantine=True,
            fail_closed_seconds=float(exit_seconds),
        )
        case.request.assert_any_call("/periodic", "POST")
        return case

    def test_clean_terminal_poweroff_does_not_require_an_advisory_record(self):
        self.exercise(1)

    def test_workload_must_stop_within_the_containment_budget(self):
        with self.assertRaisesRegex(AssertionError, "did not stop the workload"):
            self.exercise(646, workload_stop_seconds=466)

    def test_terminal_poweroff_lifetime_boundaries(self):
        for lifetime in (1, 120, 645):
            with self.subTest(lifetime=lifetime):
                self.exercise(lifetime)

    def test_poweroff_after_upper_budget_is_rejected(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.exercise(646)

    def test_fault_cleanup_does_not_change_measured_lifetime(self):
        self.exercise(120, cleanup_seconds=900)


if __name__ == "__main__":
    unittest.main()
