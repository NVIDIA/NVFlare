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
    def exercise(self, exit_seconds, *, audit_seconds=0, cleanup_seconds=0, late_request_seconds=0, hardware=None):
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
            clock.now = max(clock.now, started + audit_seconds)
            return '{"decision":"quarantine"}'

        def request(path="/state", method="GET"):
            if method != "POST":
                if case.request.call_count == 3:
                    clock.now += late_request_seconds
                raise OSError("workload is stopped")

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
            fail_closed_seconds=float(exit_seconds),
        )
        self.assertEqual(case.request.call_count, 3)
        return case

    def test_early_poweroff_is_rejected_even_with_slow_fault_cleanup(self):
        for cleanup in (0, 900):
            with self.subTest(cleanup=cleanup):
                with self.assertRaisesRegex(AssertionError, "quarantine recovery window"):
                    self.exercise(120, cleanup_seconds=cleanup)

    def test_minimum_and_upper_lifetime_boundaries(self):
        for lifetime in (900, 901, 1725):
            with self.subTest(lifetime=lifetime):
                self.exercise(lifetime)

    def test_poweroff_after_upper_budget_is_rejected(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.exercise(1726)

    def test_delayed_audit_does_not_reset_minimum_lifetime(self):
        self.exercise(900, audit_seconds=300)

    def test_fault_cleanup_does_not_change_measured_lifetime(self):
        self.exercise(901, cleanup_seconds=900)

    def test_slow_workload_probe_cannot_hide_early_guest_exit(self):
        with self.assertRaisesRegex(AssertionError, "minimum-lifetime observation"):
            self.exercise(120, late_request_seconds=900)


if __name__ == "__main__":
    unittest.main()
