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

"""Exercise bootstrap diagnostics with immediate kills and an unread console."""

import os
import signal
import subprocess
import sys
import unittest

PROGRAM = """
import os, signal, sys
from cvm.common.errors import StartupPhaseError
from cvm.runtime import bootstrap
sys.argv = ["bootstrap", "bootstrap"]
def fail():
    raise StartupPhaseError(PHASE) from RuntimeError("private-token-content")
bootstrap.bootstrap = fail
bootstrap.emit = lambda event: None
NOTIFY
bootstrap.main()
"""


class BootstrapDiagnosticsTests(unittest.TestCase):
    def test_phase_reaches_stderr_before_immediate_watchdog_kill(self):
        for phase in ("attestation", "vault_identity"):
            with self.subTest(phase=phase):
                code = PROGRAM.replace("PHASE", repr(phase)).replace(
                    "NOTIFY", "bootstrap.notify = lambda message: os.kill(os.getpid(), signal.SIGKILL)"
                )
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, -signal.SIGKILL)
                self.assertEqual(result.stderr, f"CVM bootstrap failed [phase={phase}]; PID 1 will power off\n")
                self.assertNotIn("private-token-content", result.stderr)

    def test_full_stderr_cannot_delay_shutdown_even_if_notify_fails(self):
        for notify_fails in (False, True):
            with self.subTest(notify_fails=notify_fails):
                read_fd, write_fd = os.pipe()
                try:
                    os.set_blocking(write_fd, False)
                    try:
                        while True:
                            os.write(write_fd, b"x" * 4096)
                    except BlockingIOError:
                        pass
                    os.set_blocking(write_fd, True)
                    notify = """def notify(message):
    assert message == "WATCHDOG=trigger"
    assert os.get_blocking(2), "stderr flags were not restored"
    print("shutdown requested", flush=True)
    if FAILS:
        raise OSError("notify unavailable")
bootstrap.notify = notify""".replace(
                        "FAILS", repr(notify_fails)
                    )
                    code = PROGRAM.replace("PHASE", repr("attestation")).replace("NOTIFY", notify)
                    result = subprocess.run(
                        [sys.executable, "-c", code], stderr=write_fd, stdout=subprocess.PIPE, text=True, timeout=5
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "shutdown requested\n")
                    self.assertTrue(os.get_blocking(write_fd))
                finally:
                    os.close(read_fd)
                    os.close(write_fd)
