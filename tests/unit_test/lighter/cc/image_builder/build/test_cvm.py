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

"""CVM construction failure diagnostics."""

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cvm.build.cvm import collect_reference, reference_boot_failure
from cvm.common.errors import BuildError
from cvm.common.references import SNP_POLICY_RESERVED, SNP_POLICY_SINGLE_SOCKET


class ReferenceBootFailureTests(unittest.TestCase):
    def setUp(self):
        self.policy = SNP_POLICY_RESERVED | SNP_POLICY_SINGLE_SOCKET
        self.manifest = {
            "platform": "amd_sev_snp",
            "contract": {"gpu": "none", "gpu_count": 0},
            "launch_shape": {"snp_policy": self.policy},
        }
        self.log_text = (
            "qemu-system-x86_64: sev_snp_launch_start: SNP_LAUNCH_START ret=-22 fw_error=0 " "SECRET_SENTINEL\n"
        )

    def test_snp_launch_error_reports_status_policy_and_action(self):
        error = reference_boot_failure(self.manifest, self.log_text)

        self.assertIn("SNP_LAUNCH_START failed (ret=-22, fw_error=0)", error)
        self.assertIn(f"guest policy 0x{self.policy:x}", error)
        self.assertIn("snp_single_socket to false", error)
        self.assertNotIn("SECRET_SENTINEL", error)

    def test_unrecognized_output_uses_generic_safe_error(self):
        error = reference_boot_failure(self.manifest, "untrusted SECRET_SENTINEL output\n")

        self.assertEqual(error, "Reference VM exited; inspect reference-boot.log")

    def test_single_socket_action_is_only_shown_when_requested(self):
        manifest = dict(self.manifest, launch_shape={"snp_policy": SNP_POLICY_RESERVED})

        error = reference_boot_failure(manifest, self.log_text)

        self.assertIn("Confirm host firmware support", error)
        self.assertNotIn("snp_single_socket to false", error)

    def test_collect_reference_reads_final_log_once_after_process_exits(self):
        @contextlib.contextmanager
        def failed_process(command, log):
            Path(log).write_text("")
            process = Mock()

            def poll():
                if process.poll.call_count == 1:
                    return None
                Path(log).write_text(self.log_text)
                return 1

            process.poll.side_effect = poll
            yield process

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            directory.mkdir()
            with (
                patch("cvm.build.cvm.host_capabilities", return_value={"amd_sev_snp"}),
                patch("cvm.build.cvm.sidecar"),
                patch("cvm.build.cvm.qemu_command", return_value=["qemu-system-x86_64"]),
                patch("cvm.build.cvm.cbit_position", return_value=51),
                patch("cvm.build.cvm.owned_process", side_effect=failed_process),
                patch("cvm.build.cvm.time.sleep"),
                patch("cvm.build.cvm.reference_boot_failure", wraps=reference_boot_failure) as diagnostic,
                self.assertRaises(BuildError) as raised,
            ):
                collect_reference(self.manifest, directory, timeout=1)

        error = str(raised.exception)
        diagnostic.assert_called_once_with(self.manifest, self.log_text)
        self.assertIn("SNP_LAUNCH_START failed (ret=-22, fw_error=0)", error)
        self.assertIn(f"guest policy 0x{self.policy:x}", error)
        self.assertNotIn("SECRET_SENTINEL", error)


if __name__ == "__main__":
    unittest.main()
