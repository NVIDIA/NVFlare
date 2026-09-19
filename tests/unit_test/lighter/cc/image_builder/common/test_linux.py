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

"""Actionable child errors must not disclose command arguments or secret output."""

import subprocess
import unittest
from unittest.mock import patch

from cvm.common.errors import BuildError
from cvm.common.linux import run


class ProcessTests(unittest.TestCase):
    def test_operation_labels_preserve_output_suppression(self):
        secret = "private-token-or-key"
        command = ["/usr/bin/kbs-client", secret]
        for operation in ("KBS quote/appraisal", "KBS resource retrieval/decryption"):
            failures = (
                subprocess.CompletedProcess(command, 7, secret.encode(), secret.encode()),
                subprocess.TimeoutExpired(command, 1, output=secret.encode(), stderr=secret.encode()),
                FileNotFoundError(secret),
            )
            for failure in failures:
                with self.subTest(operation=operation, failure=type(failure).__name__):
                    options = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                    with patch("cvm.common.linux.subprocess.run", **options), self.assertRaises(BuildError) as error:
                        run(command, operation=operation)
                    self.assertIn(operation, str(error.exception))
                    self.assertNotIn(secret, str(error.exception))

    def test_existing_callers_retain_executable_label(self):
        with patch("cvm.common.linux.subprocess.run", return_value=subprocess.CompletedProcess([], 2)):
            with self.assertRaisesRegex(BuildError, r"^kbs-client failed \(exit 2\)"):
                run(["/usr/bin/kbs-client"])
