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

"""Audit metadata remains narrow even when inputs contain sensitive fields."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvm.runtime.audit import append, record


class AuditTests(unittest.TestCase):
    def test_short_writes_complete_and_no_progress_does_not_spin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.log"
            write = os.write
            with patch("cvm.runtime.audit.os.write", side_effect=lambda fd, data: write(fd, data[:2])):
                append(path, b"complete line\n")
            self.assertEqual(path.read_bytes(), b"complete line\n")
            for result in (0, BlockingIOError()):
                options = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
                with patch("cvm.runtime.audit.os.write", **options) as call, self.assertRaises(OSError):
                    append(path, b"next\n")
                call.assert_called_once()

    def test_only_public_fields_are_serialized(self):
        value = record(
            {"build_id": "bundle-1", "attestation_policy_id": "policy-1", "secret": "never-log"},
            {"luks_uuid": "public-id", "measurements": {"mr_td": "public-measurement"}, "token": "never-log"},
            "deny",
        )
        self.assertEqual(set(value), {"ts", "cvm_build_id", "vault_id", "measurement", "policy_id", "decision"})
        self.assertNotIn("never-log", str(value))
        self.assertEqual(value["decision"], "deny")

    def test_append_does_not_follow_sidecar_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_bytes(b"unchanged")
            link = Path(directory) / "attestation.log"
            link.symlink_to(target)
            with self.assertRaises(OSError):
                append(link, b"public metadata\n")
            self.assertEqual(target.read_bytes(), b"unchanged")
            link.unlink()
            append(link, b"first\n")
            append(link, b"second\n")
            self.assertEqual(link.read_bytes(), b"first\nsecond\n")
