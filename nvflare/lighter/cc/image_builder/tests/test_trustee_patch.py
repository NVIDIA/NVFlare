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

"""Reproduce the deployment patch from exact upstream pins, without hardware."""

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from builder.config import PROFILE_DEFAULTS

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(
    os.environ.get("CVM_TRUSTEE_SOURCE") and os.environ.get("CVM_GUEST_SOURCE"),
    "Opt in with local clones of the two pinned upstream repositories",
)
class TrusteePatchTests(unittest.TestCase):
    def test_clean_checkout_patch_matches_profile_and_is_reproducible(self):
        digests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "trustee"
                subprocess.run(
                    ["git", "clone", "--quiet", "--shared", os.environ["CVM_TRUSTEE_SOURCE"], str(source)], check=True
                )
                subprocess.run(
                    ["git", "-C", str(source), "checkout", "--quiet", "--detach", PROFILE_DEFAULTS["trustee_commit"]],
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/patch_trustee.py"),
                        str(source),
                        "--guest-source",
                        os.environ["CVM_GUEST_SOURCE"],
                    ],
                    check=True,
                    capture_output=True,
                )
                digest = hashlib.sha256((source / "cvm-boundary.patch").read_bytes()).hexdigest()
                self.assertEqual(digest, PROFILE_DEFAULTS["trustee_patch_digest"])
                digests.append(digest)
                # Provenance validation must notice edits to either source tree.
                fake_binary = Path(directory) / "binary"
                fake_binary.write_bytes(b"fixture")
                command = [
                    sys.executable,
                    str(ROOT / "scripts/trustee_provenance.py"),
                    str(source),
                    str(fake_binary),
                    str(Path(directory) / "record.json"),
                ]
                subprocess.run(command, check=True, capture_output=True)
                guest = source / "cvm_guest/attestation-agent/attester/src/nvidia.rs"
                guest.write_text(guest.read_text() + "\n// unreviewed edit\n")
                self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(digests[0], digests[1])
