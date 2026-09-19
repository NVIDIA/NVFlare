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

"""Compatibility boundaries for an unmodified CoCo Trustee deployment."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvm.common.errors import BuildError
from cvm.common.io import digest_file
from cvm.trustee.admin import read_resource_policy
from cvm.trustee.provenance import provenance


class UpstreamTrusteeTests(unittest.TestCase):
    def test_provenance_rejects_modified_or_wrong_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            tracked = source / "Cargo.toml"
            tracked.write_text("fixture")
            subprocess.run(["git", "-C", str(source), "add", "Cargo.toml"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.org",
                    "commit",
                    "-qm",
                    "fixture",
                    "--no-gpg-sign",
                ],
                check=True,
            )
            commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            binary = root / "kbs"
            binary.write_bytes(b"fixture binary")
            with self.assertRaisesRegex(BuildError, "revision"):
                provenance(source, binary)
            with patch("cvm.trustee.provenance.TRUSTEE_COMMIT", commit):
                self.assertEqual(
                    provenance(source, binary),
                    {
                        "trustee_commit": commit,
                        "source_clean": True,
                        "binary_sha256": digest_file(binary),
                    },
                )
                tracked.write_text("patched")
                with self.assertRaisesRegex(BuildError, "unmodified"):
                    provenance(source, binary)
                tracked.write_text("fixture")
                (source / "untracked.rs").write_text("unreviewed")
                with self.assertRaisesRegex(BuildError, "unmodified"):
                    provenance(source, binary)

    def test_policy_listing_is_not_policy_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kbs").mkdir()
            policy = b"package policy\nimport rego.v1\ndefault allow := false\n"
            (root / "kbs/resource-policy.rego").write_bytes(policy)
            with patch("cvm.trustee.admin.api", return_value=b'["resource-policy"]') as api:
                self.assertEqual(read_resource_policy({"storage_directory": directory}), policy)
                api.assert_called_once_with({"storage_directory": directory}, "GET", "resource-policy")
            for response in (b"[]", b'{"resource-policy": "encoded"}'):
                with patch("cvm.trustee.admin.api", return_value=response), self.assertRaises(BuildError):
                    read_resource_policy({"storage_directory": directory})
