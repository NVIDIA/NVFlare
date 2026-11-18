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

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvm.common.errors import BuildError
from cvm.common.io import digest_file
from cvm.common.versions import (
    COCO_ACTIX_HTTP_CRATE_SHA256,
    COCO_ACTIX_HTTP_SOURCE_SHA256,
    COCO_ACTIX_HTTP_VERSION,
    COCO_SERVICE_BUILD_PROFILE,
    COCO_SERVICE_PATCH_SHA256,
)
from cvm.trustee.admin import read_resource_policy, verify_server_provenance
from cvm.trustee.provenance import provenance


class UpstreamTrusteeTests(unittest.TestCase):
    def test_reviewed_client_patch_is_independent_of_git_abbreviation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            tracked = source / "Cargo.toml"
            tracked.write_text("original content\n")
            other = source / "README.md"
            other.write_text("unchanged\n")
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
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
            tracked.write_text("reviewed content\n")
            expected = subprocess.check_output(
                ["git", "-C", str(source), "diff", "--binary", "--abbrev=7", "--no-ext-diff", "--no-color", "HEAD"]
            )
            reviewed = root / "reviewed.patch"
            reviewed.write_bytes(expected)
            binary = root / "kbs-client"
            binary.write_bytes(b"fixture binary")
            with (
                patch("cvm.trustee.provenance.TRUSTEE_COMMIT", commit),
                patch("cvm.trustee.provenance.CLIENT_PATCH", reviewed),
            ):
                for width in (4, 7, 8, 12, 40):
                    with self.subTest(width=width):
                        subprocess.run(["git", "-C", str(source), "config", "core.abbrev", str(width)], check=True)
                        result = provenance(source, binary)
                        self.assertFalse(result["source_clean"])
                        self.assertEqual(result["kbs_client_patch_sha256"], hashlib.sha256(expected).hexdigest())
                # Normalizing index rendering must not admit different source, modes or files.
                for variant in ("content", "other_file", "mode", "untracked", "index_prefix"):
                    with self.subTest(variant=variant):
                        tracked.write_text(
                            "reviewed content\nextra change\n" if variant == "content" else "reviewed content\n"
                        )
                        other.write_text("extra change\n" if variant == "other_file" else "unchanged\n")
                        tracked.chmod(0o755 if variant == "mode" else 0o644)
                        subprocess.run(["git", "-C", str(source), "config", "core.filemode", "true"], check=True)
                        untracked = source / "unreviewed.rs"
                        if variant == "untracked":
                            untracked.write_text("extra source\n")
                        reviewed.write_bytes(
                            expected.replace(b"index ", b"index fffffff", 1) if variant == "index_prefix" else expected
                        )
                        with self.assertRaisesRegex(BuildError, "unmodified"):
                            provenance(source, binary)
                        untracked.unlink(missing_ok=True)

    def test_shared_coco_service_provenance_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "kbs"
            binary.write_bytes(b"fixture binary")
            recipe = {
                "source_clean": False,
                "build_profile": COCO_SERVICE_BUILD_PROFILE,
                "source_patch_sha256": COCO_SERVICE_PATCH_SHA256,
                "actix_http_version": COCO_ACTIX_HTTP_VERSION,
                "actix_http_crate_sha256": COCO_ACTIX_HTTP_CRATE_SHA256,
                "actix_http_source_sha256": COCO_ACTIX_HTTP_SOURCE_SHA256,
                "kbs_image_id": "sha256:" + "a" * 64,
            }
            build = dict(recipe, binary_sha256=digest_file(binary))
            config = {
                "trustee_binary": str(binary),
                "trustee_image_id": build["kbs_image_id"],
            }
            verify_server_provenance(config, build, recipe)
            for name, value in (
                ("source_patch_sha256", "b" * 64),
                ("actix_http_version", "other"),
                ("kbs_image_id", "sha256:" + "b" * 64),
            ):
                with self.subTest(name=name), self.assertRaises(BuildError):
                    verify_server_provenance(config, dict(build, **{name: value}), recipe)

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
