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

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cvm.common.errors import BuildError
from cvm.common.io import digest_file, read_json, write_json
from cvm.trustee.bootstrap import create_deployment_receipt, issue_admin_token


class TrusteeBootstrapTests(unittest.TestCase):
    def test_issue_short_lived_scoped_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = ed25519.Ed25519PrivateKey.generate()
            private = root / "admin.key"
            private.write_bytes(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
            with patch("cvm.trustee.bootstrap.protect_process") as protect:
                output = issue_admin_token(private, "cvm-resources-build_1", root / "token.jwt", lifetime=60, now=100)
            protect.assert_called_once_with()
            header, claims, signature = output.read_text().strip().split(".")
            payload = json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))
            self.assertEqual(
                payload,
                {
                    "aud": "coco-trustee",
                    "exp": 160,
                    "iat": 100,
                    "iss": "cvm-builder",
                    "nbf": 95,
                    "role": "cvm-resources-build_1",
                },
            )
            key.public_key().verify(
                base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4)), f"{header}.{claims}".encode()
            )
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with patch("cvm.trustee.bootstrap.protect_process") as protect, self.assertRaises(BuildError):
                issue_admin_token(private, "cvm-resources", root / "long.jwt", lifetime=30 * 86400 + 1)
            protect.assert_not_called()

    def test_deployment_receipt_hashes_exact_policy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build.json"
            config = root / "kbs.json"
            policy = root / "cpu.rego"
            write_json(build, {"trustee_commit": "a" * 40, "source_clean": True, "binary_sha256": "b" * 64})
            write_json(
                config,
                {"attestation_service": {"policy_id_map": {"cvm-profile": ["cvm-profile"]}}},
            )
            policy.write_text("package policy\n")
            output = create_deployment_receipt(
                build,
                config,
                [f"cvm-profile_cpu={policy}"],
                root / "receipt.json",
                policy_selection_tested=True,
                unauthorized_administration_denied=True,
            )
            receipt = read_json(output)
            self.assertEqual(receipt["immutable_as_policies"], {"cvm-profile_cpu": digest_file(policy)})
            self.assertEqual(receipt["policy_id_map"], {"cvm-profile": ["cvm-profile"]})
            self.assertNotIn("binary_sha256", receipt)

    def test_deployment_receipt_requires_completed_negative_tests(self):
        with self.assertRaisesRegex(BuildError, "policy-selection"):
            create_deployment_receipt("missing", "missing", [], "receipt")

    def test_deployment_receipt_rejects_unapproved_patched_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build.json"
            config = root / "kbs.json"
            policy = root / "cpu.rego"
            write_json(build, {"trustee_commit": "a" * 40, "source_clean": False})
            write_json(config, {"attestation_service": {"policy_id_map": {"cvm-profile": ["cvm-profile"]}}})
            policy.write_text("package policy\n")
            with self.assertRaisesRegex(BuildError, "approved clean or CoCo"):
                create_deployment_receipt(
                    build,
                    config,
                    [f"cvm-profile_cpu={policy}"],
                    root / "receipt.json",
                    policy_selection_tested=True,
                    unauthorized_administration_denied=True,
                )
