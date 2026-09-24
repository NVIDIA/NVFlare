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

"""Production approval needs a trusted acceptance signature and complete, exact evidence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cvm.artifacts.acceptance import aggregate
from cvm.artifacts.bundle import (
    approve_bundle,
    key_id,
    load_public_keys,
    required_acceptance_checks,
    sign_receipt,
    verify_approval,
)
from cvm.build.cvm import resolve_acceptance_runner, run_acceptance, select_acceptance_runner
from cvm.common.errors import BuildError
from cvm.common.io import digest_file, read_json, write_json


def write_key_pair(directory, name):
    key = ed25519.Ed25519PrivateKey.generate()
    private = Path(directory) / f"{name}.key"
    public = Path(directory) / f"{name}.pub"
    private.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    public.write_bytes(
        key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    return private, public


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.manifest = {
            "build_id": "fixture-bundle",
            "dev_mode": False,
            "production_ready": True,
            "platform": "intel_tdx",
            "contract": {"gpu": "none", "production_ready": True},
        }
        self.checks = required_acceptance_checks(self.manifest)
        write_json(self.directory / "cvm_manifest.json", self.manifest)
        self.signing_key, self.public_key = write_key_pair(self.directory, "acceptance")
        self.receipt = {
            "build_id": self.manifest["build_id"],
            "manifest_sha256": digest_file(self.directory / "cvm_manifest.json"),
            "status": "approved",
            "checks": {name: {"passed": True, "evidence_sha256": "ab" * 32} for name in self.checks},
        }

    def verify(self, trusted=None, receipt=None):
        if receipt is None:
            receipt = sign_receipt(self.receipt, self.signing_key)
        write_json(self.directory / "approval.json", receipt)
        # Bundle byte verification has separate coverage; isolate the evidence
        # gate here. These temporary fixture digests are never deployment receipts.
        with patch("cvm.artifacts.bundle.verify_bundle", return_value=self.manifest):
            return verify_approval(self.directory, [self.public_key] if trusted is None else trusted)

    def test_complete_evidence_is_required_for_every_check(self):
        self.assertEqual(self.verify(), self.manifest)
        for name in self.checks:
            item = self.receipt["checks"].pop(name)
            with self.subTest(check=name), self.assertRaises(BuildError):
                self.verify()
            self.receipt["checks"][name] = item

    def test_failed_or_unhashed_evidence_is_denied(self):
        name = sorted(self.checks)[0]
        for item in ({"passed": False, "evidence_sha256": "ab" * 32}, {"passed": True}, True):
            self.receipt["checks"][name] = item
            with self.subTest(item=item), self.assertRaises(BuildError):
                self.verify()

    def test_approval_is_bound_to_exact_manifest_and_identity(self):
        for field in ("build_id", "manifest_sha256", "status"):
            original = self.receipt[field]
            self.receipt[field] = "wrong"
            with self.subTest(field=field), self.assertRaises(BuildError):
                self.verify()
            self.receipt[field] = original

    def test_development_can_never_be_approved(self):
        self.manifest["dev_mode"] = True
        with self.assertRaises(BuildError):
            self.verify()

    def test_nonproduction_profile_can_never_be_approved(self):
        self.manifest["production_ready"] = False
        self.manifest["contract"]["production_ready"] = False
        with self.assertRaisesRegex(BuildError, "not eligible"):
            self.verify()
        report = {"manifest_sha256": self.receipt["manifest_sha256"], "checks": self.receipt["checks"]}
        with (
            patch("cvm.artifacts.bundle.verify_bundle", return_value=self.manifest),
            self.assertRaisesRegex(BuildError, "not eligible"),
        ):
            approve_bundle(self.directory, report, self.signing_key)

    def test_platform_and_gpu_specific_evidence(self):
        for platform, gpu, required in (
            ("amd_sev_snp", "none", "snp_collateral_availability"),
            ("intel_tdx", "nvidia_cc", "gpu_negative_key_denial"),
        ):
            self.manifest.update(platform=platform, contract={"gpu": gpu})
            self.receipt["checks"] = {
                name: {"passed": True, "evidence_sha256": "ab" * 32}
                for name in required_acceptance_checks(self.manifest)
            }
            item = self.receipt["checks"].pop(required)
            with self.subTest(required=required), self.assertRaises(BuildError):
                self.verify()
            self.receipt["checks"][required] = item

    def test_unsigned_receipt_is_never_approval(self):
        with self.assertRaisesRegex(BuildError, "unsigned"):
            self.verify(receipt=self.receipt)
        trusted = load_public_keys([self.public_key])[0]
        forged = dict(self.receipt, signature={"algorithm": "ed25519", "key_id": key_id(trusted)})
        with self.assertRaisesRegex(BuildError, "unsigned|unsupported"):
            self.verify(receipt=forged)

    def test_signature_by_an_untrusted_authority_is_denied(self):
        other_key, other_public = write_key_pair(self.directory, "other")
        with self.assertRaisesRegex(BuildError, "trusted acceptance authority"):
            self.verify(receipt=sign_receipt(self.receipt, other_key))
        # The same receipt is accepted once that authority is trusted as well.
        self.assertEqual(
            self.verify(trusted=[self.public_key, other_public], receipt=sign_receipt(self.receipt, other_key)),
            self.manifest,
        )
        with self.assertRaisesRegex(BuildError, "No approval signing keys"):
            self.verify(trusted=[])

    def test_tampered_signed_receipt_fails_verification(self):
        signed = sign_receipt(self.receipt, self.signing_key)
        name = sorted(self.checks)[0]
        signed["checks"] = dict(signed["checks"], **{name: {"passed": True, "evidence_sha256": "cd" * 32}})
        with self.assertRaisesRegex(BuildError, "does not match"):
            self.verify(receipt=signed)

    def test_public_keys_must_be_ed25519_pem(self):
        with self.assertRaisesRegex(BuildError, "Ed25519"):
            load_public_keys([str(self.directory / "cvm_manifest.json")])
        with self.assertRaises(BuildError):
            load_public_keys([str(self.directory / "absent.pub")])
        loaded = load_public_keys([str(self.public_key)])[0]
        self.assertEqual(len(key_id(loaded)), 64)
        self.assertEqual(sign_receipt(self.receipt, self.signing_key)["signature"]["key_id"], key_id(loaded))

    def test_approve_bundle_publishes_a_signed_receipt_for_the_exact_manifest(self):
        report = {"manifest_sha256": self.receipt["manifest_sha256"], "checks": self.receipt["checks"]}
        with patch("cvm.artifacts.bundle.verify_bundle", return_value=self.manifest):
            approve_bundle(self.directory, report, self.signing_key)
            receipt = read_json(self.directory / "approval.json")
            self.assertEqual(receipt["signature"]["key_id"], key_id(load_public_keys([str(self.public_key)])[0]))
            self.assertEqual(receipt["status"], "approved")
            self.assertEqual(verify_approval(self.directory, [str(self.public_key)]), self.manifest)
            _, other_public = write_key_pair(self.directory, "other")
            with self.assertRaises(BuildError):
                verify_approval(self.directory, [str(other_public)])
            with self.assertRaisesRegex(BuildError, "another bundle"):
                approve_bundle(self.directory, dict(report, manifest_sha256="0" * 64), self.signing_key)

    def test_acceptance_runner_approves_its_generated_report(self):
        runner = self.directory / "acceptance-runner"
        runner.write_text('#!/bin/sh\nprintf \'{"runner":"ok"}\\n\' > "$2"\n')
        runner.chmod(0o755)
        approve = Mock()
        with patch("cvm.build.cvm.approve_bundle", approve):
            run_acceptance(runner, self.directory, str(self.signing_key))
        approve.assert_called_once_with(self.directory, {"runner": "ok"}, str(self.signing_key))

    def test_acceptance_runner_can_be_resolved_from_path(self):
        runner = self.directory / "site_acceptance"
        runner.write_text('#!/bin/sh\nprintf \'{"runner":"path"}\\n\' > "$2"\n')
        runner.chmod(0o755)
        approve = Mock()
        with patch.dict("os.environ", {"PATH": str(self.directory)}), patch("cvm.build.cvm.approve_bundle", approve):
            run_acceptance("site_acceptance", self.directory, str(self.signing_key))
        approve.assert_called_once_with(self.directory, {"runner": "path"}, str(self.signing_key))

    def test_missing_acceptance_runner_fails_before_build(self):
        with patch.dict("os.environ", {"PATH": str(self.directory)}), self.assertRaises(BuildError):
            resolve_acceptance_runner("site_acceptance")

    def test_acceptance_runner_is_explicit_and_requires_a_production_profile(self):
        runner = self.directory / "site_acceptance"
        runner.write_text("#!/bin/sh\nexit 0\n")
        runner.chmod(0o755)
        with patch.dict("os.environ", {"PATH": str(self.directory)}):
            self.assertIsNone(select_acceptance_runner({"production_ready": False}))
            profile = {"acceptance_runner": "site_acceptance", "production_ready": True}
            self.assertEqual(select_acceptance_runner(profile), str(runner))
            self.assertIsNone(select_acceptance_runner(profile, defer_measurements=True))
            self.assertIsNone(select_acceptance_runner(profile, dev=True))
            with self.assertRaises(BuildError):
                select_acceptance_runner({}, "site_acceptance", defer_measurements=True)
            with self.assertRaisesRegex(BuildError, "nonproduction"):
                select_acceptance_runner({"production_ready": False}, "site_acceptance")

    def test_acceptance_report_aggregates_only_exact_complete_evidence(self):
        first = self.directory / "first.json"
        second = self.directory / "second.json"
        names = sorted(self.checks)
        common = {
            "schema_version": 1,
            "manifest_sha256": self.receipt["manifest_sha256"],
            "platform": self.manifest["platform"],
        }

        def signed(selected, **changes):
            value = dict(common, checks={name: {"passed": True} for name in selected}, **changes)
            return sign_receipt(value, self.signing_key)

        write_json(first, signed(names[: len(names) // 2]))
        write_json(second, signed(names[len(names) // 2 :]))
        with patch("cvm.artifacts.acceptance.verify_bundle", return_value=self.manifest):
            report = aggregate(self.directory, [first, second], [self.public_key])
            self.assertEqual(set(report["checks"]), self.checks)
            self.assertEqual(report["checks"][names[0]]["evidence_sha256"], digest_file(first))
            write_json(second, signed(names[len(names) // 2 + 1 :]))
            with self.assertRaisesRegex(BuildError, "Missing acceptance checks"):
                aggregate(self.directory, [first, second], [self.public_key])
            write_json(second, signed(names[len(names) // 2 :]))
            write_json(first, signed(names[: len(names) // 2], manifest_sha256="0" * 64))
            with self.assertRaisesRegex(BuildError, "another finalized manifest"):
                aggregate(self.directory, [first, second], [self.public_key])

    def test_acceptance_report_requires_signed_explicit_success(self):
        path = self.directory / "result.json"
        name = sorted(self.checks)[0]
        common = {
            "schema_version": 1,
            "manifest_sha256": self.receipt["manifest_sha256"],
            "platform": self.manifest["platform"],
        }
        with patch("cvm.artifacts.acceptance.verify_bundle", return_value=self.manifest):
            write_json(path, dict(common, checks={name: {"passed": True}}))
            with self.assertRaisesRegex(BuildError, "unsigned"):
                aggregate(self.directory, [path], [self.public_key])
            foreign_key, _ = write_key_pair(self.directory, "foreign-evidence")
            write_json(path, sign_receipt(dict(common, checks={name: {"passed": True}}), foreign_key))
            with self.assertRaisesRegex(BuildError, "trusted acceptance authority"):
                aggregate(self.directory, [path], [self.public_key])
            write_json(path, sign_receipt(dict(common, checks={name: {"passed": False}}), self.signing_key))
            with self.assertRaisesRegex(BuildError, "invalid or inapplicable"):
                aggregate(self.directory, [path], [self.public_key])
            write_json(path, sign_receipt(dict(common, checks={name: {"passed": 1}}), self.signing_key))
            with self.assertRaisesRegex(BuildError, "invalid or inapplicable"):
                aggregate(self.directory, [path], [self.public_key])
            signed = sign_receipt(dict(common, checks={name: {"passed": True}}), self.signing_key)
            signed["checks"][name]["passed"] = False
            write_json(path, signed)
            with self.assertRaisesRegex(BuildError, "does not match"):
                aggregate(self.directory, [path], [self.public_key])
