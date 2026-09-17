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

"""Production approval cannot silently accept incomplete or mismatched evidence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from builder.builder import resolve_acceptance_runner, run_acceptance, select_acceptance_runner
from builder.common import BuildError, digest_file, write_json
from builder.policy import required_acceptance_checks, verify_approval


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.manifest = {
            "build_id": "fixture-bundle",
            "dev_mode": False,
            "platform": "intel_tdx",
            "contract": {"gpu": "none"},
        }
        self.checks = required_acceptance_checks(self.manifest)
        write_json(self.directory / "cvm_manifest.json", self.manifest)
        self.receipt = {
            "build_id": self.manifest["build_id"],
            "manifest_sha256": digest_file(self.directory / "cvm_manifest.json"),
            "status": "approved",
            "checks": {name: {"passed": True, "evidence_sha256": "ab" * 32} for name in self.checks},
        }

    def verify(self):
        write_json(self.directory / "approval.json", self.receipt)
        # Bundle byte verification has separate coverage; isolate the evidence
        # gate here. These temporary fixture digests are never deployment receipts.
        with patch("builder.policy.verify_bundle", return_value=self.manifest):
            return verify_approval(self.directory)

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

    def test_platform_and_gpu_specific_evidence(self):
        for platform, gpu, required in (
            ("amd_sev_snp", "none", "snp_vcek_cache"),
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

    def test_acceptance_runner_approves_its_generated_report(self):
        runner = self.directory / "acceptance-runner"
        runner.write_text('#!/bin/sh\nprintf \'{"runner":"ok"}\\n\' > "$2"\n')
        runner.chmod(0o755)
        approve = Mock()
        with patch("builder.builder.approve_bundle", approve):
            run_acceptance(runner, self.directory)
        approve.assert_called_once_with(self.directory, {"runner": "ok"})

    def test_acceptance_runner_can_be_resolved_from_path(self):
        runner = self.directory / "site_acceptance"
        runner.write_text('#!/bin/sh\nprintf \'{"runner":"path"}\\n\' > "$2"\n')
        runner.chmod(0o755)
        approve = Mock()
        with patch.dict("os.environ", {"PATH": str(self.directory)}), patch("builder.builder.approve_bundle", approve):
            run_acceptance("site_acceptance", self.directory)
        approve.assert_called_once_with(self.directory, {"runner": "path"})

    def test_missing_acceptance_runner_fails_before_build(self):
        with patch.dict("os.environ", {"PATH": str(self.directory)}), self.assertRaises(BuildError):
            resolve_acceptance_runner("site_acceptance")

    def test_simple_build_automatically_selects_site_acceptance(self):
        runner = self.directory / "site_acceptance"
        runner.write_text("#!/bin/sh\nexit 0\n")
        runner.chmod(0o755)
        with patch.dict("os.environ", {"PATH": str(self.directory)}):
            self.assertEqual(select_acceptance_runner({}), str(runner))
            profile = {"acceptance_runner": "site_acceptance"}
            self.assertIsNone(select_acceptance_runner(profile, defer_measurements=True))
            self.assertIsNone(select_acceptance_runner(profile, dev=True))
            with self.assertRaises(BuildError):
                select_acceptance_runner({}, "site_acceptance", defer_measurements=True)
