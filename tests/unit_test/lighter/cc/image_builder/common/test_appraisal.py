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

"""Evaluate the actual AS policy, including TCB and event-log failure paths."""

import copy
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from cvm.build.config import SOURCE

ENGINE = Path(
    os.environ.get("CVM_POLICY_EVAL", str(Path(__file__).parents[1] / "policy_engine/target/release/cvm-policy-eval"))
)
POLICY = SOURCE / "config/attestation_policy.rego"


@unittest.skipUnless(ENGINE.is_file(), "Build the pinned policy engine first")
class AppraisalTests(unittest.TestCase):
    def evaluate(self, claims, refs, dimension):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "input.json").write_text(json.dumps(claims))
            (root / "data.json").write_text(
                json.dumps({"reference": dict(refs, cvm_reference_expiry={name: time.time() + 300 for name in refs})})
            )
            expected = 3 if dimension == "executables" else 2
            result = subprocess.run(
                [
                    str(ENGINE),
                    str(POLICY),
                    str(root / "input.json"),
                    str(root / "data.json"),
                    f"data.policy.{dimension} == {expected}",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout.strip() == "true"

    def test_tdx_tcb_collateral_event_log_and_configuration(self):
        body = {
            "mr_td": "1" * 96,
            "rtmr_0": "0" * 96,
            "rtmr_1": "2" * 96,
            "rtmr_2": "3" * 96,
            "mr_seam": "4" * 96,
            "tcb_svn": "5" * 32,
            "xfam": "6" * 16,
        }
        tdx = {
            "quote": {"header": {"tee_type": "81000000"}, "body": body},
            "uefi_event_logs": {"kernel": "measured"},
            "tcb_status": "OK",
            "collateral_expiration_status": "0",
            "advisory_ids": [],
            "td_attributes": {"debug": False},
        }
        refs = {key: [value] for key, value in body.items()}
        refs["allowed_advisory_ids"] = []
        for dimension in ("executables", "hardware", "configuration"):
            self.assertTrue(self.evaluate({"tdx": tdx}, refs, dimension))
            self.assertFalse(self.evaluate({"tdx": tdx}, {}, dimension))
        for key, value, dimension in (
            ("uefi_event_logs", {}, "executables"),
            ("tcb_status", "OutOfDate", "hardware"),
            ("collateral_expiration_status", "1", "hardware"),
            ("advisory_ids", ["UNAPPROVED"], "hardware"),
            ("td_attributes", {"debug": True}, "configuration"),
        ):
            bad = copy.deepcopy(tdx)
            bad[key] = value
            with self.subTest(key=key):
                self.assertFalse(self.evaluate({"tdx": bad}, refs, dimension))

    def test_snp_measurement_tcb_debug_and_migration(self):
        snp = {
            "measurement": "m",
            "reported_tcb_bootloader": 1,
            "reported_tcb_tee": 2,
            "reported_tcb_snp": 3,
            "reported_tcb_microcode": 4,
            "policy_debug_allowed": False,
            "policy_migrate_ma": False,
            "platform_smt_enabled": True,
            "platform_tsme_enabled": False,
            "policy_abi_major": 1,
            "policy_abi_minor": 0,
            "policy_single_socket": False,
            "policy_smt_allowed": True,
        }
        refs = {
            "snp_launch_measurement": ["m"],
            "snp_bootloader": [1],
            "snp_tee_svn": [2],
            "snp_snp_svn": [3],
            "snp_microcode": [4],
            "snp_smt_enabled": True,
            "snp_tsme_enabled": False,
            "snp_guest_abi_major": 1,
            "snp_guest_abi_minor": 0,
            "snp_single_socket": False,
            "snp_smt_allowed": True,
        }
        for dimension in ("executables", "hardware", "configuration"):
            self.assertTrue(self.evaluate({"snp": snp}, refs, dimension))
            self.assertFalse(self.evaluate({"snp": snp}, {}, dimension))
        for key, value, dimension in (
            ("measurement", "unapproved", "executables"),
            ("reported_tcb_snp", 99, "hardware"),
            ("policy_debug_allowed", True, "configuration"),
            ("policy_migrate_ma", True, "configuration"),
        ):
            bad = dict(snp)
            bad[key] = value
            with self.subTest(key=key):
                self.assertFalse(self.evaluate({"snp": bad}, refs, dimension))
