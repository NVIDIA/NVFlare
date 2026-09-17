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

"""Regressions for PR security boundaries and fresh-install failures."""

import datetime
import socket
import ssl
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from builder.common import BuildError, write_json
from builder.key_service import BoundedTLSServer, handler
from builder.launcher import find_bundle
from builder.references import check_profile, merge_records, validate_references
from builder.services import validate_service

ROOT = Path(__file__).resolve().parent.parent


class ReviewRegressionTests(unittest.TestCase):
    def test_gpu_receipt_requires_both_immutable_policy_digests(self):
        from builder.admin import install
        from builder.common import digest_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "kbs"
            binary.write_bytes(b"fixture binary")
            contract = {"gpu": "nvidia_cc", "trustee_commit": "a" * 40, "trustee_patch_digest": "b" * 64}
            manifest = {
                "build_id": "test-gpu",
                "profile_version": "test-gpu-r2",
                "contract": contract,
                "attestation_policy_id": "gpu-r2",
                "sha256": {"attestation_policy.rego": "c" * 64, "gpu_attestation_policy.rego": "d" * 64},
            }
            cfg = {
                name: str(root / name)
                for name in ("state", "resources", "key_service_state", "deployment_receipt", "trustee_build")
            }
            cfg["trustee_binary"] = str(binary)
            write_json(cfg["trustee_build"], dict(contract, binary_sha256=digest_file(binary)))
            receipt = dict(
                contract,
                policy_selection_tested=True,
                unauthorized_administration_denied=True,
                immutable_as_policies={"gpu-r2_cpu": "c" * 64, "gpu-r2_gpu": "d" * 64},
            )
            with (
                patch("builder.admin.verify_bundle", return_value=manifest),
                patch("builder.admin.api", side_effect=RuntimeError("validated receipt")) as request,
            ):
                for missing in ("gpu-r2_cpu", "gpu-r2_gpu"):
                    value = dict(
                        receipt,
                        immutable_as_policies={
                            key: digest for key, digest in receipt["immutable_as_policies"].items() if key != missing
                        },
                    )
                    write_json(cfg["deployment_receipt"], value)
                    with self.assertRaisesRegex(BuildError, "not installed immutably"):
                        install(cfg, root, candidate=True)
                    request.assert_not_called()
                write_json(cfg["deployment_receipt"], receipt)
                with self.assertRaisesRegex(RuntimeError, "validated receipt"):
                    install(cfg, root, candidate=True)
                binary.write_bytes(b"unreviewed binary")
                with self.assertRaisesRegex(BuildError, "binary differs"):
                    install(cfg, root, candidate=True)

    def test_path_traversal_and_systemd_expansion_are_denied(self):
        for executable in (
            "/vault/application/../../usr/bin/id",
            "/vault/application/%i",
            "/vault/application/$CMD",
            "/usr/bin/id",
        ):
            with self.subTest(executable=executable), self.assertRaises(BuildError):
                validate_service("app_test.service", "[Service]\nExecStart=" + executable)
        validate_service("app_test.service", '[Service]\nExecStart="/vault/application/run me" --argument')

    def test_unknown_tcb_keys_fail_before_build(self):
        with self.assertRaisesRegex(BuildError, "_comment_note"):
            validate_references({"_comment_note": "documentation"})
        with self.assertRaisesRegex(BuildError, "tcb_svn"):
            validate_references({"tcb_svn": ["invalid"]})

    def test_reference_import_cannot_union_or_extend_expiry(self):
        old = [{"name": "snp_bootloader", "value": [1], "version": "0.1.0", "expiration": "2020-01-01T00:00:00Z"}]
        expires = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)).isoformat()
        self.assertEqual(merge_records(old, {"snp_bootloader": [1]}, expires), old)
        with self.assertRaisesRegex(BuildError, "Conflicting"):
            merge_records(old, {"snp_bootloader": [1, 2]}, expires)

    def test_profile_isolation_includes_version_and_contract(self):
        original = {"profile_version": "cpu-r1", "contract": {"reference_values_sha256": "aa"}}
        for updated in (
            dict(original, profile_version="cpu-r2"),
            dict(original, contract={"reference_values_sha256": "bb"}),
        ):
            with self.assertRaisesRegex(BuildError, "another security profile"):
                check_profile(original, updated)

    def test_embedded_bundle_wins_over_build_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delivery = root / "vault_1" / "intel_tdx"
            embedded = delivery / "cvm_bundle"
            cached = root / "cvm_test-r1" / "intel_tdx"
            for directory in (embedded, cached):
                directory.mkdir(parents=True)
                write_json(directory / "cvm_manifest.json", {"build_id": "same"})
            manifest = {"profile_version": "test-r1", "platform": "intel_tdx", "cvm_build_id": "same"}
            self.assertEqual(find_bundle(delivery, manifest), embedded.resolve())
            write_json(embedded / "cvm_manifest.json", {"build_id": "wrong"})
            with self.assertRaisesRegex(BuildError, "Embedded"):
                find_bundle(delivery, manifest)

    def test_logs_do_not_disclose_exception_text(self):
        from builder.key_service import log_failure

        with self.assertLogs("builder.key_service", level="ERROR") as captured:
            log_failure("PUT", PermissionError(13, "secret request content"))
        self.assertIn("PermissionError", captured.output[0])
        self.assertIn("errno=13", captured.output[0])
        self.assertNotIn("secret request", captured.output[0])

    def test_slow_handshake_has_absolute_deadline_and_bounded_slots(self):
        server = BoundedTLSServer(
            ("127.0.0.1", 0), handler(None, {}), ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER), workers=1, deadline=0.2
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            first = socket.create_connection(server.server_address, timeout=1)
            time.sleep(0.05)
            second = socket.create_connection(server.server_address, timeout=1)
            self.assertEqual(second.recv(1), b"")
            second.close()
            self.assertEqual(first.recv(1), b"")
            first.close()
            time.sleep(0.05)
            self.assertTrue(server.slots.acquire(blocking=False))
            server.slots.release()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_nfs_requires_vault_config_and_authenticated_transport(self):
        from builder.config import validate_nfs_mount
        from builder.runtime import mount_user_data

        valid = {"server": "nfs.example.org", "export": "/data", "security": "krb5p"}
        validate_nfs_mount(valid)
        for invalid in (dict(valid, security="sys"), dict(valid, export="/../data")):
            with self.assertRaises(BuildError):
                validate_nfs_mount(invalid)
        with patch("builder.runtime.Path.exists", return_value=True), patch("builder.runtime.run") as mount:
            with self.assertRaisesRegex(BuildError, "ext_mount"):
                mount_user_data()
            mount.assert_not_called()


if __name__ == "__main__":
    unittest.main()
