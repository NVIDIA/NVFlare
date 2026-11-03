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

import json
import os
import queue
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cvm.runtime import audit
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

    def test_diagnostic_has_separate_console_stream_and_no_identity_or_sidecar(self):
        for code in ("gpu_readiness_revoke_failed", "gpu_quarantine_terminal"):
            with (
                self.subTest(code=code),
                patch.object(audit.Path, "exists", return_value=False),
                patch.object(audit, "read_json") as read,
                patch.object(audit, "append") as append_log,
                patch("builtins.print") as output,
            ):
                audit._write(audit._Diagnostic(code))
            output.assert_called_once()
            line = output.call_args.args[0]
            self.assertTrue(line.startswith("CVM_DIAGNOSTIC="))
            value = json.loads(line.removeprefix("CVM_DIAGNOSTIC="))
            self.assertEqual(set(value), {"ts", "code"})
            self.assertEqual(value["code"], code)
            read.assert_not_called()
            append_log.assert_not_called()

    def test_authorization_writer_retains_six_fields_and_sidecar(self):
        with (
            patch.object(audit.Path, "exists", return_value=False),
            patch.object(audit, "read_json", return_value={"build_id": "build", "attestation_policy_id": "policy"}),
            patch.object(audit, "append") as append_log,
            patch("builtins.print") as output,
        ):
            audit._write("allow")
        line = output.call_args.args[0]
        self.assertTrue(line.startswith("CVM_ATTESTATION="))
        value = json.loads(line.removeprefix("CVM_ATTESTATION="))
        self.assertEqual(set(value), {"ts", "cvm_build_id", "vault_id", "measurement", "policy_id", "decision"})
        self.assertEqual(value["decision"], "allow")
        append_log.assert_called_once_with("/applog/attestation.log", audit.canonical(value) + b"\n")

    def test_development_mode_suppresses_both_record_types(self):
        with (
            patch.object(audit.Path, "exists", return_value=True),
            patch.object(audit, "read_json") as read,
            patch.object(audit, "append") as append_log,
            patch("builtins.print") as output,
        ):
            audit._write("allow")
            audit._write(audit._Diagnostic("gpu_readiness_revoke_failed"))
        read.assert_not_called()
        append_log.assert_not_called()
        output.assert_not_called()

    def test_unknown_diagnostic_values_are_not_queued_or_serialized(self):
        with patch.object(audit, "_enqueue") as enqueue:
            for code in ("private driver stderr", "", None, {"secret": "never-log"}, RuntimeError("never-log")):
                audit.emit_diagnostic(code)
            enqueue.assert_not_called()
            audit.emit_diagnostic("gpu_readiness_revoke_failed")
            audit.emit_diagnostic("gpu_quarantine_terminal")
        self.assertEqual(
            enqueue.call_args_list,
            [
                unittest.mock.call(audit._Diagnostic("gpu_readiness_revoke_failed")),
                unittest.mock.call(audit._Diagnostic("gpu_quarantine_terminal")),
            ],
        )

    def test_both_record_types_share_bounded_nonblocking_queue(self):
        records = queue.Queue(maxsize=8)
        with patch.object(audit, "_writer", object()), patch.object(audit, "_records", records):
            audit.emit("quarantine")
            for _ in range(100):
                audit.emit_diagnostic("gpu_readiness_revoke_failed")
            audit.emit("allow")
        self.assertEqual(records.qsize(), 8)
        self.assertEqual(records.get_nowait(), "quarantine")
        while not records.empty():
            self.assertEqual(records.get_nowait(), audit._Diagnostic("gpu_readiness_revoke_failed"))

    def test_flush_waits_until_prior_records_are_written(self):
        # The production worker is intentionally process-long. Exercise it in
        # a child so the test cannot leave a daemon consuming the suite's
        # shared audit queue after its mocks are restored.
        program = textwrap.dedent(
            """
            from unittest.mock import patch
            from cvm.runtime import audit
            with patch.object(audit.Path, 'exists', return_value=False), \
                 patch.object(audit, 'read_json', return_value={
                     'build_id': 'build', 'attestation_policy_id': 'policy'
                 }), \
                 patch.object(audit, 'append'):
                audit.emit('quarantine')
                audit.emit_diagnostic('gpu_quarantine_terminal')
                assert audit.flush(1)
            """
        )
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual([line.split("=", 1)[0] for line in lines], ["CVM_ATTESTATION", "CVM_DIAGNOSTIC"])
        self.assertEqual(json.loads(lines[0].split("=", 1)[1])["decision"], "quarantine")
        self.assertEqual(json.loads(lines[1].split("=", 1)[1])["code"], "gpu_quarantine_terminal")

    def test_flush_is_bounded_without_writer_progress(self):
        records = queue.Queue(maxsize=8)
        with patch.object(audit, "_writer", object()), patch.object(audit, "_records", records):
            audit.emit("allow")
            started = time.monotonic()
            self.assertFalse(audit.flush(0))
            self.assertLess(time.monotonic() - started, 0.1)
        self.assertEqual(records.get_nowait(), "allow")
        self.assertIsInstance(records.get_nowait(), audit._Barrier)

    def test_worker_start_failure_does_not_escape_diagnostic_emitter(self):
        for constructor_fails in (True, False):
            worker = Mock()
            if not constructor_fails:
                worker.start.side_effect = RuntimeError("cannot create worker")
            with (
                self.subTest(constructor_fails=constructor_fails),
                patch.object(audit, "_writer", None),
                patch.object(
                    audit.threading,
                    "Thread",
                    **(
                        {"side_effect": RuntimeError("cannot create worker")}
                        if constructor_fails
                        else {"return_value": worker}
                    ),
                ),
                patch.object(audit, "_records") as records,
            ):
                audit.emit_diagnostic("gpu_readiness_revoke_failed")
                records.put_nowait.assert_not_called()

    def test_diagnostic_output_error_is_advisory(self):
        with (
            patch.object(audit.Path, "exists", return_value=False),
            patch("builtins.print", side_effect=OSError("blocked console")),
            patch.object(audit, "append") as append_log,
        ):
            audit._write(audit._Diagnostic("gpu_readiness_revoke_failed"))
        append_log.assert_not_called()
