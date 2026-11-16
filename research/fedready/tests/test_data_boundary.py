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

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, patch

from fedready.agents import ClientAgent
from fedready.agents.bridge import CodexWorkerBackend
from fedready.agents.local_adapter import sanitize_adapter_diagnostic
from fedready.client import FedReadyTaskQueryExecutor
from fedready.data.contracts.classification import label_value
from fedready.data.extractor import _run_generated_data_materializer
from fedready.data.parser import DataParserConfig, list_client_ids, parse_site_dataset
from fedready.flare.channel import (
    HEADER_SESSION_ID,
    HEADER_TASK_ID,
    FlareMessage,
    from_shareable,
    set_round_headers,
    to_shareable,
)
from fedready.utils.io import contained_path
from fedready.utils.logging import FlowLogger, payload_digest
from PIL import Image

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.signal import Signal


class _CapturingAgentBackend:
    def __init__(self) -> None:
        self.prompt: str | None = None
        self.context: dict[str, Any] | None = None

    def request(self, **kwargs: Any) -> dict[str, Any]:
        self.prompt = kwargs["prompt"]
        context = kwargs["context"]
        self.context = context
        return context["output_template"]


def _generated_materializer(source: str) -> dict[str, Any]:
    source_sha = payload_digest(source)
    return {
        "schema_version": "fedready.generated_data_materializer.v1",
        "status": "implemented",
        "entry_script": "materializer.py",
        "source_files": [
            {
                "path": "materializer.py",
                "content": source,
                "sha256": source_sha,
            }
        ],
        "source_file_count": 1,
        "source_digest": payload_digest(
            {
                "entry_script": "materializer.py",
                "source_files": [{"path": "materializer.py", "sha256": source_sha}],
            }
        ),
    }


class DataBoundaryTestCase(unittest.TestCase):
    def test_guardrail_preserves_denials_and_rejects_invalid_redactions(self) -> None:
        payload = {
            "schema_version": "fedready.client_response.v1",
            "client_id": "SITE_A",
            "note": "sensitive",
            "notes": ["[redacted-by-guardrail]"],
        }
        for allowed, fields, expected in (
            (False, ["note"], False),
            (False, ["does_not_exist"], False),
            (True, ["does_not_exist"], False),
            (True, ["note", "does_not_exist"], False),
            (True, ["client_id"], False),
            (True, ["notes.0"], False),
            (True, ["note"], True),
        ):
            with self.subTest(allowed=allowed, fields=fields):
                backend = Mock()
                backend.request.return_value = {
                    "schema_version": "fedready.guardrail_decision.v1",
                    "allowed": allowed,
                    "reason_code": "TEST_DECISION",
                    "reason": "test decision",
                    "redacted_fields": fields,
                }
                decision = ClientAgent("SITE_A", backend).authorize_outgoing_profile_response(response=payload)
                self.assertEqual(decision.allowed, expected)
                if not allowed:
                    self.assertEqual(decision.reason_code, "TEST_DECISION")
                self.assertEqual(payload["note"], "sensitive")

    def test_task_envelope_rejects_path_ids_before_client_writes(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "runs"
            executor = FedReadyTaskQueryExecutor(site_meta_path="unused.json", output_dir=str(output))
            executor.log_exception = Mock()
            for header in (HEADER_SESSION_ID, HEADER_TASK_ID):
                for value in ("/tmp/escape", "../escape", "a/../escape", "..", "C:\\escape", "a\\escape"):
                    with self.subTest(header=header, value=value):
                        message = to_shareable(FlareMessage("session", "correlation", "schema", "task", {}))
                        message.set_header(header, value)
                        set_round_headers(message, current_round=1, total_rounds=2, action="profile")
                        with self.assertRaises(ValueError):
                            from_shareable(message)
                        reply = executor.execute(executor.task_name, message, Mock(), Signal())
                        self.assertEqual(reply.get_return_code(), ReturnCode.BAD_TASK_DATA)
                        self.assertFalse(output.exists())

    def test_workflow_paths_reject_escaping_symlinks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            outside = Path(tmp) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "session").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                contained_path(root, "session")
            (root / "session").unlink()
            adapter_parent = root / "session" / "sites" / "SITE_A" / "local_adapters"
            adapter_parent.mkdir(parents=True)
            (adapter_parent / "task").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                contained_path(root, "session", "sites", "SITE_A", "local_adapters", "task")
            (root / "session" / "server").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                FlowLogger(root / "session")
            (root / "session" / "agent_mailbox").symlink_to(outside, target_is_directory=True)
            backend = CodexWorkerBackend(root / "session", session_id="session")
            with patch("fedready.agents.bridge.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    backend.request(
                        role="client_agent",
                        action="SMOKE",
                        phase="data_profile",
                        input_schema="in",
                        output_schema="out",
                        prompt="test",
                        context={},
                        site_id="SITE_A",
                    )
                run.assert_not_called()
            self.assertEqual(list(outside.iterdir()), [])

    def test_client_guardrail_receives_exact_generated_materializer_source(self) -> None:
        backend = _CapturingAgentBackend()
        source = 'Path("/private/client-data").read_bytes()\n'
        materializer = {
            "schema_version": "fedready.generated_data_materializer.v1",
            "entry_script": "materializer.py",
            "source_digest": "sha256:bundle",
            "source_file_count": 1,
            "source_files": [
                {
                    "path": "materializer.py",
                    "content": source,
                    "sha256": "sha256:file",
                }
            ],
        }

        decision = ClientAgent("SITE_A", backend).authorize_extraction(
            policy={
                "schema_version": "fedready.site_extraction_policy.v1",
                "client_id": "SITE_A",
                "generated_data_materializer": materializer,
            }
        )

        self.assertTrue(decision.allowed)
        self.assertIsNotNone(backend.context)
        reviewed = backend.context["payload_for_review"]["generated_data_materializer"]
        self.assertEqual(reviewed, materializer)
        self.assertEqual(reviewed["source_files"][0]["content"], source)
        self.assertNotIn("source_files_redacted", reviewed)
        self.assertIn("inspect every source_files[].content entry", backend.prompt or "")

    def test_generated_materializer_rejects_unbound_source_before_execution(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            marker = root / "materializer-executed"
            content = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
            materializer = _generated_materializer(content)
            cases: list[tuple[str, dict[str, Any], str]] = []

            missing_file_digest = deepcopy(materializer)
            missing_file_digest["source_files"][0].pop("sha256")
            cases.append(("missing file digest", missing_file_digest, "source checksum missing"))

            mismatched_file_digest = deepcopy(materializer)
            mismatched_file_digest["source_files"][0]["sha256"] = payload_digest("different source")
            cases.append(("mismatched file digest", mismatched_file_digest, "source checksum mismatch"))

            mismatched_bundle_digest = deepcopy(materializer)
            mismatched_bundle_digest["source_digest"] = payload_digest({"different": "bundle"})
            cases.append(("mismatched bundle digest", mismatched_bundle_digest, "source_digest mismatch"))

            missing_bundle_digest = deepcopy(materializer)
            missing_bundle_digest.pop("source_digest")
            cases.append(("missing bundle digest", missing_bundle_digest, "source_digest missing"))

            mismatched_source_count = deepcopy(materializer)
            mismatched_source_count["source_file_count"] = 2
            cases.append(("mismatched source count", mismatched_source_count, "source_file_count mismatch"))

            for name, candidate, expected_error in cases:
                with self.subTest(name=name):
                    output_dir = root / name.replace(" ", "-")
                    result = _run_generated_data_materializer(
                        output_dir=output_dir,
                        adapter_manifest={},
                        policy={},
                        generated_contract={},
                        generated_materializer=candidate,
                    )

                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["issue_code"], "GENERATED_MATERIALIZER_RUNTIME_ERROR")
                    self.assertIn(expected_error, result["local_diagnostic"])
                    self.assertFalse((output_dir / "_generated_materializer_runtime" / "materializer.py").exists())
                    self.assertFalse(marker.exists())

    def test_generated_materializer_executes_digest_bound_source(self) -> None:
        source = """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--report-path", required=True)
args, _ = parser.parse_known_args()
Path(args.report_path).write_text(
    json.dumps({"schema_version": "fedready.generated_materializer_report.v1", "status": "passed"}),
    encoding="utf-8",
)
"""
        with TemporaryDirectory() as tmp:
            result = _run_generated_data_materializer(
                output_dir=Path(tmp).resolve(),
                adapter_manifest={},
                policy={},
                generated_contract={},
                generated_materializer=_generated_materializer(source),
            )

        self.assertEqual(result["status"], "passed")

    def test_classification_label_parser_rejects_booleans(self) -> None:
        self.assertIsNone(label_value(True))
        self.assertIsNone(label_value(False))
        self.assertEqual(label_value(0), 0)
        self.assertEqual(label_value("1"), 1)

    def test_adapter_diagnostic_redacts_long_hyphenated_identifier(self) -> None:
        identifier = "A-" + "--" * 10_000 + "1234"

        self.assertEqual(
            sanitize_adapter_diagnostic(identifier, private_roots=[]),
            "[redacted-identifier]",
        )

    def test_site_registry_rejects_unsafe_or_duplicate_client_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            meta = Path(tmp) / "site-meta.json"
            for client_ids, error in (
                (["../outside"], "invalid client_id"),
                (["SITE_A", "SITE_A"], "duplicate client_id"),
                ([""], "invalid client_id"),
            ):
                with self.subTest(client_ids=client_ids):
                    meta.write_text(
                        json.dumps(
                            {
                                "clients": [
                                    {"client_id": client_id, "data_path": f"data/{index}"}
                                    for index, client_id in enumerate(client_ids)
                                ]
                            }
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, error):
                        list_client_ids(meta)

    def test_site_registry_resolves_local_data_without_sharing_paths_or_names(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            project = Path(tmp)
            dataset = project / "private" / "site_a"
            for label in ("reviewer_alice", "patient_ab12"):
                label_dir = dataset / "train" / label
                label_dir.mkdir(parents=True)
                Image.new("RGB", (16, 12), "white").save(label_dir / "case.png")
            meta = project / "site-meta.json"
            meta.write_text(
                json.dumps(
                    {
                        "client_count": 1,
                        "clients": [{"client_id": "SITE_A", "data_path": "private/site_a"}],
                    }
                ),
                encoding="utf-8",
            )

            profile = parse_site_dataset(meta, "SITE_A", config=DataParserConfig(min_count=1))
            payload = json.dumps(profile).lower()

            self.assertEqual(list_client_ids(meta)["clients"], [{"client_id": "SITE_A"}])
            self.assertTrue(profile["site_meta"]["data_path_redacted"])
            self.assertNotIn("private/site_a", payload)
            self.assertNotIn("reviewer_alice", payload)
            self.assertNotIn("patient_ab12", payload)


if __name__ == "__main__":
    unittest.main()
