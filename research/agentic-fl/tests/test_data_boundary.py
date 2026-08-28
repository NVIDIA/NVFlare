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
from pathlib import Path
from tempfile import TemporaryDirectory

from agenticfl.agents.local_adapter import sanitize_adapter_diagnostic
from agenticfl.data.contracts.classification import label_value
from agenticfl.data.parser import DataParserConfig, list_client_ids, parse_site_dataset
from PIL import Image


class DataBoundaryTestCase(unittest.TestCase):
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
