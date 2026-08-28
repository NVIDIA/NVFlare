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
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_references import prepare_reference_bundle


class ReferencePreparationTestCase(unittest.TestCase):
    def test_references_are_selected_from_existing_prepared_records(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 12), "white").save(root / "image.png")
            Image.new("L", (20, 12), 255).save(root / "mask.png")
            segmentation = root / "segmentation.jsonl"
            segmentation.write_text(
                json.dumps({"image": "image.png", "mask": "mask.png", "split": "train"}) + "\n",
                encoding="utf-8",
            )
            detection = root / "detection.jsonl"
            detection.write_text(
                json.dumps(
                    {
                        "image": "image.png",
                        "boxes": [[2, 2, 10, 10]],
                        "labels": [1],
                        "bbox_format": "xyxy_abs",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            classification = root / "classification.json"
            classification.write_text(
                json.dumps({"training": [{"image": "image.png", "label": 1}]}),
                encoding="utf-8",
            )
            config = root / "sources.json"
            config.write_text(
                json.dumps(
                    {
                        "examples": [
                            {
                                "task_key": "cup_seg",
                                "sample_manifest": str(segmentation),
                            },
                            {
                                "task_key": "disc_seg",
                                "sample_manifest": str(segmentation),
                            },
                            {
                                "task_key": "disc_detec",
                                "sample_manifest": str(detection),
                                "bbox_format": "xyxy_abs",
                            },
                            {
                                "task_key": "glaucoma_cls",
                                "sample_manifest": str(classification),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = prepare_reference_bundle(config, root / "task_example")

            self.assertEqual(
                [item["task_key"] for item in manifest["examples"]],
                [
                    "cup_seg",
                    "disc_seg",
                    "disc_detec",
                    "glaucoma_cls",
                ],
            )
            self.assertTrue(all(item["source"]["kind"] == "prepared_client_record" for item in manifest["examples"]))
            self.assertTrue(all(item["asset_sha256"] for item in manifest["examples"]))


if __name__ == "__main__":
    unittest.main()
