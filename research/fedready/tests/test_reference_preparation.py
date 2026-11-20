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
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fedready.agents import references
from prepare_references import _write_mask_png, prepare_reference_bundle


class ReferencePreparationTestCase(unittest.TestCase):
    def test_references_are_selected_from_existing_prepared_records(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 12), "white").save(root / "image.png")
            mask = Image.new("L", (20, 12), 0)
            mask.putpixel((5, 5), 17)
            mask.save(root / "mask.png")
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
            for task_key in ("cup_seg", "disc_seg"):
                with Image.open(root / "task_example" / task_key / "mask.png") as prepared_mask:
                    self.assertEqual({value for _, value in prepared_mask.getcolors(maxcolors=3) or []}, {0, 255})

            with mock.patch.object(references, "TASK_EXAMPLE_DIR", root / "task_example"):
                with mock.patch.dict(os.environ, {"FEDREADY_TASK_EXAMPLE_ID": "disc_seg"}):
                    self.assertEqual(references.task_example_image_path(), root / "task_example/disc_seg/image.png")
                    self.assertIsNone(references.adapter_preflight_task_example_path(record_type="classification"))
                    context = references.task_example_context()
                    snapshot = root / "request" / "task_example"
                    references.write_task_example_snapshot(snapshot, context)
                    copied = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()}
                    self.assertEqual(copied, {"manifest.json", "disc_seg/image.png", "disc_seg/mask.png"})
                    selected = json.loads((snapshot / "manifest.json").read_text())["examples"]
                    self.assertEqual([item["task_key"] for item in selected], ["disc_seg"])
                    (root / "task_example/disc_seg/image.png").write_bytes(b"changed")
                    self.assertIsNone(references.task_example_image_path())
                    with self.assertRaisesRegex(ValueError, "valid task example"):
                        references.write_task_example_snapshot(root / "bad_request" / "task_example", context)
                    with mock.patch.object(references, "TASK_EXAMPLE_DIR", snapshot):
                        self.assertIsNotNone(references.task_example_image_path())
                for task_key in ("", "disc", "unknown"):
                    with mock.patch.dict(os.environ, {"FEDREADY_TASK_EXAMPLE_ID": task_key}):
                        self.assertIsNone(references.task_example_image_path())

    def test_reference_mask_rejects_multiclass_values(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mask.png"
            mask = Image.new("L", (3, 1))
            mask.putdata([0, 1, 2])
            mask.save(source)

            with self.assertRaisesRegex(ValueError, "must be binary"):
                _write_mask_png(source, root / "prepared.png", expected_size=(3, 1))


if __name__ == "__main__":
    unittest.main()
