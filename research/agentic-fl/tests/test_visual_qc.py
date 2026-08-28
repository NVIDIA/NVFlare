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

from agenticfl.data.extractor import apply_automatic_orientation_repair
from agenticfl.data.qc import visual_qc_decision_passed
from agenticfl.job_train import ready_training_clients
from PIL import Image


def _extracted_result(*, transform: str = "as_is", visual_passed: bool = True) -> dict[str, object]:
    return {
        "data": "extracted",
        "verification": {"passed": True},
        "counts": {"by_split": {"train": 2, "validation": 1}},
        "label_orientation": {"selected_transform": transform},
        "visual_qc": {
            "status": "passed" if visual_passed else "failed",
            "passed": visual_passed,
            "reviewed": True,
            "consensus_reached": True,
            "selected_transform": transform,
        },
    }


class VisualQCContractTestCase(unittest.TestCase):
    def test_consensus_transform_repairs_every_prepared_segmentation_mask(self) -> None:
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            site_dir = project_root / "data" / "dataset_fl" / "SITE_A"
            mask_path = site_dir / "masks" / "train" / "sample.png"
            mask_path.parent.mkdir(parents=True)
            mask = Image.new("L", (3, 1), 0)
            mask.putpixel((0, 0), 255)
            mask.save(mask_path)
            (site_dir / "samples.jsonl").write_text(
                json.dumps(
                    {
                        "image": "images/train/sample.png",
                        "mask": "masks/train/sample.png",
                        "split": "train",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (site_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "record_type": "segmentation",
                        "sample_manifest": "samples.jsonl",
                        "label_orientation": {"selected_transform": "as_is"},
                    }
                ),
                encoding="utf-8",
            )
            extraction_result = {
                "client_id": "SITE_A",
                "record_type": "segmentation",
                "extraction": {
                    "output_root": "data/dataset_fl",
                    "client_folder": "SITE_A",
                },
                "label_orientation": {"selected_transform": "as_is"},
            }
            decision = {
                "status": "failed",
                "passed": False,
                "reviewed": True,
                "consensus_reached": True,
                "selected_transform": "hflip",
            }

            repaired_result, repaired_decision = apply_automatic_orientation_repair(
                extraction_result=extraction_result,
                decision=decision,
                project_root=project_root,
            )

            with Image.open(mask_path) as repaired_mask:
                self.assertEqual([repaired_mask.getpixel((x, 0)) for x in range(3)], [0, 0, 255])
            self.assertTrue(repaired_decision["passed"])
            self.assertEqual(repaired_decision["orientation_repair"]["repaired_mask_count"], 1)
            self.assertEqual(repaired_decision["orientation_repair"]["applied_transform"], "hflip")
            self.assertEqual(repaired_result["label_orientation"]["selected_transform"], "as_is")
            self.assertEqual(repaired_result["label_orientation"]["applied_transform"], "hflip")
            self.assertTrue(
                visual_qc_decision_passed(
                    repaired_decision,
                    label_orientation=repaired_result["label_orientation"],
                )
            )

    def test_orientation_repair_must_match_the_persisted_visual_decision(self) -> None:
        decision = _extracted_result(transform="hflip")["visual_qc"]
        self.assertTrue(
            visual_qc_decision_passed(
                decision,
                label_orientation={"selected_transform": "hflip"},
            )
        )
        self.assertFalse(
            visual_qc_decision_passed(
                decision,
                label_orientation={"selected_transform": "as_is"},
            )
        )
        unknown_decision = _extracted_result(transform="transpose")["visual_qc"]
        self.assertFalse(
            visual_qc_decision_passed(
                unknown_decision,
                label_orientation={"selected_transform": "transpose"},
            )
        )

    def test_segmentation_training_fails_closed_without_visual_pass(self) -> None:
        summary = {
            "extracted_clients": ["SITE_A", "SITE_B"],
            "visual_qc": {"passed_clients": ["SITE_A"]},
            "extraction_results": {
                "SITE_A": _extracted_result(),
                "SITE_B": _extracted_result(visual_passed=False),
            },
        }
        self.assertEqual(ready_training_clients(summary, task="optic disc segmentation"), ["SITE_A"])

    def test_classification_does_not_require_spatial_alignment_qc(self) -> None:
        result = _extracted_result(visual_passed=False)
        result.pop("visual_qc")
        summary = {
            "extracted_clients": ["SITE_A"],
            "extraction_results": {"SITE_A": result},
        }
        self.assertEqual(
            ready_training_clients(summary, task="binary image classification"),
            ["SITE_A"],
        )


if __name__ == "__main__":
    unittest.main()
