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

import pathlib
import sys
import tempfile
import unittest

EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[5] / "examples" / "advanced" / "medgemma"
_EXAMPLE_DIR_STR = str(EXAMPLE_DIR)
_ADDED_EXAMPLE_DIR = False
if _EXAMPLE_DIR_STR not in sys.path:
    sys.path.insert(0, _EXAMPLE_DIR_STR)
    _ADDED_EXAMPLE_DIR = True

from data_utils import (  # noqa: E402
    ALT_TISSUE_LABELS,
    TISSUE_CLASSES,
    collect_image_records,
    format_training_example,
    parse_prediction_label,
    sample_records,
    split_records_for_clients,
)

if _ADDED_EXAMPLE_DIR:
    sys.path.remove(_EXAMPLE_DIR_STR)


class MedGemmaDataUtilsTest(unittest.TestCase):
    def test_format_training_example_builds_multimodal_messages(self):
        example = {"image": "ADI/sample.tif", "label": 0, "label_name": TISSUE_CLASSES[0]}

        formatted = format_training_example(example)

        self.assertEqual(formatted["messages"][0]["role"], "user")
        self.assertEqual(formatted["messages"][0]["content"][0]["type"], "image")
        self.assertEqual(formatted["messages"][1]["content"][0]["text"], TISSUE_CLASSES[0])

    def test_parse_prediction_label_accepts_primary_and_alt_formats(self):
        self.assertEqual(parse_prediction_label("Answer: A: adipose"), 0)
        self.assertEqual(parse_prediction_label("I would choose (B) background."), 1)
        self.assertEqual(parse_prediction_label("No known label"), -1)

    def test_alt_tissue_labels_match_parenthesized_format(self):
        self.assertEqual(ALT_TISSUE_LABELS[TISSUE_CLASSES[1]], "(B) background")

    def test_split_records_for_clients_creates_non_overlapping_shards(self):
        records = [
            {"image": f"img_{idx}.tif", "label": idx % len(TISSUE_CLASSES), "label_name": TISSUE_CLASSES[idx % 9]}
            for idx in range(12)
        ]

        site_splits = split_records_for_clients(
            records=records,
            num_clients=3,
            samples_per_client=4,
            validation_size_per_client=1,
            seed=13,
        )

        all_images = []
        for site_name in ("site-1", "site-2", "site-3"):
            self.assertEqual(len(site_splits[site_name]["train"]), 3)
            self.assertEqual(len(site_splits[site_name]["validation"]), 1)
            all_images.extend(record["image"] for record in site_splits[site_name]["train"])
            all_images.extend(record["image"] for record in site_splits[site_name]["validation"])
        self.assertEqual(len(all_images), len(set(all_images)))

    def test_collect_image_records_discovers_known_label_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = pathlib.Path(temp_dir)
            adipose_dir = dataset_root / "ADI"
            tumor_dir = dataset_root / "TUM"
            adipose_dir.mkdir()
            tumor_dir.mkdir()
            (adipose_dir / "a.tif").write_bytes(b"")
            (tumor_dir / "b.tif").write_bytes(b"")

            records = collect_image_records(str(dataset_root))

        self.assertEqual({record["raw_label"] for record in records}, {"ADI", "TUM"})
        self.assertEqual({record["label_name"] for record in records}, {TISSUE_CLASSES[0], TISSUE_CLASSES[8]})

    def test_sample_records_is_deterministic_and_respects_limit(self):
        records = [{"image": f"img_{idx}.tif"} for idx in range(10)]

        sampled_once = sample_records(records, max_samples=4, seed=7)
        sampled_twice = sample_records(records, max_samples=4, seed=7)

        self.assertEqual(sampled_once, sampled_twice)
        self.assertEqual(len(sampled_once), 4)


if __name__ == "__main__":
    unittest.main()
