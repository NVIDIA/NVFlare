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

import json

from src.data import SPLITS, SyntheticDataConfig, generate_synthetic_data, load_manifest


def test_synthetic_data_are_deterministic_and_disjoint(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = dict(
        train_samples_per_site=8,
        val_samples_per_site=4,
        test_samples_per_site=4,
        proxy_strength=0.9,
        seed=13,
        image_size=64,
    )
    summary = generate_synthetic_data(SyntheticDataConfig(output_dir=first, **config))
    generate_synthetic_data(SyntheticDataConfig(output_dir=second, **config))

    assert summary["total_examples"] == 48
    seen = set()
    for site_index in range(1, 4):
        site = f"site-{site_index}"
        for split in SPLITS:
            first_records = load_manifest(first / site / f"{split}.jsonl")
            second_records = load_manifest(second / site / f"{split}.jsonl")
            assert first_records == second_records
            ids = {record["example_id"] for record in first_records}
            assert not seen.intersection(ids)
            seen.update(ids)

    assert summary["sites"]["site-1"]["splits"]["train"]["image_available"] == 8
    assert summary["sites"]["site-2"]["splits"]["train"]["image_available"] == 4
    assert summary["sites"]["site-3"]["splits"]["train"]["image_available"] == 0


def test_qwen_sft_records_match_modality_availability(tmp_path):
    generate_synthetic_data(
        SyntheticDataConfig(
            output_dir=tmp_path,
            train_samples_per_site=4,
            val_samples_per_site=2,
            test_samples_per_site=2,
            seed=7,
            image_size=64,
        )
    )
    site_one = json.loads((tmp_path / "site-1" / "train.json").read_text())
    site_three = json.loads((tmp_path / "site-3" / "train.json").read_text())
    assert all("image" in record for record in site_one)
    assert all("<image>" in record["conversations"][0]["value"] for record in site_one)
    assert all("image" not in record for record in site_three)
    assert all("<image>" not in record["conversations"][0]["value"] for record in site_three)


def test_uninformative_proxy_is_balanced_within_each_class(tmp_path):
    generate_synthetic_data(
        SyntheticDataConfig(
            output_dir=tmp_path,
            train_samples_per_site=16,
            val_samples_per_site=8,
            test_samples_per_site=8,
            proxy_strength=0.5,
            seed=7,
            image_size=64,
        )
    )
    for site_index in range(1, 4):
        records = load_manifest(tmp_path / f"site-{site_index}" / "train.jsonl")
        for label in (0, 1):
            class_records = [record for record in records if record["label"] == label]
            assert sum(record["proxy_matches_label"] for record in class_records) == len(class_records) // 2
