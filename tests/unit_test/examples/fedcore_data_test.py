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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.unit_test.examples.fedcore_test_utils import fedcore_import_context


class _MNISTFixture:
    def __init__(self, examples_per_digit: int):
        self.targets = []
        self.images = []
        for digit in range(10):
            for example_index in range(examples_per_digit):
                pixels = np.zeros((28, 28), dtype=np.uint8)
                pixels[4:24, 5 + digit % 4 : 7 + digit % 4] = 160 + example_index % 96
                self.images.append(Image.fromarray(pixels, mode="L"))
                self.targets.append(digit)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index].copy(), self.targets[index]


def _loader(_root, train):
    return _MNISTFixture(30 if train else 12)


def _config(output_dir, scenario="recoverable"):
    with fedcore_import_context():
        from src.data import MNISTDataConfig

        return MNISTDataConfig(
            output_dir=output_dir,
            dataset_root=output_dir / "mnist-cache",
            scenario=scenario,
            train_samples_per_site=16,
            val_samples_per_site=8,
            test_samples_per_site=8,
            proxy_strength=0.75,
            seed=13,
            image_size=64,
        )


def test_mnist_data_are_deterministic_balanced_and_disjoint(tmp_path):
    with fedcore_import_context():
        from src.data import SPLITS, generate_mnist_data, load_manifest

        first = tmp_path / "first"
        second = tmp_path / "second"
        first_summary = generate_mnist_data(_config(first), dataset_loader=_loader)
        generate_mnist_data(_config(second), dataset_loader=_loader)

        assert first_summary["dataset"] == "MNIST"
        assert first_summary["total_examples"] == 96
        seen_sources = set()
        for site_index in range(1, 4):
            site = f"site-{site_index}"
            for split in SPLITS:
                first_records = load_manifest(first / site / f"{split}.jsonl")
                second_records = load_manifest(second / site / f"{split}.jsonl")
                assert first_records == second_records
                assert sum(record["label"] for record in first_records) == len(first_records) // 2
                for record in first_records:
                    source = (record["source_split"], record["source_index"])
                    assert source not in seen_sources
                    seen_sources.add(source)
                    assert record["label"] == int(record["digit"] <= 4)
                    assert record["ocr_label"] == int(record["ocr_digit"] <= 4)
                    assert record["sensor_matches_label"] == (record["ocr_label"] == record["label"])

        assert first_summary["sites"]["site-1"]["splits"]["train"]["image_available"] == 16
        assert first_summary["sites"]["site-2"]["splits"]["train"]["image_available"] == 8
        assert first_summary["sites"]["site-3"]["splits"]["train"]["image_available"] == 0
        image = Image.open(first / "site-1" / "images" / "train-s1-00000.png")
        assert image.mode == "RGB"
        assert image.size == (64, 64)


def test_qwen_sft_records_include_ocr_context_and_match_modality_availability(tmp_path):
    with fedcore_import_context():
        from src.data import generate_mnist_data

        generate_mnist_data(_config(tmp_path), dataset_loader=_loader)
        site_one = json.loads((tmp_path / "site-1" / "train.json").read_text())
        site_three = json.loads((tmp_path / "site-3" / "train.json").read_text())

        assert all("image" in record for record in site_one)
        assert all("<image>" in record["conversations"][0]["value"] for record in site_one)
        assert all("image" not in record for record in site_three)
        assert all("<image>" not in record["conversations"][0]["value"] for record in site_three)
        prompts = [record["conversations"][0]["value"] for record in site_one + site_three]
        assert all("Secondary OCR report:" in prompt for prompt in prompts)
        assert all("estimated digit=" in prompt and "sensor confidence=" in prompt for prompt in prompts)


def test_recoverable_ocr_accuracy_and_confidence_are_informative(tmp_path):
    with fedcore_import_context():
        from src.data import generate_mnist_data, load_manifest

        generate_mnist_data(_config(tmp_path), dataset_loader=_loader)
        for site_index in range(1, 4):
            records = load_manifest(tmp_path / f"site-{site_index}" / "train.jsonl")
            for label in (0, 1):
                class_records = [record for record in records if record["label"] == label]
                matched = [record for record in class_records if record["sensor_matches_label"]]
                mismatched = [record for record in class_records if not record["sensor_matches_label"]]
                assert len(matched) == 6
                assert len(mismatched) == 2
                matched_high = sum(record["sensor_confidence"] == "high" for record in matched) / len(matched)
                mismatched_high = sum(record["sensor_confidence"] == "high" for record in mismatched) / len(mismatched)
                assert matched_high > mismatched_high


def test_uninformative_ocr_and_confidence_are_exactly_independent(tmp_path):
    with fedcore_import_context():
        from src.data import generate_mnist_data, load_manifest

        config = _config(tmp_path, scenario="uninformative")
        generate_mnist_data(config, dataset_loader=_loader)
        for site_index in range(1, 4):
            for split in ("train", "val", "test"):
                records = load_manifest(tmp_path / f"site-{site_index}" / f"{split}.jsonl")
                for label in (0, 1):
                    class_records = [record for record in records if record["label"] == label]
                    assert sum(record["sensor_matches_label"] for record in class_records) == len(class_records) // 2
                    for matches in (False, True):
                        group = [record for record in class_records if record["sensor_matches_label"] == matches]
                        assert sum(record["sensor_confidence"] == "high" for record in group) == len(group) // 2


def test_stratified_mask_uses_round_half_up():
    with fedcore_import_context():
        from src.data import _stratified_mask

        labels = np.asarray([0, 1], dtype=np.int64)
        mask = _stratified_mask(labels, 0.5, np.random.default_rng(7))

    assert mask.tolist() == [True, True]


def test_recoverable_split_rejects_unrealizable_proxy_errors_before_writing(tmp_path):
    with fedcore_import_context():
        from src.data import generate_mnist_data

        output_dir = tmp_path / "too-small"
        config = replace(
            _config(output_dir),
            train_samples_per_site=8,
            proxy_strength=0.9,
        )
        with pytest.raises(ValueError, match="cannot represent both correct and incorrect OCR"):
            generate_mnist_data(config, dataset_loader=_loader)

    assert not output_dir.exists()


def test_dataset_summary_reports_requested_and_realized_proxy_strength(tmp_path):
    with fedcore_import_context():
        from src.data import generate_mnist_data

        summary = generate_mnist_data(_config(tmp_path), dataset_loader=_loader)

    assert summary["proxy_strength_requested"] == 0.75
    assert summary["proxy_strength_realized"] == 0.75


def test_prepare_data_default_output_is_scenario_specific():
    with fedcore_import_context():
        import prepare_data

        recoverable = prepare_data.define_parser().parse_args(["--scenario", "recoverable"])
        uninformative = prepare_data.define_parser().parse_args(["--scenario", "uninformative"])

    assert prepare_data._resolve_output_dir(recoverable) == Path("data/recoverable")
    assert prepare_data._resolve_output_dir(uninformative) == Path("data/uninformative")


def test_manifest_loader_rejects_unknown_schema(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"schema_version": 999}) + "\n")

    with fedcore_import_context():
        from src.data import load_manifest

        with pytest.raises(ValueError, match="expected 2"):
            load_manifest(manifest)
