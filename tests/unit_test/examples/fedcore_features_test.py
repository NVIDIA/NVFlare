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

import numpy as np
import pytest
import torch

from tests.unit_test.examples.fedcore_test_utils import fedcore_import_context


def _valid_payload() -> dict:
    return {
        "schema_version": 1,
        "example_ids": ["paired", "unpaired"],
        "labels": torch.tensor([1, 0], dtype=torch.long),
        "image_available": torch.tensor([True, False]),
        "paired_mask": torch.tensor([True, False]),
        "missing_features": torch.zeros((2, 2), dtype=torch.float32),
        "missing_logits": torch.zeros(2, dtype=torch.float32),
        "full_logits": torch.tensor([1.0, torch.nan], dtype=torch.float32),
    }


def _save_payload(tmp_path, payload) -> None:
    cache_path = tmp_path / "site-1" / "train.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def test_qwen_cache_is_modality_neutral_and_preserves_missing_clients(tmp_path, monkeypatch):
    with fedcore_import_context():
        from src.features import create_feature_cache, load_cache_split

        data_dir = tmp_path / "data"
        cache_dir = tmp_path / "cache"
        for site_index in range(1, 4):
            site = f"site-{site_index}"
            available = site_index == 1
            for split in ("train", "val", "test"):
                path = data_dir / site / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "schema_version": 2,
                    "example_id": f"{site}-{split}",
                    "label": site_index % 2,
                    "image_available": available,
                }
                path.write_text(json.dumps(record) + "\n")

        class Extractor:
            @staticmethod
            def extract(records, data_dir):
                del data_dir
                available = torch.tensor([record["image_available"] for record in records], dtype=torch.bool)
                full_logits = torch.tensor([1.0 if value else torch.nan for value in available])
                return {
                    "schema_version": 1,
                    "example_ids": [record["example_id"] for record in records],
                    "labels": torch.tensor([record["label"] for record in records]),
                    "image_available": available,
                    "paired_mask": available.clone(),
                    "missing_features": torch.zeros((len(records), 8)),
                    "missing_logits": torch.zeros(len(records)),
                    "full_logits": full_logits,
                }

            @staticmethod
            def metadata():
                return {"model_name_or_path": "test-qwen"}

        metadata = create_feature_cache(data_dir, cache_dir, qwen_extractor=Extractor())
        original_load = torch.load
        load_kwargs = []

        def record_load(*args, **kwargs):
            load_kwargs.append(kwargs)
            return original_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", record_load)
        site_one = load_cache_split(cache_dir, "site-1", "train")
        site_three = load_cache_split(cache_dir, "site-3", "train")
        assert metadata["target_modality"] == "image"
        assert metadata["input_dim"] == 8
        assert bool(site_one["paired_mask"].all())
        assert not bool(site_three["paired_mask"].any())
        assert bool(torch.isnan(site_three["full_logits"]).all())
        assert load_kwargs
        assert all(kwargs["weights_only"] is True for kwargs in load_kwargs)


def test_cache_loader_rejects_numpy_arrays_with_actionable_error(tmp_path):
    payload = {
        "schema_version": 1,
        "example_ids": ["example-1"],
        "labels": np.array([1]),
        "image_available": np.array([True]),
        "paired_mask": np.array([True]),
        "missing_features": np.zeros((1, 2), dtype=np.float32),
        "missing_logits": np.zeros(1, dtype=np.float32),
        "full_logits": np.ones(1, dtype=np.float32),
    }
    cache_path = tmp_path / "site-1" / "train.pt"
    cache_path.parent.mkdir(parents=True)
    torch.save(payload, cache_path)

    with fedcore_import_context():
        from src.features import load_cache_split

        with pytest.raises(ValueError, match="convert NumPy arrays"):
            load_cache_split(tmp_path, "site-1", "train")


def test_cache_loader_rejects_inconsistent_tensor_shapes(tmp_path):
    payload = {
        "schema_version": 1,
        "example_ids": ["example-1"],
        "labels": torch.tensor([1, 0]),
        "image_available": torch.tensor([True]),
        "paired_mask": torch.tensor([True]),
        "missing_features": torch.zeros((1, 2)),
        "missing_logits": torch.zeros(1),
        "full_logits": torch.ones(1),
    }
    cache_path = tmp_path / "site-1" / "train.pt"
    cache_path.parent.mkdir(parents=True)
    torch.save(payload, cache_path)

    with fedcore_import_context():
        from src.features import load_cache_split

        with pytest.raises(ValueError, match="shapes inconsistent"):
            load_cache_split(tmp_path, "site-1", "train")


def test_cache_loader_rejects_float_labels(tmp_path):
    payload = _valid_payload()
    payload["labels"] = payload["labels"].float()
    _save_payload(tmp_path, payload)

    with fedcore_import_context():
        from src.features import load_cache_split

        with pytest.raises(ValueError, match="integer dtype"):
            load_cache_split(tmp_path, "site-1", "train")


def test_cache_loader_rejects_unknown_schema_version(tmp_path):
    payload = _valid_payload()
    payload["schema_version"] = 999
    _save_payload(tmp_path, payload)

    with fedcore_import_context():
        from src.features import load_cache_split

        with pytest.raises(ValueError, match="expected 1"):
            load_cache_split(tmp_path, "site-1", "train")


def test_cache_metadata_is_required_and_schema_checked(tmp_path):
    with fedcore_import_context():
        from src.features import load_cache_metadata

        with pytest.raises(FileNotFoundError, match="metadata not found"):
            load_cache_metadata(tmp_path)
        (tmp_path / "metadata.json").write_text(json.dumps({"schema_version": 999, "input_dim": 2, "sites": {}}))
        with pytest.raises(ValueError, match="expected 1"):
            load_cache_metadata(tmp_path)
        (tmp_path / "metadata.json").write_text(json.dumps({"schema_version": 1, "input_dim": 2, "sites": {}}))
        assert load_cache_metadata(tmp_path)["input_dim"] == 2


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("paired_mask", torch.tensor([False, False]), "must be identical"),
        ("full_logits", torch.tensor([torch.nan, torch.nan]), "finite for paired"),
        ("full_logits", torch.tensor([1.0, 0.0]), "NaN for unpaired"),
    ],
)
def test_cache_loader_rejects_inconsistent_paired_logits(tmp_path, field, value, match):
    payload = _valid_payload()
    payload[field] = value
    _save_payload(tmp_path, payload)

    with fedcore_import_context():
        from src.features import load_cache_split

        with pytest.raises(ValueError, match=match):
            load_cache_split(tmp_path, "site-1", "train")


def test_cache_loader_reports_corrupt_files_without_numpy_diagnosis(tmp_path):
    cache_path = tmp_path / "site-1" / "train.pt"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"this is not a PyTorch cache")

    with fedcore_import_context():
        from src.features import load_cache_split

        with pytest.raises(ValueError, match="corrupt or contains objects") as error:
            load_cache_split(tmp_path, "site-1", "train")
    assert "NumPy" not in str(error.value)
