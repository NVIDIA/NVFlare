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

import numpy as np
import pytest
import torch

from tests.unit_test.examples.fedcore_test_utils import fedcore_import_context


def test_mock_cache_is_modality_neutral_and_preserves_missing_clients(tmp_path, monkeypatch):
    with fedcore_import_context():
        from src.data import SyntheticDataConfig, generate_synthetic_data
        from src.features import create_feature_cache, load_cache_split

        data_dir = tmp_path / "data"
        cache_dir = tmp_path / "cache"
        generate_synthetic_data(
            SyntheticDataConfig(
                output_dir=data_dir,
                train_samples_per_site=6,
                val_samples_per_site=4,
                test_samples_per_site=4,
                image_size=64,
            )
        )
        metadata = create_feature_cache(data_dir, cache_dir, backend="mock", mock_hidden_dim=8)
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
