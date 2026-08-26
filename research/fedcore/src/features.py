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

"""Feature-cache creation shared by the Qwen and mock backends."""

import json
from pathlib import Path

import numpy as np
import torch
from src.data import SPLITS, load_manifest

CACHE_SCHEMA_VERSION = 1


def mock_features(records: list[dict], hidden_dim: int = 16) -> dict:
    labels = torch.tensor([record["label"] for record in records], dtype=torch.long)
    image_available = torch.tensor([record["image_available"] for record in records], dtype=torch.bool)
    features = []
    missing_logits = []
    full_logits = []
    for record in records:
        proxy_signed = 1.0 if int(record["proxy_label"]) == 1 else -1.0
        vector = np.zeros(hidden_dim, dtype=np.float32)
        vector[0] = proxy_signed
        if hidden_dim > 1:
            vector[1] = 1.0
        features.append(vector)
        # The mock frozen predictor does not know the arbitrary KAPPA/SIGMA mapping.
        missing_logits.append(0.0)
        full_logits.append(3.0 if int(record["label"]) == 1 else -3.0)
    full_tensor = torch.tensor(full_logits, dtype=torch.float32)
    full_tensor[~image_available] = torch.nan
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "example_ids": [record["example_id"] for record in records],
        "labels": labels,
        "image_available": image_available,
        "paired_mask": image_available.clone(),
        "missing_features": torch.tensor(np.stack(features), dtype=torch.float32),
        "missing_logits": torch.tensor(missing_logits, dtype=torch.float32),
        "full_logits": full_tensor,
    }


def create_feature_cache(
    data_dir: Path,
    cache_dir: Path,
    backend: str,
    qwen_extractor=None,
    mock_hidden_dim: int = 16,
) -> dict:
    data_dir = data_dir.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "backend": backend,
        "data_dir": str(data_dir),
        "sites": {},
        "target_modality": "image",
    }
    input_dim = None
    for site_index in range(1, 4):
        site = f"site-{site_index}"
        metadata["sites"][site] = {}
        for split in SPLITS:
            records = load_manifest(data_dir / site / f"{split}.jsonl")
            if backend == "mock":
                payload = mock_features(records, hidden_dim=mock_hidden_dim)
            elif backend == "qwen":
                if qwen_extractor is None:
                    raise ValueError("qwen_extractor is required for backend='qwen'.")
                payload = qwen_extractor.extract(records, data_dir=data_dir)
            else:
                raise ValueError(f"Unsupported feature backend: {backend}")
            current_dim = int(payload["missing_features"].shape[1])
            if input_dim is None:
                input_dim = current_dim
            elif input_dim != current_dim:
                raise ValueError(f"Feature dimension changed from {input_dim} to {current_dim} for {site}/{split}.")
            site_cache_dir = cache_dir / site
            site_cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, site_cache_dir / f"{split}.pt")
            metadata["sites"][site][split] = {
                "examples": len(records),
                "paired_examples": int(payload["paired_mask"].sum().item()),
                "missing_examples": int((~payload["image_available"]).sum().item()),
            }
    metadata["input_dim"] = int(input_dim or 0)
    if qwen_extractor is not None:
        metadata.update(qwen_extractor.metadata())
    with (cache_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    return metadata


def load_cache_split(cache_dir: Path, site: str, split: str) -> dict:
    path = cache_dir.expanduser().resolve() / site / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "example_ids",
        "labels",
        "image_available",
        "paired_mask",
        "missing_features",
        "missing_logits",
        "full_logits",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Cache {path} is missing keys: {sorted(missing)}")
    return payload
