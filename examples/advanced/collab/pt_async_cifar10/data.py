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

"""Prepared-data helpers shared by the recipe, server, and clients."""

import json
from pathlib import Path

import numpy as np

MANIFEST_FILE = "manifest.json"
SPLITS_DIR = "splits"


def load_manifest(data_root: str) -> dict:
    manifest_path = Path(data_root).expanduser().resolve() / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared-data manifest not found at '{manifest_path}'. "
            f"Run `python prepare_data.py --data-root {data_root}` first."
        )
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("dataset") != "CIFAR10" or manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported prepared-data manifest: {manifest_path}")
    return manifest


def split_path(data_root: str, site_name: str) -> Path:
    return Path(data_root).expanduser().resolve() / SPLITS_DIR / f"{site_name}.npy"


def validate_prepared_data(data_root: str) -> dict:
    manifest = load_manifest(data_root)
    num_clients = manifest.get("num_clients")
    if not isinstance(num_clients, int) or num_clients < 1:
        raise ValueError("Prepared-data manifest has an invalid num_clients value")

    missing = [str(split_path(data_root, f"site-{i}")) for i in range(1, num_clients + 1)]
    missing = [path for path in missing if not Path(path).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise FileNotFoundError(f"Prepared client splits are missing: {preview}{suffix}")
    return manifest


def make_client_indices(
    class_indices: dict[int, np.ndarray],
    train_pool: np.ndarray,
    subset_size: int,
    dirichlet_alpha: float,
    seed: int,
) -> np.ndarray:
    """Sample one deterministic logical-client subset.

    Logical-client subsets intentionally may overlap. This models a large
    logical population whose participants independently sample local data from
    the smaller common CIFAR-10 training pool.
    """

    rng = np.random.default_rng(seed)
    class_probs = rng.dirichlet(np.full(10, dirichlet_alpha))
    counts = np.floor(class_probs * subset_size).astype(int)
    remainder = subset_size - int(counts.sum())
    if remainder:
        for class_id in np.argsort(class_probs)[-remainder:]:
            counts[class_id] += 1

    selected = []
    for class_id, count in enumerate(counts):
        candidates = class_indices[class_id]
        if count and len(candidates):
            selected.extend(rng.choice(candidates, size=min(count, len(candidates)), replace=False).tolist())

    if len(selected) < subset_size:
        remaining = np.setdiff1d(train_pool, np.asarray(selected, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)
        selected.extend(remaining[: subset_size - len(selected)].tolist())

    result = np.asarray(selected[:subset_size], dtype=np.int64)
    rng.shuffle(result)
    return result
