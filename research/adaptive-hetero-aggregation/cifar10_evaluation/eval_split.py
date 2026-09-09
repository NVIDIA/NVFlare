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

"""Create deterministic site-specific CIFAR-10 evaluation partitions.

Training indices come from NVIDIA FLARE's standard Dirichlet splitter. Test
examples are allocated per class in proportion to each site's training-class
mass, producing local evaluation distributions that reflect the training
heterogeneity without removing examples from the training sets.
"""

import json
import os

import numpy as np
from data.cifar10_data_utils import CIFAR10_ROOT, load_cifar10_data
from torchvision import datasets

NUM_CLASSES = 10


def _integer_allocation(total: int, proportions: np.ndarray) -> np.ndarray:
    proportions = np.asarray(proportions, dtype=np.float64)
    if proportions.ndim != 1 or proportions.size == 0:
        raise ValueError("proportions must be a non-empty one-dimensional array")
    if np.any(proportions < 0.0) or not np.all(np.isfinite(proportions)):
        raise ValueError("proportions must be finite and non-negative")
    if float(proportions.sum()) == 0.0:
        proportions = np.ones_like(proportions)
    proportions = proportions / proportions.sum()
    raw = proportions * total
    allocated = np.floor(raw).astype(np.int64)
    remainder = total - int(allocated.sum())
    if remainder:
        order = np.argsort(-(raw - allocated), kind="stable")
        allocated[order[:remainder]] += 1
    return allocated


def create_eval_splits(train_idx_root: str, output_root: str, n_clients: int, seed: int) -> str:
    os.makedirs(output_root, exist_ok=True)
    train_labels = load_cifar10_data()
    test_dataset = datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=True)
    test_labels = np.asarray(test_dataset.targets, dtype=np.int64)

    site_class_counts = np.zeros((n_clients, NUM_CLASSES), dtype=np.int64)
    for site_index in range(n_clients):
        train_path = os.path.join(train_idx_root, f"site-{site_index + 1}.npy")
        if not os.path.isfile(train_path):
            raise ValueError(f"missing training split for site-{site_index + 1}: {train_path}")
        train_indices = np.load(train_path)
        site_class_counts[site_index] = np.bincount(train_labels[train_indices], minlength=NUM_CLASSES)

    rng = np.random.default_rng(seed + 100_000)
    site_eval_indices = [[] for _ in range(n_clients)]
    for class_index in range(NUM_CLASSES):
        class_indices = np.flatnonzero(test_labels == class_index)
        rng.shuffle(class_indices)
        allocations = _integer_allocation(len(class_indices), site_class_counts[:, class_index])
        start = 0
        for site_index, count in enumerate(allocations):
            stop = start + int(count)
            site_eval_indices[site_index].extend(class_indices[start:stop].tolist())
            start = stop
        if start != len(class_indices):
            raise RuntimeError("CIFAR-10 evaluation split did not allocate every test example")

    summary = {
        "seed": seed,
        "n_clients": n_clients,
        "sites": {},
    }
    for site_index, indices in enumerate(site_eval_indices):
        if not indices:
            raise ValueError(f"site-{site_index + 1} received an empty evaluation split")
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        path = os.path.join(output_root, f"site-{site_index + 1}.npy")
        np.save(path, indices)
        counts = np.bincount(test_labels[indices], minlength=NUM_CLASSES)
        summary["sites"][f"site-{site_index + 1}"] = {
            "examples": int(indices.size),
            "class_counts": counts.astype(int).tolist(),
        }

    with open(os.path.join(output_root, "summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
    return output_root
