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

"""Create deterministic CIFAR-10 train/validation and final-test partitions.

NVIDIA FLARE's standard Dirichlet splitter first assigns CIFAR-10 training
examples to sites. This module then:

1. carves a deterministic held-out validation subset from each site's assigned
   *training* examples; and
2. creates site-specific partitions of the untouched CIFAR-10 test set for the
   common post-training evaluator.

The test set is never used to produce adaptive or FedCE aggregation signals.
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


def create_train_validation_splits(
    assignment_root: str,
    train_output_root: str,
    validation_output_root: str,
    n_clients: int,
    seed: int,
    validation_fraction: float = 0.10,
) -> tuple[str, str]:
    """Split each site's Dirichlet training assignment into train and validation.

    The split is site-local and deterministic. Validation examples remain from
    CIFAR-10's training set; the official CIFAR-10 test set stays untouched for
    final reporting.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    os.makedirs(train_output_root, exist_ok=True)
    os.makedirs(validation_output_root, exist_ok=True)

    train_labels = load_cifar10_data()
    summary = {
        "seed": seed,
        "n_clients": n_clients,
        "validation_fraction": validation_fraction,
        "sites": {},
    }

    for site_index in range(n_clients):
        site_name = f"site-{site_index + 1}"
        assignment_path = os.path.join(assignment_root, f"{site_name}.npy")
        if not os.path.isfile(assignment_path):
            raise ValueError(f"missing Dirichlet assignment for {site_name}: {assignment_path}")
        assigned = np.load(assignment_path).astype(np.int64)
        if assigned.ndim != 1 or assigned.size < 2:
            raise ValueError(f"{site_name} needs at least two assigned training examples")

        rng = np.random.default_rng(seed + 10_000 + site_index)
        shuffled = assigned.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(shuffled.size * validation_fraction)))
        validation_count = min(validation_count, shuffled.size - 1)
        validation_indices = shuffled[:validation_count]
        train_indices = shuffled[validation_count:]

        np.save(os.path.join(train_output_root, f"{site_name}.npy"), train_indices)
        np.save(os.path.join(validation_output_root, f"{site_name}.npy"), validation_indices)

        train_counts = np.bincount(train_labels[train_indices], minlength=NUM_CLASSES)
        validation_counts = np.bincount(train_labels[validation_indices], minlength=NUM_CLASSES)
        summary["sites"][site_name] = {
            "assigned_examples": int(assigned.size),
            "train_examples": int(train_indices.size),
            "validation_examples": int(validation_indices.size),
            "train_class_counts": train_counts.astype(int).tolist(),
            "validation_class_counts": validation_counts.astype(int).tolist(),
        }

    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    with open(os.path.join(train_output_root, "summary.json"), "w") as summary_file:
        summary_file.write(summary_text + "\n")
    with open(os.path.join(validation_output_root, "summary.json"), "w") as summary_file:
        summary_file.write(summary_text + "\n")
    return train_output_root, validation_output_root


def create_eval_splits(assignment_root: str, output_root: str, n_clients: int, seed: int) -> str:
    """Partition the untouched CIFAR-10 test set for client-level final metrics."""

    os.makedirs(output_root, exist_ok=True)
    train_labels = load_cifar10_data()
    test_dataset = datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=True)
    test_labels = np.asarray(test_dataset.targets, dtype=np.int64)

    site_class_counts = np.zeros((n_clients, NUM_CLASSES), dtype=np.int64)
    for site_index in range(n_clients):
        assignment_path = os.path.join(assignment_root, f"site-{site_index + 1}.npy")
        if not os.path.isfile(assignment_path):
            raise ValueError(f"missing Dirichlet assignment for site-{site_index + 1}: {assignment_path}")
        assignment_indices = np.load(assignment_path)
        site_class_counts[site_index] = np.bincount(train_labels[assignment_indices], minlength=NUM_CLASSES)

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
            raise RuntimeError("CIFAR-10 test split did not allocate every test example")

    summary = {
        "seed": seed,
        "n_clients": n_clients,
        "source": "untouched_cifar10_test_set",
        "sites": {},
    }
    for site_index, indices in enumerate(site_eval_indices):
        site_name = f"site-{site_index + 1}"
        if not indices:
            raise ValueError(f"{site_name} received an empty final evaluation split")
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        np.save(os.path.join(output_root, f"{site_name}.npy"), indices)
        counts = np.bincount(test_labels[indices], minlength=NUM_CLASSES)
        summary["sites"][site_name] = {
            "examples": int(indices.size),
            "class_counts": counts.astype(int).tolist(),
        }

    with open(os.path.join(output_root, "summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    return output_root
