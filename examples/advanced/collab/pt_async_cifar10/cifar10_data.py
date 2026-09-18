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

"""Download and partition CIFAR-10 for the standalone asynchronous Collab example."""

import json
from pathlib import Path

import numpy as np
from torchvision import datasets


def download_cifar10(data_root: str) -> None:
    """Download both CIFAR-10 splits used by training and server evaluation."""
    datasets.CIFAR10(root=data_root, train=True, download=True)
    datasets.CIFAR10(root=data_root, train=False, download=True)


def _partition(labels: np.ndarray, num_sites: int, alpha: float, seed: int) -> list[list[int]]:
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero")
    random = np.random.default_rng(seed)
    minimum_size = 0
    while minimum_size < 10:
        site_indices = [[] for _ in range(num_sites)]
        for label in np.unique(labels):
            label_indices = np.flatnonzero(labels == label)
            random.shuffle(label_indices)
            proportions = random.dirichlet(np.full(num_sites, alpha))
            proportions *= np.array([len(indices) < len(labels) / num_sites for indices in site_indices])
            proportions /= proportions.sum()
            boundaries = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
            site_indices = [
                indices + partition.tolist()
                for indices, partition in zip(site_indices, np.split(label_indices, boundaries))
            ]
        minimum_size = min(map(len, site_indices))
    for indices in site_indices:
        random.shuffle(indices)
    return site_indices


def split_and_save(split_dir_prefix: str, num_sites: int, alpha: float, seed: int = 0) -> str:
    """Create deterministic Dirichlet client shards and return their directory."""
    data_root = "/tmp/cifar10"
    download_cifar10(data_root)
    labels = np.asarray(datasets.CIFAR10(root=data_root, train=True, download=False).targets)
    site_indices = _partition(labels, num_sites, alpha, seed)
    split_dir = Path(f"{split_dir_prefix}_{num_sites}sites_alpha{alpha:.2f}_seed{seed}")
    split_dir.mkdir(parents=True, exist_ok=True)

    class_summary = {}
    for site, indices in enumerate(site_indices, start=1):
        np.save(split_dir / f"site-{site}.npy", np.asarray(indices))
        values, counts = np.unique(labels[indices], return_counts=True)
        class_summary[str(site)] = {str(int(value)): int(count) for value, count in zip(values, counts)}

    (split_dir / "summary.json").write_text(
        json.dumps(
            {"num_sites": num_sites, "alpha": alpha, "seed": seed, "class_counts": class_summary},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(split_dir)
