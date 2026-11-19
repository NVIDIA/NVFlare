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

from nvflare.utils.data.partition import dirichlet_label_partition


def download_cifar10(data_root: str) -> None:
    """Download both CIFAR-10 splits used by training and server evaluation."""
    datasets.CIFAR10(root=data_root, train=True, download=True)
    datasets.CIFAR10(root=data_root, train=False, download=True)


def _partition(labels: np.ndarray, num_sites: int, alpha: float, seed: int) -> list[list[int]]:
    result = dirichlet_label_partition(
        labels=np.asarray(labels).reshape(-1), num_sites=num_sites, alpha=alpha, seed=seed
    )
    return [result.site_indices[site].tolist() for site in range(num_sites)]


def split_and_save(split_dir_prefix: str, num_sites: int, alpha: float, seed: int = 0) -> str:
    """Create deterministic Dirichlet client shards and return their directory."""
    data_root = "/tmp/cifar10"
    download_cifar10(data_root)
    labels = np.asarray(datasets.CIFAR10(root=data_root, train=True, download=False).targets)
    result = dirichlet_label_partition(labels=labels, num_sites=num_sites, alpha=alpha, seed=seed)
    split_dir = Path(f"{split_dir_prefix}_{num_sites}sites_alpha{alpha:.2f}_seed{seed}")
    split_dir.mkdir(parents=True, exist_ok=True)

    class_summary = {}
    for site in range(num_sites):
        np.save(split_dir / f"site-{site + 1}.npy", result.site_indices[site])
        class_summary[str(site + 1)] = {str(label): count for label, count in result.class_counts[site].items()}

    (split_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_sites": num_sites,
                "alpha": alpha,
                "seed": seed,
                "partitioner": "nvflare.utils.data.partition.dirichlet_label_partition",
                "class_counts": class_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(split_dir)
