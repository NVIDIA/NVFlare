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

"""Deterministic, framework-agnostic data partitioning for federated learning.

The functions in this module only depend on NumPy. They take the labels (or the number of samples) of a dataset and
return, for each site, the indices of the samples that belong to that site. Downloading the dataset, normalizing it
and building data loaders stay with the caller.

All functions are seeded through numpy.random.default_rng(seed) and never touch the global NumPy random state, so the
same arguments always produce the same partition.

Typical use in a one-time prepare_data.py script::

    labels = np.array(train_dataset.targets)
    result = dirichlet_label_partition(labels, num_sites=8, alpha=0.5, seed=0)
    result.save("/tmp/splits/cifar10_alpha0.5.json")

and in a client script::

    result = PartitionResult.load("/tmp/splits/cifar10_alpha0.5.json")
    indices = result.site_indices[site_id]
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

_DEFAULT_MAX_ATTEMPTS = 1000


@dataclass
class PartitionResult:
    """Outcome of a partitioning call.

    Attributes:
        site_indices: maps site index (0 to num_sites - 1) to a 1-D integer array of dataset indices.
        class_counts: maps site index to {class label: number of samples}. Empty when no labels were given.
    """

    site_indices: Dict[int, np.ndarray]
    class_counts: Dict[int, Dict[int, int]] = field(default_factory=dict)

    @property
    def num_sites(self) -> int:
        return len(self.site_indices)

    def to_dict(self) -> dict:
        """Returns a JSON-serializable representation."""
        return {
            "num_sites": self.num_sites,
            "site_indices": {str(site): idx.tolist() for site, idx in self.site_indices.items()},
            "class_counts": {
                str(site): {str(label): int(count) for label, count in counts.items()}
                for site, counts in self.class_counts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PartitionResult":
        """Rebuilds a result from the output of to_dict."""
        site_indices = {int(site): np.asarray(idx, dtype=np.int64) for site, idx in data["site_indices"].items()}
        class_counts = {
            int(site): {int(label): int(count) for label, count in counts.items()}
            for site, counts in data.get("class_counts", {}).items()
        }
        return cls(site_indices=site_indices, class_counts=class_counts)

    def save(self, path: str) -> None:
        """Writes the partition to a JSON file, creating parent directories if needed."""
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "PartitionResult":
        """Reads a partition written by save."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _check_common(num_sites: int, alpha: Optional[float], min_size: int, max_attempts: int, num_samples: int) -> None:
    if not isinstance(num_sites, (int, np.integer)) or num_sites < 1:
        raise ValueError(f"num_sites must be a positive integer but got {num_sites!r}")
    if not isinstance(num_samples, (int, np.integer)) or num_samples < 0:
        raise ValueError(f"num_samples must be a non-negative integer but got {num_samples!r}")
    if alpha is not None and (not isinstance(alpha, (int, float, np.number)) or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError(f"alpha must be a finite number larger than 0 but got {alpha!r}")
    if not isinstance(min_size, (int, np.integer)) or min_size < 0:
        raise ValueError(f"min_size must be a non-negative integer but got {min_size!r}")
    if min_size * num_sites > num_samples:
        raise ValueError(f"cannot give {num_sites} sites at least {min_size} samples each from {num_samples} samples")
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1:
        raise ValueError(f"max_attempts must be a positive integer but got {max_attempts!r}")


def _as_labels(labels: Sequence, num_samples: Optional[int] = None) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D but got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be integer class ids but got dtype {labels.dtype}")
    if num_samples is not None and labels.shape[0] != num_samples:
        raise ValueError(f"labels has {labels.shape[0]} entries but num_samples is {num_samples}")
    return labels


def _class_counts(labels: Optional[np.ndarray], site_indices: Dict[int, np.ndarray]) -> Dict[int, Dict[int, int]]:
    if labels is None:
        return {}
    counts = {}
    for site, idx in site_indices.items():
        unique, unique_counts = np.unique(labels[idx], return_counts=True)
        counts[site] = {int(label): int(count) for label, count in zip(unique, unique_counts)}
    return counts


def _build_result(
    parts: Sequence[Sequence[int]], rng: np.random.Generator, labels: Optional[np.ndarray]
) -> PartitionResult:
    """Shuffles the indices of each site and wraps them in a PartitionResult."""
    site_indices = {}
    for site, part in enumerate(parts):
        part = np.array(part, dtype=np.int64)
        rng.shuffle(part)
        site_indices[site] = part
    return PartitionResult(site_indices=site_indices, class_counts=_class_counts(labels, site_indices))


def dirichlet_label_partition(
    labels: Sequence,
    num_sites: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> PartitionResult:
    """Splits a labeled dataset into label-skewed shards using per-class Dirichlet sampling.

    For every class, the samples are shuffled and divided among the sites according to proportions drawn from
    Dirichlet(alpha). Sites that already hold at least len(labels) / num_sites samples receive no further samples,
    which keeps the shard sizes balanced. The whole draw is repeated until every site has at least min_size samples.
    Small alpha values give strongly skewed label distributions, large values approach an IID split.

    The sampling strategy is adopted from FedMA (https://github.com/IBM/FedMA).

    Args:
        labels: 1-D sequence with one integer class label per sample.
        num_sites: number of sites to partition the data into.
        alpha: concentration parameter of the Dirichlet distribution, must be larger than 0.
        seed: seed for numpy.random.default_rng.
        min_size: minimum number of samples every site must receive.
        max_attempts: number of times the draw is repeated to reach min_size before giving up.

    Returns:
        A PartitionResult with the sample indices and the per-class counts of every site.

    Raises:
        ValueError: if an argument is invalid or min_size cannot be reached within max_attempts.
    """
    labels = _as_labels(labels)
    num_samples = labels.shape[0]
    _check_common(num_sites, alpha, min_size, max_attempts, num_samples)

    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == c) for c in np.unique(labels)]

    for _ in range(max_attempts):
        parts: List[List[int]] = [[] for _ in range(num_sites)]
        for idx_k in class_indices:
            idx_k = rng.permutation(idx_k)
            proportions = rng.dirichlet(np.repeat(alpha, num_sites))
            # sites that are already full receive nothing from this class
            has_room = np.array([len(part) < num_samples / num_sites for part in parts])
            masked = proportions * has_room
            if masked.sum() <= 0:
                masked = proportions
            masked = masked / masked.sum()
            cut_points = (np.cumsum(masked) * len(idx_k)).astype(int)[:-1]
            for part, chunk in zip(parts, np.split(idx_k, cut_points)):
                part.extend(chunk.tolist())
        if min(len(part) for part in parts) >= min_size:
            return _build_result(parts, rng, labels)

    raise ValueError(
        f"could not give every site at least {min_size} samples in {max_attempts} attempts; "
        "increase alpha, lower min_size or use fewer sites"
    )


def iid_partition(
    num_samples: int,
    num_sites: int,
    seed: int,
    labels: Optional[Sequence] = None,
) -> PartitionResult:
    """Splits a dataset into shards of (almost) equal size.

    Without labels the samples are shuffled and divided evenly. With labels the split is stratified: every class is
    shuffled and divided evenly, so each site sees every class in nearly the same proportion as the full dataset.
    The number of samples of a class differs by at most one between any two sites.

    Args:
        num_samples: number of samples in the dataset.
        num_sites: number of sites to partition the data into.
        seed: seed for numpy.random.default_rng.
        labels: optional 1-D sequence with one integer class label per sample, used for a stratified split and for the class
            counts in the result.

    Returns:
        A PartitionResult. class_counts is empty when no labels are given.

    Raises:
        ValueError: if an argument is invalid or labels does not have num_samples entries.
    """
    _check_common(num_sites, None, 0, 1, num_samples)
    if num_samples < num_sites:
        raise ValueError(f"cannot split {num_samples} samples across {num_sites} sites")
    rng = np.random.default_rng(seed)

    if labels is None:
        return _build_result(np.array_split(rng.permutation(num_samples), num_sites), rng, None)

    labels = _as_labels(labels, num_samples)
    parts: List[List[int]] = [[] for _ in range(num_sites)]
    offset = 0
    for c in np.unique(labels):
        idx_c = rng.permutation(np.flatnonzero(labels == c))
        # rotate which sites get the larger chunks so the remainders do not pile up on the first sites
        for i, chunk in enumerate(np.array_split(idx_c, num_sites)):
            parts[(i + offset) % num_sites].extend(chunk.tolist())
        offset += len(idx_c) % num_sites
    return _build_result(parts, rng, labels)


def quantity_skew_partition(
    num_samples: int,
    num_sites: int,
    seed: int,
    alpha: float = 0.5,
    min_size: int = 1,
    labels: Optional[Sequence] = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> PartitionResult:
    """Splits a dataset into shards of different sizes, with no control over the label distribution.

    The shard sizes follow proportions drawn from Dirichlet(alpha) and the samples are assigned at random, so every
    site sees roughly the overall label distribution but holds a different amount of data. Small alpha values give
    very unequal sizes.

    Args:
        num_samples: number of samples in the dataset.
        num_sites: number of sites to partition the data into.
        seed: seed for numpy.random.default_rng.
        alpha: concentration parameter of the Dirichlet distribution, must be larger than 0.
        min_size: minimum number of samples every site must receive.
        labels: optional 1-D sequence with one integer class label per sample, used for the class counts in the result.
        max_attempts: number of times the draw is repeated to reach min_size before giving up.

    Returns:
        A PartitionResult. class_counts is empty when no labels are given.

    Raises:
        ValueError: if an argument is invalid or min_size cannot be reached within max_attempts.
    """
    _check_common(num_sites, alpha, min_size, max_attempts, num_samples)
    if labels is not None:
        labels = _as_labels(labels, num_samples)

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        proportions = rng.dirichlet(np.repeat(alpha, num_sites))
        cut_points = (np.cumsum(proportions) * num_samples).astype(int)[:-1]
        sizes = np.diff(np.concatenate(([0], cut_points, [num_samples])))
        if sizes.min() >= min_size:
            return _build_result(np.split(rng.permutation(num_samples), cut_points), rng, labels)

    raise ValueError(
        f"could not give every site at least {min_size} samples in {max_attempts} attempts; "
        "increase alpha, lower min_size or use fewer sites"
    )
