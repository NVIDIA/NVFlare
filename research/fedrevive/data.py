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

"""CIFAR-10 preparation and transforms for the FedRevive experiments."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
DELAY_SCHEDULES = {
    "default": {
        "train": ((0.25, 1.0), (0.5, 1.3), (0.25, 1.6)),
        "download": ((1.0, 0.1),),
        "upload": ((0.5, 0.15), (0.5, 0.25)),
        "upload_half_width": 0.02,
    },
    # Appendix D.1 / Figure 3 changes only the client runtime model.  Keeping
    # the schedule as prepared-data metadata makes it impossible to
    # accidentally compare runs that use different arrival processes.
    "shifted": {
        "train": ((1.0, 0.1),),
        "download": ((1.0, 0.02),),
        "upload": ((0.5, 0.05), (0.5, 0.10)),
        "upload_half_width": 0.01,
    },
}


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def test_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def _select_mean(mean_distribution):
    draw = np.random.random()
    cumulative = 0.0
    total = sum(probability for probability, _ in mean_distribution)
    for probability, mean in mean_distribution:
        cumulative += probability / total
        if draw <= cumulative:
            return float(mean)
    return float(mean_distribution[-1][1])


def _make_runtime_profiles(num_clients: int, delay_schedule: str) -> list[dict[str, float]]:
    """Sample persistent per-client means for a paper delay schedule.

    These profiles are written to the manifest rather than inferred from real
    worker duration.  Consequently a small reusable process pool can replay an
    arbitrary logical arrival sequence.  Delay schedule is an experimental
    parameter, not a performance detail: it changes staleness and which updates
    are grouped by FedBuff, and can therefore materially change its curve.
    """

    try:
        schedule = DELAY_SCHEDULES[delay_schedule]
    except KeyError as error:
        raise ValueError(f"Unknown delay schedule: {delay_schedule}") from error

    profiles = []
    for _ in range(num_clients):
        profiles.append(
            {
                # Select all three means even when a distribution has one
                # member.  Consuming one RNG draw for each property keeps the
                # later client-selection and duration streams reproducible.
                "train_mean": _select_mean(schedule["train"]),
                "download_mean": _select_mean(schedule["download"]),
                "upload_mean": _select_mean(schedule["upload"]),
                "upload_half_width": schedule["upload_half_width"],
            }
        )
    return profiles


def _split_dirichlet(labels, num_clients: int, client_data_size: int, alpha: float):
    indices_by_class = {label: [] for label in range(10)}
    for index, label in enumerate(labels):
        indices_by_class[int(label)].append(index)
    for indices in indices_by_class.values():
        random.shuffle(indices)

    used = {label: 0 for label in indices_by_class}
    client_indices = []
    for _ in range(num_clients):
        proportions = np.random.dirichlet(np.repeat(alpha, 10))
        counts = np.round(proportions * client_data_size).astype(int)
        difference = client_data_size - int(counts.sum())
        if difference > 0:
            counts[random.choice(range(10))] += difference
        while difference < 0:
            largest = int(np.argmax(counts))
            counts[largest] -= 1
            difference += 1

        indices = []
        for label, count in enumerate(counts):
            if count <= 0:
                continue
            class_indices = indices_by_class[label]
            if used[label] + count > len(class_indices):
                random.shuffle(class_indices)
                used[label] = 0
            indices.extend(class_indices[used[label] : used[label] + count])
            used[label] += int(count)
        client_indices.append(indices)
    return client_indices


def prepare_cifar10(
    output_root: str | Path,
    download_root: str | Path,
    num_logical_clients: int = 1000,
    client_data_size: int = 350,
    alpha: float = 0.5,
    setup_seed: int = 10,
    split_seed: int = 42,
    delay_schedule: str = "default",
) -> Path:
    """Create the train split, logical shards, proxies, and runtime profiles."""

    if num_logical_clients < 1 or client_data_size < 1:
        raise ValueError("num_logical_clients and client_data_size must be >= 1")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    output_root = Path(output_root).expanduser().resolve()
    split_root = output_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    download_root = Path(download_root).expanduser().resolve()
    dataset = datasets.CIFAR10(root=download_root, train=True, download=True)

    generator = torch.Generator().manual_seed(split_seed)
    train_subset, validation_subset, kd_subset = random_split(dataset, [37500, 7500, 5000], generator=generator)
    train_indices = list(train_subset.indices)
    train_labels = [dataset.targets[index] for index in train_indices]

    random.seed(setup_seed)
    np.random.seed(setup_seed)
    # Persist the sampled identities with the data partition.  Reusing this
    # manifest across methods prevents each baseline from silently receiving a
    # different client-speed population or participant sequence.
    runtime_profiles = _make_runtime_profiles(num_logical_clients, delay_schedule)
    relative_splits = _split_dirichlet(train_labels, num_logical_clients, client_data_size, alpha)

    class_proportions = {}
    for logical_index, relative_indices in enumerate(relative_splits):
        logical_name = f"client-{logical_index}"
        full_indices = np.asarray([train_indices[index] for index in relative_indices], dtype=np.int64)
        np.save(split_root / f"{logical_name}.npy", full_indices)
        counts = np.bincount([dataset.targets[index] for index in full_indices], minlength=10).astype(float)
        class_proportions[logical_name] = (counts / counts.sum()).tolist()

    manifest = {
        "dataset": "cifar10",
        "num_logical_clients": num_logical_clients,
        "client_data_size": client_data_size,
        "dirichlet_alpha": alpha,
        "setup_seed": setup_seed,
        "split_seed": split_seed,
        "delay_schedule": delay_schedule,
        "train_size": len(train_indices),
        "validation_size": len(validation_subset),
        "kd_size": len(kd_subset),
        "class_proportions": class_proportions,
        "runtime_profiles": {f"client-{index}": profile for index, profile in enumerate(runtime_profiles)},
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return output_root


def load_manifest(prepared_data_root: str | Path) -> dict:
    path = Path(prepared_data_root).expanduser().resolve() / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared FedRevive manifest not found: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
