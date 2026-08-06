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

"""Prepared CIFAR-10 data helpers for the synchronous Collab examples."""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MANIFEST_FILE = "manifest.json"
SPLITS_DIR = "splits"
DEFAULT_DATA_ROOT = "/tmp/nvflare/datasets/cifar10_sync"

_MEAN = tuple(value / 255.0 for value in (125.3, 123.0, 113.9))
_STD = tuple(value / 255.0 for value in (63.0, 62.1, 66.7))
_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(_MEAN, _STD),
    ]
)
_TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])


def split_path(data_root: str | Path, site_name: str) -> Path:
    return Path(data_root).expanduser().resolve() / SPLITS_DIR / f"{site_name}.npy"


def load_manifest(data_root: str | Path) -> dict:
    manifest_path = Path(data_root).expanduser().resolve() / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; run prepare_data.py first")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("dataset") != "CIFAR10" or manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported prepared-data manifest: {manifest_path}")
    return manifest


def validate_prepared_data(data_root: str | Path, num_clients: int) -> dict:
    manifest = load_manifest(data_root)
    if manifest.get("num_clients") != num_clients:
        raise ValueError(
            f"Prepared data contains {manifest.get('num_clients')} clients, but --num-clients={num_clients}; "
            "run prepare_data.py with matching arguments"
        )
    missing = [split_path(data_root, f"site-{index}") for index in range(1, num_clients + 1)]
    missing = [path for path in missing if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prepared client split: {missing[0]}")
    return manifest


def make_train_loader(
    data_root: str | Path,
    site_name: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    indices_file = split_path(data_root, site_name)
    if not indices_file.is_file():
        raise FileNotFoundError(f"Missing prepared split for {site_name}: {indices_file}")
    dataset = datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=_TRAIN_TRANSFORM)
    indices = np.load(indices_file)
    if indices.size == 0:
        raise ValueError(f"Prepared split for {site_name} contains no examples: {indices_file}")
    indices = indices.tolist()
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def make_test_loader(data_root: str | Path, batch_size: int, num_workers: int) -> DataLoader:
    dataset = datasets.CIFAR10(root=str(data_root), train=False, download=False, transform=_TEST_TRANSFORM)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
