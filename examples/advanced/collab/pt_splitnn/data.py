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

"""Prepared CIFAR-10 data helpers for the Collab SplitNN example."""

import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

MANIFEST_FILE = "splitnn_manifest.json"
INTERSECTION_FILE = "splitnn_intersection.npy"

_NORMALIZE = transforms.Normalize(
    mean=[value / 255.0 for value in (125.3, 123.0, 113.9)],
    std=[value / 255.0 for value in (63.0, 62.1, 66.7)],
)
_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)
_VALID_TRANSFORM = transforms.Compose([transforms.ToTensor(), _NORMALIZE])


def load_manifest(data_root: str) -> dict:
    manifest_path = Path(data_root) / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared SplitNN manifest not found at '{manifest_path}'. Run prepare_data.py before job.py."
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_prepared_data(data_root: str) -> dict:
    root = Path(data_root)
    manifest = load_manifest(str(root))
    required = {"dataset", "overlap", "seed", "train_size", "test_size"}
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Prepared SplitNN manifest is missing fields: {missing}")
    if manifest["dataset"] != "CIFAR10":
        raise ValueError(f"Expected a CIFAR10 manifest but found {manifest['dataset']!r}")

    intersection_path = root / INTERSECTION_FILE
    if not intersection_path.is_file():
        raise FileNotFoundError(f"Prepared intersection indices not found at '{intersection_path}'")
    intersection = np.load(intersection_path, allow_pickle=False)
    if intersection.ndim != 1 or len(intersection) != manifest["overlap"]:
        raise ValueError(
            f"Expected {manifest['overlap']} one-dimensional intersection indices, got shape {intersection.shape}"
        )
    if len(intersection) == 0 or intersection.min() < 0 or intersection.max() >= manifest["train_size"]:
        raise ValueError("Prepared intersection contains indices outside the CIFAR-10 training set")
    return manifest


class Cifar10SplitDataset:
    """Expose either images or labels for the same prepared sample indices."""

    def __init__(self, data_root: str, role: str, train: bool):
        if role not in ("image", "label"):
            raise ValueError(f"role must be 'image' or 'label', got {role!r}")

        dataset = datasets.CIFAR10(root=data_root, train=train, download=False)
        self.role = role
        self.images = dataset.data
        self.labels = np.asarray(dataset.targets, dtype=np.int64)
        self.transform = _TRAIN_TRANSFORM if train else _VALID_TRANSFORM

        if train:
            intersection = np.load(Path(data_root) / INTERSECTION_FILE, allow_pickle=False)
            intersection = np.sort(intersection.astype(np.int64, copy=False))
            self.images = self.images[intersection]
            self.labels = self.labels[intersection]

    def __len__(self) -> int:
        return len(self.labels)

    def get_batch(self, batch_indices):
        indices = np.asarray(batch_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"batch indices must be one-dimensional, got shape {indices.shape}")
        if self.role == "label":
            return torch.as_tensor(self.labels[indices], dtype=torch.long)

        images = [self.transform(self.images[index]) for index in indices]
        return torch.stack(images, dim=0)
