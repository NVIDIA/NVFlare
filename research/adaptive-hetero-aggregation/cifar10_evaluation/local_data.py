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

"""Load matched local CIFAR-10 train/validation data from training-set indices."""

import os

import numpy as np
from data.cifar10_data_utils import CIFAR10_ROOT
from data.cifar10_dataset import CIFAR10_Idx
from torchvision import transforms


def _indices(root: str, site_name: str, kind: str) -> list[int]:
    path = os.path.join(root, f"{site_name}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {kind} indices for {site_name}: {path}")
    values = np.load(path).astype(np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{kind} indices for {site_name} must be a non-empty vector")
    return values.tolist()


def create_local_datasets(site_name: str, train_idx_root: str, validation_idx_root: str):
    """Return local train and held-out validation datasets from CIFAR-10 train."""

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )
    transform_validation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    train_dataset = CIFAR10_Idx(
        root=CIFAR10_ROOT,
        data_idx=_indices(train_idx_root, site_name, "training"),
        train=True,
        download=False,
        transform=transform_train,
    )
    validation_dataset = CIFAR10_Idx(
        root=CIFAR10_ROOT,
        data_idx=_indices(validation_idx_root, site_name, "validation"),
        train=True,
        download=False,
        transform=transform_validation,
    )
    return train_dataset, validation_dataset
