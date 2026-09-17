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

import os
import warnings

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from data.cifar10_dataset import CIFAR10_Idx
from torchvision import transforms

warnings.filterwarnings("ignore", message=r".*align.*")

CIFAR10_ROOT = "/tmp/cifar10"


def load_cifar10_data():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*align.*")
        train_dataset = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=True)
    train_label = np.array(train_dataset.targets)
    return train_label


def get_site_class_summary(train_label, site_idx):
    class_sum = {}
    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_label[data_idx], return_counts=True)
        class_sum[site] = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
    return class_sum


def create_datasets(site_name, train_idx_root, central=False):
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
    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    if not central:
        site_idx_file_name = os.path.join(train_idx_root, site_name + ".npy")
        if os.path.exists(site_idx_file_name):
            site_idx = np.load(site_idx_file_name).tolist()
        else:
            raise FileNotFoundError(f"No subset index found: {site_idx_file_name}")
    else:
        site_idx = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*align.*")
        train_dataset = CIFAR10_Idx(
            root=CIFAR10_ROOT,
            data_idx=site_idx,
            train=True,
            download=True,
            transform=transform_train,
        )
        valid_dataset = torchvision.datasets.CIFAR10(
            root=CIFAR10_ROOT,
            train=False,
            download=True,
            transform=transform_valid,
        )

    return train_dataset, valid_dataset


def create_data_loaders(train_dataset, valid_dataset, batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, valid_loader
