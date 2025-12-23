# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# This Dirichlet sampling strategy for creating a heterogeneous partition is adopted
# from FedMA (https://github.com/IBM/FedMA).

# MIT License

# Copyright (c) 2020 International Business Machines

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision
from torchvision import transforms
import os
from data.cifar10_dataset import CIFAR10_Idx

CIFAR10_ROOT = "/tmp/cifar10"  # will be used for all CIFAR-10 experiments


def load_cifar10_data():
    # load data
    train_dataset = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=True)

    # only training label is needed for doing split
    train_label = np.array(train_dataset.targets)
    return train_label


def get_site_class_summary(train_label, site_idx):
    class_sum = {}

    for site, data_idx in site_idx.items():
        unq, unq_cnt = np.unique(train_label[data_idx], return_counts=True)
        tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        class_sum[site] = tmp
    return class_sum


def create_datasets(site_name, train_idx_root, central=False):
    """To be called only after cifar10_data_split.split_and_save() downloaded the data and computed splits"""

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
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
        # Set datalist, here the path and filename are hard-coded, can also be fed as an argument
        site_idx_file_name = os.path.join(train_idx_root, site_name + ".npy")
        if os.path.exists(site_idx_file_name):
            print(f"Loading subset index for client {site_name} from {site_idx_file_name}")
            site_idx = np.load(site_idx_file_name).tolist()
        else:
            raise FileNotFoundError(f"No subset index found! File {site_idx_file_name} does not exist!")
        print(f"Client {site_name} subset size: {len(site_idx)}")
    else:
        site_idx = None  # use whole training dataset if central=True

    train_dataset = CIFAR10_Idx(
        root=CIFAR10_ROOT,
        data_idx=site_idx,
        train=True,
        download=False,
        transform=transform_train,
    )

    valid_dataset = torchvision.datasets.CIFAR10(
        root=CIFAR10_ROOT,
        train=False,
        download=False,
        transform=transform_valid,
    )

    return train_dataset, valid_dataset


def create_data_loaders(train_dataset, valid_dataset, batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
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


def main():
    """
    Load CIFAR data if run as a main script
    """
    load_cifar10_data()


if __name__ == "__main__":
    main()
