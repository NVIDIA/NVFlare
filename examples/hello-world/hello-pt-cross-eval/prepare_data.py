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

"""
Prepare CIFAR-10 data to avoid race conditions when multiple clients download simultaneously.

Run this before executing job.py to pre-download the dataset.
"""

import torchvision

DATASET_PATH = "/tmp/nvflare/data/cifar10"


def main():
    print(f"Downloading CIFAR-10 dataset to {DATASET_PATH}...")

    # Download training set
    print("Downloading training set...")
    torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True)

    # Download test set
    print("Downloading test set...")
    torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True)

    print("\nDataset preparation complete!")
    print(f"Dataset location: {DATASET_PATH}")


if __name__ == "__main__":
    main()
