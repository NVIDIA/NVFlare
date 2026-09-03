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

"""Prepare synthetic quickstart data or download optional CIFAR-10 data."""

import argparse
import hashlib

import torch
from torch.utils.data import Dataset

IMAGE_SHAPE = (3, 32, 32)
NUM_CLASSES = 10
DATASET_PATH = "/tmp/nvflare/data"
DATASET_CHOICES = ("synthetic", "cifar10")
DEFAULT_DATASET = "synthetic"
_BASE_SEED = 202610
_VALID_SPLITS = ("train", "eval")


def stable_seed(site_name: str, purpose: str) -> int:
    """Derive a process-independent PyTorch seed for one site and purpose."""
    if not site_name:
        raise ValueError("site_name must be non-empty")
    if not purpose:
        raise ValueError("purpose must be non-empty")

    value = f"{_BASE_SEED}:{site_name}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63 - 1)


class SyntheticImageDataset(Dataset):
    """Small site-specific dataset whose labels are encoded by image regions.

    Each class has a bright patch at a distinct position. Low-amplitude noise,
    generated from the site and split seed, makes samples distinct while
    retaining a simple class-related signal for the example CNN.
    """

    def __init__(self, site_name: str, split: str, size: int):
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")

        self.site_name = site_name
        self.split = split
        self.sample_ids = tuple(f"{split}:{site_name}:{index}" for index in range(size))

        generator = torch.Generator().manual_seed(stable_seed(site_name, split))
        labels = torch.arange(size, dtype=torch.long) % NUM_CLASSES
        labels = labels[torch.randperm(size, generator=generator)]
        images = torch.rand((size, *IMAGE_SHAPE), generator=generator) * 0.08

        for index, label in enumerate(labels.tolist()):
            # Encode the class as a bright patch in one of ten positions. This
            # creates a real image-to-label relationship instead of random labels.
            row = 2 + (label // 5) * 16
            column = 1 + (label % 5) * 6
            images[index, :, row : row + 5, column : column + 5] = 1.0

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def download_cifar10(data_root: str = DATASET_PATH):
    """Download both CIFAR-10 splits once, before simulated clients start."""
    from torchvision.datasets import CIFAR10

    for train in (True, False):
        CIFAR10(root=data_root, train=train, download=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download CIFAR-10 for the optional Hello PyTorch CIFAR path.")
    parser.add_argument("--data_root", default=DATASET_PATH)
    args = parser.parse_args(argv)
    download_cifar10(args.data_root)
    print(f"CIFAR-10 is ready under {args.data_root}")


if __name__ == "__main__":
    main()
