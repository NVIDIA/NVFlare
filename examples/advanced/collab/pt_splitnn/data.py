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

"""Role-specific CIFAR-10 views aligned by existing SplitNN PSI artifacts."""

import numpy as np
import torch
from torchvision import datasets, transforms

_NORMALIZE = transforms.Normalize(
    mean=[value / 255.0 for value in (125.3, 123.0, 113.9)],
    std=[value / 255.0 for value in (63.0, 62.1, 66.7)],
)
_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)
_VALID_TRANSFORM = transforms.Compose([transforms.ToTensor(), _NORMALIZE])


class Cifar10SplitDataset:
    """Expose either images or labels for the same prepared sample indices."""

    def __init__(self, dataset_root: str, intersection_file: str, role: str, train: bool):
        if role not in ("image", "label"):
            raise ValueError(f"role must be 'image' or 'label', got {role!r}")

        dataset = datasets.CIFAR10(root=dataset_root, train=train, download=False)
        indices = np.arange(len(dataset))
        if train:
            indices = np.loadtxt(intersection_file)
            indices = np.sort(indices).astype(np.int64)

        self.role = role
        self.size = len(indices)
        if role == "image":
            self.images = dataset.data[indices]
            self.labels = None
            self.transform = _TRAIN_TRANSFORM if train else _VALID_TRANSFORM
        else:
            self.images = None
            self.labels = np.asarray(dataset.targets, dtype=np.int64)[indices]
            self.transform = None

    def __len__(self) -> int:
        return self.size

    def get_batch(self, batch_indices):
        indices = np.asarray(batch_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"batch indices must be one-dimensional, got shape {indices.shape}")
        if self.role == "label":
            return torch.as_tensor(self.labels[indices], dtype=torch.long)

        images = [self.transform(self.images[index]) for index in indices]
        return torch.stack(images, dim=0)
