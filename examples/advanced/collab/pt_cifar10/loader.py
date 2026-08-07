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

"""CIFAR-10 loading shared by the synchronous Collab examples."""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_ROOT = "/tmp/cifar10"
SPLIT_ROOT = "/tmp/cifar10_splits/pt_cifar10_2sites_alpha0.50_seed0"
_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def make_data_loader(train: bool, batch_size: int, site_name: str | None = None) -> DataLoader:
    """Load the shared test set or one prepared client training split."""

    dataset = datasets.CIFAR10(root=DATA_ROOT, train=train, download=False, transform=_TRANSFORM)
    if train:
        if not site_name:
            raise ValueError("site_name is required for a client training loader")

        # The standard splitter names each client index file after its NVFlare site.
        split_file = Path(SPLIT_ROOT) / f"{site_name}.npy"
        indices = np.load(split_file, allow_pickle=False).tolist()
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
