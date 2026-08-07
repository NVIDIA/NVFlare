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

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = "/tmp/nvflare/data"
_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def make_data_loader(train: bool, batch_size: int) -> DataLoader:
    """Load CIFAR-10 the same way as the hello-pt example."""

    dataset = datasets.CIFAR10(root=DATA_ROOT, train=train, download=True, transform=_TRANSFORM)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
