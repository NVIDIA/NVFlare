# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .et_task_processor import ETTaskProcessor


class ETCIFAR10TaskProcessor(ETTaskProcessor):
    def get_dataset(self, data_path: str) -> Dataset:
        transform = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
