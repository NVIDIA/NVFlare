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

from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from nvflare.edge.simulation.et_task_processor import ETTaskProcessor


class Cifar10ETTaskProcessor(ETTaskProcessor):
    def __init__(self, data_path: str, training_config=None, subset_size: int = 100):
        """Initialize CIFAR10 task processor with subset capability.

        Args:
            data_path: Path to store CIFAR10 data
            training_config: Training configuration dict
            subset_size: Number of samples per device (default: 100 samples)
        """
        super().__init__(data_path, training_config)
        self.subset_size = subset_size

    def create_dataset(self, data_path: str) -> Dataset:
        transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)

        # Create a subset for edge device simulation
        indices = list(range(min(self.subset_size, len(full_dataset))))
        subset = Subset(full_dataset, indices)
        device_id_str = self.device.device_id if self.device else "unknown"
        print(
            f"Device {device_id_str}: Using subset of {len(subset)} samples from CIFAR10 (first {self.subset_size} samples)"
        )
        return subset
