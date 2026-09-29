# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
pytorch model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_SEED = 202610


class SimpleNetwork(nn.Module):
    def __init__(self, seed=None):
        super(SimpleNetwork, self).__init__()
        self.seed = seed
        if seed is None:
            self._initialize_layers()
        else:
            # Only seed the CPU generator used by these layers. ``fork_rng``
            # restores it afterward without resetting CUDA/MPS/XPU generators.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self._initialize_layers()

    def _initialize_layers(self):
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model():
    """Create the recipe model with an explicitly serialized initialization seed."""
    return SimpleNetwork(seed=MODEL_SEED)
