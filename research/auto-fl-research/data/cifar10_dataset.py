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

import numpy as np
from torchvision.datasets import CIFAR10


class CIFAR10_Idx(CIFAR10):
    """CIFAR-10 dataset wrapper that exposes a subset by index."""

    def __init__(self, *args, data_idx=None, **kwargs):
        super().__init__(*args, **kwargs)
        if data_idx is not None:
            data_idx = np.asarray(data_idx)
            self.data = self.data[data_idx]
            targets = np.asarray(self.targets)
            self.targets = targets[data_idx].tolist()
