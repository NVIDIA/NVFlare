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

"""Small synthetic PyTorch model whose parameters enter tensor streaming."""

import torch
from torch import nn


class SyntheticModel(nn.Module):
    def __init__(self, tensor_elements: int = 1048576):
        super().__init__()
        self.weight_1 = nn.Parameter(torch.zeros(tensor_elements, dtype=torch.float32))
        self.weight_2 = nn.Parameter(torch.zeros(tensor_elements, dtype=torch.float32))
