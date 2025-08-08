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

import torch.nn as nn

from nvflare.edge.models.model import DeviceModel


class XorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.sigmoid_1 = nn.Sigmoid()
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid_1(x)
        x = self.linear2(x)
        return x


class TrainingNet(DeviceModel):
    def __init__(self):
        DeviceModel.__init__(self, XorNet())
