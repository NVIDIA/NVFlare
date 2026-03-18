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

"""Large MLP model for B1 pass-through integration testing.

Designed to have parameters well above the ViaDownloaderDecomposer streaming
threshold (2 MB) so that tensors are sent via the download service rather
than inlined (native mode).  This ensures the B1 PASS_THROUGH path —
where the CJ forwards LazyDownloadRef placeholders and the subprocess
downloads directly from the FL server — is exercised.

Parameter count:
  fc1: Linear(1024, 1024) →  1,049,600 params ≈ 4 MB (float32)
  fc2: Linear(1024, 1024) →  1,049,600 params ≈ 4 MB (float32)
  fc3: Linear(1024,   10) →     10,250 params
  Total                    ≈ 2.1 M params ≈ 8 MB
"""

import torch.nn as nn
import torch.nn.functional as F


class LargeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
