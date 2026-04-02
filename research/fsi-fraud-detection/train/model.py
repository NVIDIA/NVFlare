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

import random

import numpy as np
import torch
import torch.nn as nn


class SimpleNetwork(nn.Module):
    def __init__(self, input_size=5, num_classes=2, seed=42, width_factor=1, dropout_p=0.2):
        super(SimpleNetwork, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.width_factor = width_factor
        self.dropout_p = dropout_p

        # Set seed for deterministic initialization
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            # Additional settings for deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Pyramid architecture: monotonically decreasing hidden sizes scaled by width_factor.
        # input → 64 → LN → ReLU → drop → 32 → LN → ReLU → drop → 16 → LN → ReLU → drop → 2
        # LayerNorm is used (no batch stats, FL-friendly) and accepts (B, C) directly — no reshape.
        h1 = max(1, int(64 * width_factor))
        h2 = max(1, int(32 * width_factor))
        h3 = max(1, int(16 * width_factor))
        self.fc1 = nn.Linear(input_size, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.ln3 = nn.LayerNorm(h3)
        self.fc4 = nn.Linear(h3, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.dropout(self.relu(self.ln1(self.fc1(x))))
        x = self.dropout(self.relu(self.ln2(self.fc2(x))))
        x = self.dropout(self.relu(self.ln3(self.fc3(x))))
        x = self.fc4(x)
        return x
