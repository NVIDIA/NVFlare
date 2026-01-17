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
"""
PyTorch MLP model for protein subcellular location classification
"""

import torch.nn as nn


class ProteinMLP(nn.Module):
    """
    Multi-Layer Perceptron for protein subcellular location classification.

    Architecture:
        - Input: Protein embeddings (default: 1280 dimensions from ESM2-650m)
        - Hidden layers: 512 -> 256 -> 128
        - Output: 10 classes (subcellular locations)
    """

    def __init__(self, input_dim: int = 1280, num_classes: int = 10):
        """
        Initialize the MLP model.

        Args:
            input_dim: Dimension of input embeddings (default: 1280 for ESM2-650m)
            num_classes: Number of output classes (default: 10 subcellular locations)
        """
        super(ProteinMLP, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


# Subcellular location class labels
CLASS_LABELS = [
    "Cell_membrane",
    "Cytoplasm",
    "Endoplasmic_reticulum",
    "Extracellular",
    "Golgi_apparatus",
    "Lysosome",
    "Mitochondrion",
    "Nucleus",
    "Peroxisome",
    "Plastid",
]
