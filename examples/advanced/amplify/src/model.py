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

"""
Model definition for AMPLIFY.
"""

import torch
from torch import nn
from transformers import AutoModel


class AmplifyRegressor(nn.Module):
    def __init__(self, pretrained_model_name_or_path, layer_sizes, dropout_rate=0.1, num_groups=8):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.num_groups = num_groups

        # Load the pretrained AMPLIFY model from Hugging Face
        self.trunk = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        print("Creating fine-tuning AMPLIFY regression model...")
        print("-" * 80 + "\n")
        print(f"Using trunk: {pretrained_model_name_or_path}")
        print(f"Using layer sizes: {layer_sizes}")
        print(f"Using dropout rate: {dropout_rate}")
        print(f"Using number of groups: {num_groups}")
        print(f"Trunk hidden size: {self.trunk.config.hidden_size}")
        print("-" * 80 + "\n")

        # Create regressor layers dynamically based on layer_sizes
        layers = []
        prev_size = self.trunk.config.hidden_size

        for size in layer_sizes:
            layers.extend(
                [nn.Linear(prev_size, size), nn.GroupNorm(num_groups, size), nn.GELU(), nn.Dropout(dropout_rate)]
            )
            prev_size = size

        # Add final layer to output a single continuous value
        layers.append(nn.Linear(prev_size, 1))

        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, frozen_trunk=True, normalize_hidden_states=True, layer_idx=-1):
        with torch.no_grad() if frozen_trunk else torch.enable_grad():
            h = self.trunk(input_ids, attention_mask, output_hidden_states=True).hidden_states[layer_idx]

        if normalize_hidden_states:
            h = torch.nn.functional.normalize(h, p=2, dim=-1)

        # take mean of the hidden states for sequence regression
        h = h.mean(dim=1)

        # apply regressor
        return self.regressor(h)


def print_model_info(model, layer_sizes, args):
    """Print model architecture and training configuration details."""
    print("\nModel Architecture:")
    print(model)
    print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Trunk parameters:", sum(p.numel() for p in model.trunk.parameters()))
    print("Regressor parameters:", sum(p.numel() for p in model.regressor.parameters()))
    print("\nLayer sizes:", layer_sizes)
    print("Learning rates - Trunk:", args.trunk_lr, "Regressor:", args.regressor_lr)
    print("-" * 80 + "\n")
