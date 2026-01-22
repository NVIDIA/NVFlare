# Copyright (c) 202, NVIDIA CORPORATION.  All rights reserved.
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
    """A multi-target regression model based on a pretrained transformer trunk.

    This model combines a pretrained transformer model (trunk) with multiple independent
    regressor networks to predict multiple continuous target values. Each target has its
    own dedicated regressor network, allowing for specialized feature learning for each
    target value.

    Architecture:
        - A pretrained transformer trunk that processes input text
        - Multiple independent regressor networks, one for each target value
        - Each regressor consists of multiple linear layers with GroupNorm, GELU activation,
          and dropout for regularization.

    Args:
        pretrained_model_name_or_path (str): Name or path of the pretrained transformer model
        layer_sizes (list[int]): List of hidden layer sizes for each regressor network
        num_targets (int, optional): Number of target values to predict. Defaults to 1
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.1
        num_groups (int, optional): Number of groups for GroupNorm. Defaults to 8

    Forward Args:
        input_ids (torch.Tensor): Input token IDs
        attention_mask (torch.Tensor): Attention mask for the input
        frozen_trunk (bool, optional): Whether to freeze the trunk during forward pass. Defaults to True
        normalize_hidden_states (bool, optional): Whether to normalize hidden states. Defaults to True
        layer_idx (int, optional): Index of the transformer layer to use. Defaults to -1 (the last layer)
        regressor_idx (int or list[int], optional): Index or indices of regressors to use. If None, uses all regressors.
            Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_targets) containing predictions for each target
    """

    def __init__(self, pretrained_model_name_or_path, layer_sizes, num_targets=1, dropout_rate=0.1, num_groups=8):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.num_groups = num_groups
        self.num_targets = num_targets

        # Load the pretrained AMPLIFY model from Hugging Face
        self.trunk = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        print("Creating fine-tuning AMPLIFY regression model...")
        print("-" * 80 + "\n")
        print(f"Using trunk: {pretrained_model_name_or_path}")
        print(f"Using layer sizes: {layer_sizes}")
        print(f"Using dropout rate: {dropout_rate}")
        print(f"Using number of groups: {num_groups}")
        print(f"Number of target values: {num_targets}")
        print(f"Trunk hidden size: {self.trunk.config.hidden_size}")
        print("-" * 80 + "\n")

        # Create separate regressor networks for each target
        self.regressors = nn.ModuleList()
        for _ in range(num_targets):
            layers = []
            prev_size = self.trunk.config.hidden_size

            for size in layer_sizes:
                layers.extend(
                    [nn.Linear(prev_size, size), nn.GroupNorm(num_groups, size), nn.GELU(), nn.Dropout(dropout_rate)]
                )
                prev_size = size

            # Add final layer to output a single value
            layers.append(nn.Linear(prev_size, 1))
            self.regressors.append(nn.Sequential(*layers))

    def forward(
        self,
        input_ids,
        attention_mask,
        frozen_trunk=True,
        normalize_hidden_states=True,
        layer_idx=-1,
        regressor_idx=None,
    ):
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask for the input
            frozen_trunk (bool, optional): Whether to freeze the trunk during forward pass. Defaults to True
            normalize_hidden_states (bool, optional): Whether to normalize hidden states. Defaults to True
            layer_idx (int, optional): Index of the transformer layer to use. Defaults to -1 (the last layer)
            regressor_idx (int or list[int], optional): Index or indices of regressors to use. If None, uses all regressors.
                Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_targets) containing predictions for each target
        """
        with torch.no_grad() if frozen_trunk else torch.enable_grad():
            h = self.trunk(input_ids, attention_mask, output_hidden_states=True).hidden_states[layer_idx]

        if normalize_hidden_states:
            h = torch.nn.functional.normalize(h, p=2, dim=-1)

        # take mean of the hidden states for sequence regression
        h = h.mean(dim=1)

        # apply selected regressors and concatenate the results
        outputs = []
        if regressor_idx is None:
            # Use all regressors
            for regressor in self.regressors:
                outputs.append(regressor(h))
        else:
            # Use only specified regressors
            if isinstance(regressor_idx, int):
                regressor_idx = [regressor_idx]
            for idx in regressor_idx:
                outputs.append(self.regressors[idx](h))

        return torch.cat(outputs, dim=1)


def print_model_info(model, layer_sizes, args):
    """Print model architecture and training configuration details."""
    print("\nModel Architecture:")
    print(model)
    print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Trunk parameters:", sum(p.numel() for p in model.trunk.parameters()))
    print("Regressor parameters:", sum(p.numel() for p in model.regressors.parameters()))
    print("\nLayer sizes:", layer_sizes)
    print("Number of target values:", model.num_targets)
    print("Learning rates - Trunk:", args.trunk_lr, "Regressor:", args.regressor_lr)
    print("-" * 80 + "\n")
