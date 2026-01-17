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

import torch
import torch.nn as nn

from monai.networks.nets import DenseNet121 as MONAIDenseNet121


class DenseNet121(nn.Module):
    """
    Wrapper around MONAI's DenseNet121 implementation.

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
        pretrained: whether to use pretrained weights (only works for 2D).
        progress: whether to show progress bar when downloading pretrained weights.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
        pretrained: bool = False,
        progress: bool = True,
    ) -> None:
        super().__init__()

        # Store configuration for JobAPI
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.growth_rate = growth_rate
        self.block_config = block_config

        # Initialize MONAI's DenseNet121
        self.model = MONAIDenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            bn_size=bn_size,
            act=act,
            norm=norm,
            dropout_prob=dropout_prob,
            pretrained=pretrained,
            progress=progress,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, in_channels, *spatial_dims)

        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        return self.model(x)


# For backward compatibility
Densenet121 = densenet121 = DenseNet121
