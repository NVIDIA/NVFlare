# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn.functional as F


class Head(nn.Module):
    """
    Head block for different tasks. Upsamples twice and applies a 1x1 convolution.
    """

    def __init__(self, dim, out_ch):
        super().__init__()
        self.upsample1 = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2)
        self.last_conv = nn.Conv2d(dim // 4, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.last_conv(x)
        return x


class Decoder(nn.Module):
    """
    Transform function
    """

    def __init__(self, input_size, in_dims, embed_dim):
        super().__init__()
        assert len(in_dims) == 4  # features from encoder layers
        # [(H/4, W/4), (H/8, W/8), (H/16, W/16), (H/32, W/32)]
        self.feature_sizes = [(input_size[0] * 2 ** (3 - i), input_size[1] * 2 ** (3 - i)) for i in range(4)]
        self.in_dims = in_dims
        self.embed_dim = embed_dim

        self.linears = nn.ModuleList()
        for dim in in_dims:
            self.linears.append(nn.Linear(dim, embed_dim))

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.LeakyReLU(inplace=True)
        )

    def forward(self, inputs):
        B = inputs[0].shape[0]

        feas = []
        for i in range(4):
            # assert len(inputs[i].shape) == 3  # B, L, C
            assert inputs[i].shape[1] == self.in_dims[i], "Input feature dimension mismatch"
            fea = inputs[i].reshape(B, self.in_dims[i], -1).permute(0, 2, 1)  # B, h*w, C
            # Dimension reduction
            fea = self.linears[i](fea)
            # B, h*w, C => B, C, h*w => B, C, h, w
            fea = fea.permute(0, 2, 1).reshape(B, self.embed_dim, self.feature_sizes[i][0], self.feature_sizes[i][1])
            # B, C, h, w => B, C, H/4, W/4
            fea = F.interpolate(fea, size=self.feature_sizes[0], mode='bilinear', align_corners=False)
            feas.append(fea)

        # B, 4*C, H/4, W/4 => B, C, H/4, W/4
        x = self.linear_fuse(torch.cat(feas, dim=1).contiguous())
        return x
