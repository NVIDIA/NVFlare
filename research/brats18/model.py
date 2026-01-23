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
Model definition for BraTS18 segmentation.
"""
from monai.networks.nets.segresnet import SegResNet


class BratsSegResNet(SegResNet):
    """Wrapper around SegResNet that explicitly stores constructor arguments.

    This is needed for NVFlare's Job API to properly serialize the model configuration,
    since the base SegResNet class doesn't store all arguments as attributes.
    """

    def __init__(
        self,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ):
        super().__init__(
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )
        # Explicitly store constructor arguments for NVFlare serialization
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob
