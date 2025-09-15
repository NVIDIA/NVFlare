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

import torch.nn as nn
import torch.nn.functional as F

from ..datasets.utils.configs import TRAIN_SCALE, get_output_num
from .decoder import Decoder, Head


def build_model(tasks, dataname, backbone_type, backbone_pretrained):
    """
    Initialize the local model
    """
    backbone, backbone_channels = get_backbone(backbone_type, backbone_pretrained)
    decoders, heads = get_decoder_head(tasks, dataname, backbone_channels)
    model = MultiDecoderModel(backbone, decoders, heads, tasks)
    return model


def get_backbone(backbone_type, backbone_pretrained):
    """
    Return the backbone
    """
    if backbone_type == 'swin-t':
        from .swin import swin_t

        backbone = swin_t(pretrained=backbone_pretrained)
        backbone_channels = 96
    elif backbone_type == 'swin-s':
        from .swin import swin_s

        backbone = swin_s(pretrained=backbone_pretrained)
        backbone_channels = 96
    elif backbone_type == 'swin-b':
        from .swin import swin_b

        backbone = swin_b(pretrained=backbone_pretrained)
        backbone_channels = 128
    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_decoder_head(tasks, dataname, backbone_channels):
    """
    Return decoders and heads
    """
    input_size = TRAIN_SCALE[dataname]
    enc_out_size = (int(input_size[0] / 32), int(input_size[1] / 32))
    enc_out_dims = [(backbone_channels * 2**i) for i in range(4)]

    decoders = nn.ModuleDict()
    heads = nn.ModuleDict()

    for task in tasks:
        decoders[task] = Decoder(input_size=enc_out_size, in_dims=enc_out_dims, embed_dim=backbone_channels)
        heads[task] = Head(dim=backbone_channels, out_ch=get_output_num(task, dataname))

    return decoders, heads


class MultiDecoderModel(nn.Module):
    """
    Multi-decoder model with shared encoder + task-specific decoders + task-specific heads
    """

    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, heads: nn.ModuleDict, tasks: list):
        super().__init__()
        assert set(decoders.keys()) == set(tasks)
        self.backbone = backbone
        self.decoders = decoders
        self.heads = heads
        self.tasks = tasks

    def forward(self, x):
        out = {}
        img_size = x.size()[2:]

        encoder_output = self.backbone(x)
        for task in self.tasks:
            out[task] = F.interpolate(self.heads[task](self.decoders[task](encoder_output)), img_size, mode='bilinear')
        return out
