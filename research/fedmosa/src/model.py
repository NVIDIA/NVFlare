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

"""MoSA: SAM ViT-B with modality-specialized and shared adapters in every image-encoder block."""

import torch
import torch.nn as nn
from segment_anything import sam_model_registry


class BottleneckAdapter(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.act = nn.ReLU()
        self.up = nn.Linear(dim // reduction, dim)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class ModalityAdapter(nn.Module):
    """One expert per modality; hard-routed by the modality of the current batch."""

    def __init__(self, dim, modalities, reduction=16):
        super().__init__()
        self.modalities = list(modalities)
        self.experts = nn.ModuleList([BottleneckAdapter(dim, reduction) for _ in self.modalities])
        self.current = 0

    def forward(self, x):
        return self.experts[self.current](x)


class MoSABlock(nn.Module):
    """Frozen SAM block + modality-specialized adapters + shared adapters (attention and MLP branches).

    The shared adapters keep a running mean of their outputs (the shared prototype) for prototype alignment.
    """

    def __init__(self, block, dim, modalities, reduction=16):
        super().__init__()
        self.block = block
        self.adapter_attn = ModalityAdapter(dim, modalities, reduction)
        self.adapter_mlp = ModalityAdapter(dim, modalities, reduction)
        self.shared_adapter_attn = BottleneckAdapter(dim, reduction)
        self.shared_adapter_mlp = BottleneckAdapter(dim, reduction)
        for p in self.block.parameters():
            p.requires_grad = False
        self.reset_prototypes()

    def reset_prototypes(self):
        self.proto_sum = {"attn_shared": None, "mlp_shared": None}
        self.proto_count = 0
        self.current_proto = {}

    def _record(self, key, out):
        proto = out.flatten(0, -2).mean(dim=0)
        self.current_proto[key] = proto
        s = self.proto_sum[key]
        self.proto_sum[key] = proto.detach() if s is None else s + proto.detach()

    def forward(self, x):
        b = self.block
        h = b.norm1(x)
        with torch.no_grad():
            attn = b.attn(h)
        shared = self.shared_adapter_attn(h)
        self._record("attn_shared", shared)
        x = x + attn + self.adapter_attn(h) + shared

        h = b.norm2(x)
        with torch.no_grad():
            mlp = b.mlp(h)
        shared = self.shared_adapter_mlp(h)
        self._record("mlp_shared", shared)
        self.proto_count += 1
        return x + mlp + self.adapter_mlp(h) + shared

    def mean_prototypes(self):
        if self.proto_count == 0:
            return None
        return {k: v / self.proto_count for k, v in self.proto_sum.items()}


class MoSA(nn.Module):
    def __init__(self, checkpoint=None, modalities=("CT", "MR"), reduction=16):
        super().__init__()
        self.modalities = list(modalities)
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        blocks = self.sam.image_encoder.blocks
        dim = blocks[0].mlp.lin1.in_features
        for i, blk in enumerate(blocks):
            blocks[i] = MoSABlock(blk, dim, self.modalities, reduction)
        for name, p in self.sam.image_encoder.named_parameters():
            if "adapter" not in name:
                p.requires_grad = False
        for p in self.sam.prompt_encoder.parameters():  # no prompts are used; kept frozen
            p.requires_grad = False

    @property
    def blocks(self):
        return self.sam.image_encoder.blocks

    def set_modality(self, modality):
        idx = self.modalities.index(modality)
        for blk in self.blocks:
            blk.adapter_attn.current = idx
            blk.adapter_mlp.current = idx

    def forward(self, images):
        emb = self.sam.image_encoder(images)
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        masks, _ = self.sam.mask_decoder(
            image_embeddings=emb,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return masks

    def trainable_state(self):
        """Parameters exchanged with the server (adapters, mask decoder)."""
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def current_prototypes(self):
        return [blk.current_proto for blk in self.blocks]

    def mean_prototypes(self):
        return [blk.mean_prototypes() for blk in self.blocks]

    def reset_prototypes(self):
        for blk in self.blocks:
            blk.reset_prototypes()
