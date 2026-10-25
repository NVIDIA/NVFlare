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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Pooling and Top-k MIL model construction for FABRIC."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class DTFDMIL(nn.Module):
    """A compact double-tier MIL model for patient bags.

    This follows the DTFD idea at training time: split a large bag into
    pseudo-bags, pool each pseudo-bag, then pool the pseudo-bag embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        attn_dim: int = 256,
        pseudo_bags: int = 8,
        dropout: float = 0.25,
        n_classes: int = 2,
    ):
        super().__init__()
        self.pseudo_bags = max(1, int(pseudo_bags))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.local_attention = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.global_attention = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    @staticmethod
    def _attention_pool(
        h: torch.Tensor,
        attention: nn.Module,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = attention(h).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=0)
        if return_attention:
            return pooled, scores, weights
        return pooled

    def _forward_one(
        self,
        bag: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(bag.float())
        n_instances = h.shape[0]
        if return_attention:
            return self._attention_pool(h, self.local_attention, return_attention=True)
        if self.training and self.pseudo_bags > 1 and n_instances >= self.pseudo_bags:
            order = torch.randperm(n_instances, device=h.device)
            chunks = torch.chunk(h[order], self.pseudo_bags, dim=0)
            pseudo_embeddings = [
                self._attention_pool(chunk, self.local_attention) for chunk in chunks if chunk.numel() > 0
            ]
            tier1 = torch.stack(pseudo_embeddings, dim=0)
            return self._attention_pool(tier1, self.global_attention)
        return self._attention_pool(h, self.local_attention)

    def forward(self, data: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        if return_attention:
            pooled_rows = []
            score_rows = []
            weight_rows = []
            for i in range(data.shape[0]):
                pooled, scores, weights = self._forward_one(data[i], return_attention=True)
                pooled_rows.append(pooled)
                score_rows.append(scores)
                weight_rows.append(weights)
            pooled = torch.stack(pooled_rows, dim=0)
            logits = self.classifier(pooled)
            return {
                "logits": logits,
                "dtfd_attention_scores": score_rows,
                "dtfd_attention_weights": weight_rows,
            }

        pooled = torch.stack([self._forward_one(data[i]) for i in range(data.shape[0])], dim=0)
        logits = self.classifier(pooled)
        return {"logits": logits}


class MILModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_dim: int,
        n_classes: int = 2,
        embed_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.25,
        dtfd_pseudo_bags: int = 8,
        dtfd_top_k: int = 1,
        dtfd_eval_group_seed: int = 42,
    ):
        super().__init__()
        if model_name == "dtfd_mil":
            self.model = DTFDMIL(
                input_dim,
                embed_dim=embed_dim,
                attn_dim=attn_dim,
                pseudo_bags=dtfd_pseudo_bags,
                dropout=dropout,
                n_classes=n_classes,
            )
        elif model_name == "dtfd_topk":
            from .topk_mil import DTFDTopKMIL

            self.model = DTFDTopKMIL(
                input_dim,
                embed_dim=embed_dim,
                attn_dim=attn_dim,
                pseudo_bags=dtfd_pseudo_bags,
                dropout=dropout,
                n_classes=n_classes,
                top_k=dtfd_top_k,
                eval_group_seed=dtfd_eval_group_seed,
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def forward(self, data: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return self.model(data=data, **kwargs)


def build_mil_model(
    model_name: str,
    input_dim: int,
    n_classes: int = 2,
    embed_dim: int = 512,
    attn_dim: int = 256,
    dropout: float = 0.25,
    dtfd_pseudo_bags: int = 8,
    dtfd_top_k: int = 1,
    dtfd_eval_group_seed: int = 42,
) -> MILModel:
    return MILModel(
        model_name=model_name,
        input_dim=input_dim,
        n_classes=n_classes,
        embed_dim=embed_dim,
        attn_dim=attn_dim,
        dropout=dropout,
        dtfd_pseudo_bags=dtfd_pseudo_bags,
        dtfd_top_k=dtfd_top_k,
        dtfd_eval_group_seed=dtfd_eval_group_seed,
    )
