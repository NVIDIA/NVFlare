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

"""Optional two-tier MIL with pseudo-bag supervision and high/low patch selection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mil_models import DTFDMIL


class DTFDTopKMIL(DTFDMIL):
    """Keep the pooling model's layers, adding a supervised Tier-1 classifier.

    Both training and inference partition the patient bag, score patches using
    the Tier-1 classifier, collect high/low-scoring patch embeddings, and run
    Tier-2 attention and classification. No labels are needed for prediction.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        attn_dim: int = 256,
        pseudo_bags: int = 8,
        dropout: float = 0.25,
        n_classes: int = 2,
        top_k: int = 1,
        eval_group_seed: int = 42,
    ):
        if n_classes != 2:
            raise ValueError("Top-k recurrence selection requires two classes (0, 1)")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer per high/low set per pseudo-bag")
        super().__init__(input_dim, embed_dim, attn_dim, pseudo_bags, dropout, n_classes)
        self.top_k = top_k
        self.eval_group_seed = int(eval_group_seed)
        self.pseudo_classifier = nn.Linear(embed_dim, n_classes)

    def _forward_topk_one(self, bag: torch.Tensor, details: bool):
        if bag.ndim != 2 or bag.shape[0] == 0:
            raise ValueError("Expected a nonempty [patches, features] patient bag")
        h = self.encoder(bag.float())
        if self.training:
            order = torch.randperm(h.shape[0], device=h.device)
        else:
            # A private generator makes grouping repeatable without changing
            # global RNG state or depending on batch order or patient labels.
            generator = torch.Generator(device=h.device).manual_seed(self.eval_group_seed)
            order = torch.randperm(h.shape[0], generator=generator, device=h.device)
        # Split into nonempty pseudo-bags; torch.chunk may return fewer groups
        # than requested for some bag sizes.
        groups = torch.chunk(order, self.pseudo_bags)
        pseudo_logits, selected_groups, group_probs = [], [], []
        for indices in groups:
            chunk = h.index_select(0, indices)
            pooled, _, weights = self._attention_pool(chunk, self.local_attention, return_attention=True)
            pseudo_logits.append(self.pseudo_classifier(pooled))
            # Class-activation scores use the learned classifier on each
            # attention-weighted patch feature, not attention weights alone.
            # Rank the binary logit margin: it has exactly the same ordering
            # as P(recurrence), without softmax saturation creating extra ties.
            with torch.no_grad(), torch.autocast(device_type=h.device.type, enabled=False):
                patch_logits = F.linear(
                    chunk.float() * weights.float().unsqueeze(-1),
                    self.pseudo_classifier.weight.float(),
                    self.pseudo_classifier.bias.float(),
                )
                rank = torch.argsort(patch_logits[:, 1] - patch_logits[:, 0], descending=True, stable=True)
                k = min(self.top_k, len(indices))
                # Union of the high/low sets: small groups never duplicate a patch.
                selected = torch.cat((rank[:k], rank[max(k, len(indices) - k) :]))
            selected_groups.append(indices.index_select(0, selected))
            if details:
                group_probs.append(torch.softmax(patch_logits, dim=-1)[:, 1])

        selected_indices = torch.cat(selected_groups)
        # Collect original projected PATCH embeddings, not pooled pseudo-bag vectors.
        distilled = h.index_select(0, selected_indices)
        pooled, _, tier2_weights = self._attention_pool(distilled, self.global_attention, return_attention=True)
        extra = None
        if details:
            extra = {
                "pseudo_bag_indices": groups,
                "selected_patch_indices": selected_groups,
                "patch_recurrence_probs": group_probs,
                "tier2_attention_weights": tier2_weights,
            }
        return pooled, torch.stack(pseudo_logits), extra

    def forward(self, data: torch.Tensor, return_attention: bool = False) -> dict:
        if data.ndim != 3 or data.shape[0] == 0:
            raise ValueError("Expected [patients, patches, features]")
        rows = [self._forward_topk_one(bag, return_attention) for bag in data]
        pooled = torch.stack([row[0] for row in rows])
        output = {"logits": self.classifier(pooled), "pseudo_logits": [row[1] for row in rows]}
        if return_attention:
            output["topk_details"] = [row[2] for row in rows]
        return output


def forward_topk_bags(model, bags, labels, criterion, device):
    """Return patient logits and class-weighted, patient-averaged pseudo-bag CE.

    Each pseudo-bag inherits its patient's label as weak supervision. Average
    within each patient first so a tiny bag with fewer groups is not downweighted.
    The class-weight denominator matches the existing patient-level CE.
    """
    if not isinstance(criterion, nn.CrossEntropyLoss) or criterion.reduction != "mean":
        raise TypeError("Expected the existing mean-reduced cross-entropy criterion")
    if not len(bags) or len(bags) != len(labels):
        raise ValueError("Expected one label per patient in a nonempty batch")
    logits, pseudo_losses = [], []
    for bag, label in zip(bags, labels):  # noqa: B905
        output = model(bag.unsqueeze(0).to(device, non_blocking=True))
        logits.append(output["logits"].squeeze(0))
        pseudo = output["pseudo_logits"][0]
        pseudo_losses.append(
            F.cross_entropy(
                pseudo,
                label.expand(len(pseudo)),
                weight=criterion.weight,
                reduction="none",
            ).mean()
        )
    denominator = criterion.weight[labels].sum() if criterion.weight is not None else len(labels)
    return torch.stack(logits), torch.stack(pseudo_losses).sum() / denominator
