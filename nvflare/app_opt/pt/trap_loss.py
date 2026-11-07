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

import math

import torch
from torch import nn

from nvflare.fuel.utils.experimental import experimental


def causal_token_log_probs(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """Return shifted observed-token log probabilities and the valid loss mask.

    Labels must already mask padding and any prompt tokens. Probabilities use
    float32 for numerical stability; gradients still flow into target logits.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch, length, vocabulary] and labels [batch, length]")
    if labels.dtype != torch.long or labels.device != logits.device:
        raise ValueError("Labels must be int64 on the logits device")
    observed = labels[:, 1:]
    mask = observed != ignore_index
    safe_labels = observed.masked_fill(~mask, 0)
    log_probs = logits[:, :-1].float().log_softmax(-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return log_probs.masked_fill(~mask, 0), mask


class PTTRAPLoss(nn.Module):
    """Experimental Target Reference Advantage Penalty, added to ordinary CE.

    Computes weight * mean(relu(target_log_probs - reference_log_probs - threshold))
    over valid tokens. Reference values are detached. With gradient accumulation,
    pass the total valid-token count of the accumulation window as normalizer and
    sum microbatch losses (use the same normalization for ordinary CE).
    """

    @experimental("PTTRAPLoss API may change")
    def __init__(self, weight: float = 1.0, threshold: float = 0.0):
        super().__init__()
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("weight and threshold must be finite and nonnegative")
        self.weight = weight
        self.threshold = threshold

    def forward(self, target_log_probs, reference_log_probs, mask, normalizer=None):
        self._validate(target_log_probs, reference_log_probs, mask)
        selected = target_log_probs.masked_select(mask)
        count = mask.sum() if normalizer is None else torch.as_tensor(normalizer, device=mask.device)
        if count.ndim or not torch.isfinite(count) or count < mask.sum():
            raise ValueError("normalizer must be finite and at least the local valid-token count")
        if not self.weight:
            return selected[:0].sum()
        ref = reference_log_probs.detach().masked_select(mask)
        return self.weight * (selected.float() - ref.float() - self.threshold).relu().sum() / count.clamp_min(1)

    @staticmethod
    def _validate(target, reference, mask):
        if target.shape != reference.shape or target.shape != mask.shape:
            raise ValueError("Target, reference, and mask shapes must match")
        if mask.dtype != torch.bool or target.device != reference.device or target.device != mask.device:
            raise ValueError("A boolean mask and matching devices are required")
        if not target.is_floating_point() or not reference.is_floating_point():
            raise ValueError("Log probabilities must be floating point")

    @torch.no_grad()
    def diagnostics(self, target_log_probs, reference_log_probs, mask):
        self._validate(target_log_probs, reference_log_probs, mask)
        advantage = (target_log_probs.float() - reference_log_probs.detach().float()).masked_select(mask)
        count = advantage.numel()
        return {
            "valid_tokens": count,
            "mean_advantage": advantage.mean().item() if count else 0.0,
            "active_fraction": (advantage > self.threshold).float().mean().item() if count else 0.0,
            "penalty": self(target_log_probs, reference_log_probs, mask).item(),
        }
