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

"""Small classifier-logit completion model exchanged by FedAvg."""

import torch
import torch.nn as nn
from torch import Tensor


class LogitCompletionModel(nn.Module):
    """Predict an additive logit residual from a target-missing hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1, seed: int = 7) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.seed = int(seed)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.network = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, missing_features: Tensor) -> Tensor:
        delta = self.network(missing_features.float()).squeeze(-1)
        return torch.nan_to_num(delta, nan=0.0, posinf=20.0, neginf=-20.0)


def effect_target(full_logits: Tensor, missing_logits: Tensor) -> Tensor:
    """Return the target modality's additive contribution in classifier-logit space."""

    return full_logits.float() - missing_logits.float()


def completed_logits(missing_logits: Tensor, predicted_delta: Tensor, alpha: float) -> Tensor:
    """Apply a completion scale; alpha=0 is an exact identity operation."""

    return missing_logits + float(alpha) * predicted_delta
