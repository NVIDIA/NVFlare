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

import torch
from torch import nn


class SmilesCNN(nn.Module):
    """A compact character-level CNN for binary classification."""

    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size) for kernel_size in (3, 5, 7)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(self.convolutions), 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = [torch.amax(torch.relu(convolution(embedded)), dim=2) for convolution in self.convolutions]
        features = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(features)).squeeze(1)
