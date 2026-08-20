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

import pytorch_lightning as pl
import torch
from torch import nn


def binary_auroc(labels: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    positive_scores = probabilities[labels == 1]
    negative_scores = probabilities[labels == 0]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return torch.tensor(float("nan"), device=probabilities.device)
    comparisons = (positive_scores[:, None] > negative_scores).float()
    ties = (positive_scores[:, None] == negative_scores).float()
    return (comparisons + 0.5 * ties).mean()


class LitSmilesCNN(pl.LightningModule):
    """A compact Lightning character-level CNN for binary classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        dropout: float,
        learning_rate: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size) for kernel_size in (3, 5, 7)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(self.convolutions), 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.validation_labels: list[torch.Tensor] = []
        self.validation_probabilities: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []
        self.test_probabilities: list[torch.Tensor] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = [torch.amax(torch.relu(convolution(embedded)), dim=2) for convolution in self.convolutions]
        features = torch.cat(pooled, dim=1)
        return self.classifier(self.dropout(features)).squeeze(1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        token_ids, labels = batch
        loss = self.criterion(self(token_ids), labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_labels.clear()
        self.validation_probabilities.clear()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        token_ids, labels = batch
        logits = self(token_ids)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.validation_labels.append(labels.detach())
        self.validation_probabilities.append(torch.sigmoid(logits).detach())

    def on_validation_epoch_end(self) -> None:
        labels = torch.cat(self.validation_labels)
        probabilities = torch.cat(self.validation_probabilities)
        accuracy = ((probabilities >= 0.5).float() == labels).float().mean()
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("val_auroc", binary_auroc(labels, probabilities), on_step=False, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.test_labels.clear()
        self.test_probabilities.clear()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        token_ids, labels = batch
        logits = self(token_ids)
        loss = self.criterion(logits, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_labels.append(labels.detach())
        self.test_probabilities.append(torch.sigmoid(logits).detach())

    def on_test_epoch_end(self) -> None:
        labels = torch.cat(self.test_labels)
        probabilities = torch.cat(self.test_probabilities)
        accuracy = ((probabilities >= 0.5).float() == labels).float().mean()
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("test_auroc", binary_auroc(labels, probabilities), on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
