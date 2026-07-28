# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import lightning as L
import torch
from torch import nn


class LitClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        return nn.functional.cross_entropy(self.layer(features), labels)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train(train_loader):
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(LitClassifier(), train_dataloaders=train_loader)
