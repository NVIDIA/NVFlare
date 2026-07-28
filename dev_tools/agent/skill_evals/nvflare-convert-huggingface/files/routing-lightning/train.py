# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import lightning as L
import torch
from transformers import AutoModel


class LitEncoder(L.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def training_step(self, batch, batch_idx):
        return self.encoder(**batch).last_hidden_state.mean()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


def train(model_name, train_loader):
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(LitEncoder(model_name), train_dataloaders=train_loader)
