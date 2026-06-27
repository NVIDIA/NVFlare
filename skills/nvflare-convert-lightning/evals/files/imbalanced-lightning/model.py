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
import torch.nn as nn
import torch.nn.functional as F


class LitRiskClassifier(pl.LightningModule):
    def __init__(self, input_size=4, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.classifier(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        features, labels, example_scale = batch
        logits = self(features)
        raw_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
        loss = (raw_loss * example_scale).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels, _example_scale = batch
        logits = self(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
