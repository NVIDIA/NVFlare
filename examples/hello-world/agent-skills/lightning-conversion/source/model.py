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


class LitClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(4, 2)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        loss = torch.nn.functional.cross_entropy(self.network(features), labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        accuracy = (self.network(features).argmax(dim=1) == labels).float().mean()
        self.log("val_accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)
