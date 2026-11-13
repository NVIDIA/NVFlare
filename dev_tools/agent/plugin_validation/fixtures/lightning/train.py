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

"""Small CPU Lightning workload with no downloaded data or weights."""

import json

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Regression(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, features):
        return self.linear(features)

    def training_step(self, batch, batch_idx):
        features, target = batch
        return nn.functional.mse_loss(self(features), target)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def create_model():
    return Regression()


def main():
    pl.seed_everything(7, workers=True)
    torch.set_num_threads(1)
    features = torch.arange(64, dtype=torch.float32).reshape(32, 2) / 64
    labels = 2 * features[:, :1] - features[:, 1:] + 0.25
    loader = DataLoader(TensorDataset(features, labels), batch_size=8, shuffle=False)
    model = create_model()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=loader)
    with torch.no_grad():
        loss = nn.functional.mse_loss(model(features), labels).item()
    print(json.dumps({"framework": "lightning", "loss": loss, "examples": len(features)}))


if __name__ == "__main__":
    main()
