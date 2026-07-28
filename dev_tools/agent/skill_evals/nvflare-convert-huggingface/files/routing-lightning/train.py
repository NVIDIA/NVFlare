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
