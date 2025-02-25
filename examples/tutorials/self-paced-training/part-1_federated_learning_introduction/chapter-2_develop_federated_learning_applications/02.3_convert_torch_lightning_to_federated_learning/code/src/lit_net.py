# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

NUM_CLASSES = 10
criterion = nn.CrossEntropyLoss()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LitNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = Net()
        self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.valid_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        # (optional) pass additional information via self.__fl_meta__
        self.__fl_meta__ = {}

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, labels = batch
        outputs = self(x)
        loss = criterion(outputs, labels)
        self.train_acc(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        return loss

    def evaluate(self, batch, stage=None):
        x, labels = batch
        outputs = self(x)
        loss = criterion(outputs, labels)
        self.valid_acc(outputs, labels)

        if stage:
            self.log(f"{stage}_loss", loss)
            self.log(f"{stage}_acc", self.valid_acc, on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.evaluate(batch)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return {"optimizer": optimizer}
