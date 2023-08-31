# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy

# (2.0) import nvflare lightning client API
import nvflare.client.lightning as flare

seed_everything(7)

NUM_CLASSES = 10

DATASET_PATH = "/tmp/nvflare/data"
BATCH_SIZE = 4

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
criterion = nn.CrossEntropyLoss()


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str = DATASET_PATH, batch_size: int = BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
        self.cifar_test = torchvision.datasets.CIFAR10(
            root=DATASET_PATH, train=False, download=True, transform=transform
        )
        self.cifar_train, self.cifar_val = random_split(dataset, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)


class LitNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = Net()
        self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.valid_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)

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


def main():
    model = LitNet()
    cifar10_dm = CIFAR10DataModule()

    trainer = Trainer(max_epochs=1, accelerator="auto", devices=1 if torch.cuda.is_available() else None)
    # (2.1) patch the lightning trainer
    flare.patch(trainer)

    # (2.2) evaluate the current global model to allow server-side model selection
    print("--- validate global model ---")
    trainer.validate(model, datamodule=cifar10_dm)

    # (1.1) perform local training starting with the received global model
    print("--- train new model ---")
    trainer.fit(model, datamodule=cifar10_dm)

    # (1.2) optionally test the new local model
    print("--- test new model ---")
    trainer.test(ckpt_path="best", datamodule=cifar10_dm)

    # (1.3) optionally get predictions
    print("--- prediction with new best model ---")
    predictions = trainer.predict(ckpt_path="best", datamodule=cifar10_dm)


if __name__ == "__main__":
    main()
