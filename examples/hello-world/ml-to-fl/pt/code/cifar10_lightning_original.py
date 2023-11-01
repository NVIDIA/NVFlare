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

import torch
import torchvision
import torchvision.transforms as transforms
from lit_net import LitNet
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from torch.utils.data import DataLoader, random_split

seed_everything(7)


DATASET_PATH = "/tmp/nvflare/data"
BATCH_SIZE = 4

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir: str = DATASET_PATH, batch_size: int = BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=transform)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "validate":
            cifar_full = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=False, transform=transform
            )
            self.cifar_train, self.cifar_val = random_split(cifar_full, [0.8, 0.2])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage == "predict":
            self.cifar_test = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=False, transform=transform
            )

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)


def main():
    model = LitNet()
    cifar10_dm = CIFAR10DataModule()

    trainer = Trainer(max_epochs=1, devices=1 if torch.cuda.is_available() else None)

    # perform local training
    print("--- train new model ---")
    trainer.fit(model, datamodule=cifar10_dm)

    # test local model
    print("--- test new model ---")
    trainer.test(ckpt_path="best", datamodule=cifar10_dm)

    # get predictions
    print("--- prediction with new best model ---")
    trainer.predict(ckpt_path="best", datamodule=cifar10_dm)


if __name__ == "__main__":
    main()
