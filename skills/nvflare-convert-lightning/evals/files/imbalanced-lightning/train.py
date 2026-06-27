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
from model import LitRiskClassifier
from torch.utils.data import DataLoader, TensorDataset


FEATURES = torch.tensor(
    [
        [0.2, 0.1, 0.4, 0.7],
        [0.4, 0.2, 0.3, 0.6],
        [0.1, 0.3, 0.8, 0.5],
        [0.7, 0.6, 0.2, 0.4],
        [0.8, 0.9, 0.3, 0.2],
        [0.9, 0.7, 0.5, 0.1],
        [0.3, 0.8, 0.6, 0.4],
        [0.6, 0.4, 0.9, 0.3],
    ],
    dtype=torch.float32,
)
LABELS = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)


def label_derived_scale(labels):
    positives = labels.sum().clamp(min=1.0)
    negatives = (labels.numel() - labels.sum()).clamp(min=1.0)
    minority_scale = negatives / positives
    majority_scale = torch.tensor(1.0)
    return torch.where(labels > 0, minority_scale, majority_scale)


def make_loader(features=FEATURES, labels=LABELS):
    example_scale = label_derived_scale(labels)
    dataset = TensorDataset(features, labels, example_scale)
    return DataLoader(dataset, batch_size=4, shuffle=True)


def main():
    model = LitRiskClassifier(input_size=FEATURES.shape[1])
    train_loader = make_loader()
    val_loader = make_loader()
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
