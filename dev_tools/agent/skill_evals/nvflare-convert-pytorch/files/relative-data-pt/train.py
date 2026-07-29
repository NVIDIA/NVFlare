# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
from pathlib import Path

import torch
from model import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_PATH = Path("data/train.csv")


def load_csv(data_path):
    features = []
    labels = []
    with Path(data_path).open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            features.append([float(row[f"feature_{index}"]) for index in range(4)])
            labels.append(int(row["label"]))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    features, labels = load_csv(args.data_path)
    loader = DataLoader(TensorDataset(features, labels), batch_size=4, shuffle=True)
    model = SimpleNetwork()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for batch_features, batch_labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(batch_features), batch_labels)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
