# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TwoMoonsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(data_seed: int):
    X, y = make_moons(n_samples=128, noise=0.1, random_state=data_seed)

    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(y).long()

    X, y = make_moons(n_samples=20, noise=0.1, random_state=42)

    X_test = torch.from_numpy(X).float()
    y_test = torch.from_numpy(y).long()

    train_dataset = TwoMoonsDataset(X_train, y_train)
    test_dataset = TwoMoonsDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20)
    return train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def plot_results(job, num_clients):
    plt.style.use("ggplot")
    train_loss = {}
    test_loss = {}
    for i in range(num_clients):
        train_loss[f"site-{i + 1}"] = torch.load(f"./tmp/runs/{job}/site-{i + 1}/train_loss_sequence.pt")
        test_loss[f"site-{i + 1}"] = torch.load(f"./tmp/runs/{job}/site-{i + 1}/test_loss_sequence.pt")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot: Evolution of training loss
    for i in range(num_clients):
        time = train_loss[f"site-{i + 1}"][:, 0]
        loss = train_loss[f"site-{i + 1}"][:, 1]
        axs[0].plot(range(len(loss)), loss, label=f"site-{i + 1}")
    axs[0].legend()
    axs[0].set_ylim(-0.1, 1)
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_title("Evolution of Training Loss")

    # Second subplot: Evolution of test loss
    for i in range(num_clients):
        time = test_loss[f"site-{i + 1}"][:, 0]
        loss = test_loss[f"site-{i + 1}"][:, 1]
        axs[1].plot(range(len(loss)), loss, label=f"site-{i + 1}")
    axs[1].legend()
    axs[1].set_ylim(-0.1, 1)
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Iteration")
    axs[1].set_title("Evolution of Test Loss")

    plt.tight_layout()
    plt.savefig(f"{job}_results.png")
    # plt.show()
