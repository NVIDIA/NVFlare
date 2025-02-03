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
from config import NUM_CLIENTS
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_dataloaders(data_chunk):
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

    # split dataset so that each agent has a subset with (distinct) labels
    labels = [(training_data.targets == i).nonzero(as_tuple=True)[0].tolist() for i in range(10)]
    indices = torch.tensor_split(torch.arange(10), NUM_CLIENTS)[data_chunk]
    local_labels = []
    for i in indices:
        local_labels += labels[i]

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(training_data, local_labels),
        batch_size=128,
    )

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128)
    return train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


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
        axs[0].plot(time, loss, label=f"site-{i + 1}")
    axs[0].legend()
    axs[0].set_ylim(-0.1, 3)
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_title("Evolution of Training Loss")

    # Second subplot: Evolution of test loss
    for i in range(num_clients):
        time = test_loss[f"site-{i + 1}"][:, 0]
        loss = test_loss[f"site-{i + 1}"][:, 1]
        axs[1].plot(time, loss, label=f"site-{i + 1}")
    axs[1].legend()
    axs[1].set_ylim(-0.1, 3)
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title("Evolution of Test Loss")

    plt.tight_layout()
    # plt.savefig(f"{job}_results.png")
    plt.show()
