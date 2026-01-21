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

"""Simple PyTorch training script for cross-site evaluation example.

This script demonstrates the Client API pattern for cross-site evaluation:
- Normal training flow: receive model, train, evaluate, send back
- CSE flow: check flare.is_evaluate(), evaluate only, return metrics
"""

import argparse

import torch
import torchvision
from model import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torchvision.transforms import Compose, Normalize, ToTensor

import nvflare.client as flare

DATASET_PATH = "/tmp/nvflare/data/cifar10"
CLIENT_MODEL_DIR = "/tmp/nvflare/client_pretrain_models"


def evaluate(net, test_loader, device):
    """Evaluate the network on test data."""
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    lr = 0.01

    # Initialize model
    model = SimpleNetwork()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    # Data transforms
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load test dataset only (no training in standalone CSE)
    test_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize NVFlare client API
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    print(f"Client {client_name} initialized")

    while flare.is_running():
        # Receive FLModel from NVFlare
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")

        # Handle cross-site evaluation task
        if flare.is_evaluate():
            print(f"site = {client_name}, running cross-site evaluation")

            # Load the model to evaluate
            model.load_state_dict(input_model.params)
            model.to(device)

            # Evaluate the received model on local test data
            accuracy = evaluate(model, test_loader, device)

            # Return evaluation metrics
            output_model = flare.FLModel(metrics={"accuracy": accuracy})
            flare.send(output_model)
        elif flare.is_submit_model():
            print(f"site = {client_name}, loading pre-trained local model")

            # Load pre-trained model for this client site
            import os

            client_model_path = os.path.join(CLIENT_MODEL_DIR, f"{client_name}.pt")

            if not os.path.exists(client_model_path):
                raise ValueError(f"Pre-trained model not found at {client_model_path}")

            # Load pre-trained weights
            model.load_state_dict(torch.load(client_model_path, map_location=device))
            model.to(device)

            # Evaluate the pre-trained local model
            accuracy = evaluate(model, test_loader, device)

            # Submit the pre-trained local model
            output_model = flare.FLModel(
                params=model.cpu().state_dict(),
            )

            flare.send(output_model)
        else:
            raise ValueError("Unexpected task type")


if __name__ == "__main__":
    main()
