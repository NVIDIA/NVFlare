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
"""
    client side training scripts
"""

import argparse
import os

import torch
from model import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CIFAR10 Federated Learning Client")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/tmp/nvflare/data",
        help="Path to dataset directory (default: /tmp/nvflare/data)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of local training epochs (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    args = parser.parse_args()

    # Use parsed arguments
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dataset_path = args.dataset_path

    model = SimpleNetwork()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    transforms = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    train_dataset = CIFAR10(
        root=os.path.join(dataset_path, client_name), transform=transforms, download=True, train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validate train_loader is not empty
    if len(train_loader) == 0:
        raise ValueError(
            f"Empty train_loader for site {client_name}. " "This indicates a data preparation or loading issue."
        )

    # Create test dataset and loader for evaluation
    test_dataset = CIFAR10(
        root=os.path.join(dataset_path, client_name), transform=transforms, download=True, train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Validate test_loader is not empty
    if len(test_loader) == 0:
        raise ValueError(
            f"Empty test_loader for site {client_name}. " "This indicates a data preparation or loading issue."
        )

    summary_writer = SummaryWriter()
    while flare.is_running():
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")

        model.load_state_dict(input_model.params)
        model.to(device)

        # Evaluate the global model on test set before local training
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 10:  # Only evaluate on first 10 batches for speed
                    break
                images, labels = batch[0].to(device), batch[1].to(device)
                predictions = model(images)
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"site={client_name}, Global model test accuracy before training: {accuracy:.4f}")

        # Train the model
        model.train()
        steps = epochs * len(train_loader)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                images, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.item()

            # Calculate average loss for the epoch
            avg_loss = running_loss / len(train_loader)
            print(f"site={client_name}, Epoch: {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
            global_step = input_model.current_round * epochs + epoch
            summary_writer.add_scalar(tag="loss_per_epoch", scalar=float(avg_loss), global_step=global_step)

        print(f"Finished Training for {client_name}")

        PATH = "./cifar_net.pth"
        torch.save(model.state_dict(), PATH)

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"accuracy": accuracy},  # Global model test accuracy before training
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        flare.send(output_model)


if __name__ == "__main__":
    main()
