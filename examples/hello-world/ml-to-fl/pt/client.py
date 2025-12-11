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

"""
PyTorch client training script for federated learning.
Supports optional metrics tracking via --use_tracking flag.
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Net

# (1) import nvflare client API
import nvflare.client as flare

DATASET_PATH = "/tmp/nvflare/data"


# (2) wraps evaluation logic into a method to re-use for
#       evaluation on both trained and received model
def evaluate(net, testloader, device):
    """Evaluate model accuracy on test data."""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    print(f"Accuracy of the network on the 10000 test images: {accuracy} %")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_tracking", action="store_true", help="Enable TensorBoard tracking")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize model
    net = Net()

    # (3) initializes NVFlare client API
    flare.init()

    # (optional) metrics tracking
    summary_writer = None
    if args.use_tracking:
        from nvflare.client.tracking import SummaryWriter

        summary_writer = SummaryWriter()

    # FL training loop
    while flare.is_running():
        # (4) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"\n[Round={input_model.current_round}, Site={flare.get_site_name()}]")

        # (5) loads model from NVFlare
        net.load_state_dict(input_model.params)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        net.to(device)

        # Training loop
        steps = args.epochs * len(trainloader)
        for epoch in range(args.epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    avg_loss = running_loss / 2000
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}")

                    # Optional: Log metrics
                    if summary_writer:
                        global_step = input_model.current_round * steps + epoch * len(trainloader) + i
                        summary_writer.add_scalar(tag="loss", scalar=avg_loss, global_step=global_step)

                    running_loss = 0.0

        print("Finished Training")

        # Save local model
        torch.save(net.state_dict(), "./cifar_net.pth")

        # (6) evaluate on received model for model selection
        accuracy = evaluate(net, testloader, device)
        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
