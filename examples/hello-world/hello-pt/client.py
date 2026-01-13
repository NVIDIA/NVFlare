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
client side training scripts
"""

import argparse

import torch
import torchvision
from model import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torchvision.transforms import Compose, Normalize, ToTensor

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

DATASET_PATH = "/tmp/nvflare/data"


def evaluate(net, data_loader, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            # (optional) use GPU to speed things up
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
    return 100 * correct // total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    lr = 0.01
    model = SimpleNetwork()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    # Data transforms
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load datasets
    train_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # (3) initializes NVFlare client API
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    # (optional) metrics tracking
    summary_writer = SummaryWriter()

    while flare.is_running():
        # (4) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")
        # (5) loads model from NVFlare
        model.load_state_dict(input_model.params)
        model.to(device)
        # (6) evaluate on received model for model selection
        accuracy = evaluate(model, test_loader, device)

        # (optional) Task branch for cross-site evaluation
        if flare.is_evaluate():
            print(f"site = {client_name}, running cross-site evaluation")
            # For CSE, just return the evaluation metrics without training
            output_model = flare.FLModel(metrics={"accuracy": accuracy})
            flare.send(output_model)
            continue

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
                if i % 2000 == 1999:
                    avg_loss = running_loss / 2000
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}")

                    # Optional: Log metrics
                    global_step = input_model.current_round * steps + epoch * len(train_loader) + i
                    summary_writer.add_scalar(tag="loss", scalar=avg_loss, global_step=global_step)

                    print(f"site={client_name}, Epoch: {epoch}/{epochs}, Iteration: {i}, Loss: {running_loss}")
                    running_loss = 0.0

        print(f"Finished Training for {client_name}")

        PATH = "./cifar_net.pth"
        torch.save(model.state_dict(), PATH)

        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
