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

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nvflare.edge.models.model import Cifar10ConvNet

CIFAR10_ROOT = "/tmp/nvflare/datasets/cifar10"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    # Data loading code
    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 4
    train_set = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Training configurations
    workspace_root = "/tmp/nvflare/workspaces/cifar10_cen"
    net = Cifar10ConvNet().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    tb_writer = SummaryWriter(workspace_root)

    def evaluate(input_weights):
        net = Cifar10ConvNet()
        net.load_state_dict(input_weights)
        net.to(DEVICE)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                # (optional) use GPU to speed things up
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                # calculate outputs by running images through the network
                outputs = net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct // total

    # Training loop
    for epoch in range(10):
        print(f"Epoch: {epoch}")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # record loss every 250 mini-batches (1000 samples)
            if i % 250 == 249:
                tb_writer.add_scalar("loss", running_loss / 250, epoch * len(train_loader) + i)
                print(f"[{epoch}, {i}] loss: {running_loss / 250}")
                running_loss = 0.0

        # Evaluate global model
        acc = evaluate(net.state_dict())
        tb_writer.add_scalar("accuracy", acc, epoch)
        print(f"Epoch {epoch} accuracy: {acc}")

    # Save the final model
    model_name = "cifar_net.pth"
    torch.save(net.state_dict(), f"{workspace_root}/{model_name}")


if __name__ == "__main__":
    main()
