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
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net

# default dataset path
CIFAR10_ROOT = "/tmp/nvflare/data/cifar10"
# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
DEVICE = "cuda:0"


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=CIFAR10_ROOT, nargs="?")
    parser.add_argument("--batch_size", type=int, default=4, nargs="?")
    parser.add_argument("--num_workers", type=int, default=1, nargs="?")
    parser.add_argument("--model_path", type=str, default=f"{CIFAR10_ROOT}/cifar_net.pth", nargs="?")
    return parser.parse_args()


def _main(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_path = args.dataset_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    model_path = args.model_path

    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # (optional) use GPU to speed things up
    net.to(DEVICE)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # (optional) use GPU to speed things up
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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

    torch.save(net.state_dict(), model_path)

    # wraps evaluation logic into a method to re-use for
    #       evaluation on both trained and received model
    def evaluate(input_weights):
        net = Net()
        net.load_state_dict(input_weights)
        # (optional) use GPU to speed things up
        net.to(DEVICE)

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                # (optional) use GPU to speed things up
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
        return 100 * correct // total

    # evaluation on local trained model
    local_accuracy = evaluate(torch.load(model_path))


def main():
    _main(define_parser())


if __name__ == "__main__":
    main()
