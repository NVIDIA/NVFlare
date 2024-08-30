# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net

# (1) import nvflare client API
import nvflare.client as flare

# (optional) set a fix place so we don't need to download everytime
DATASET_PATH = "/tmp/nvflare/data"
# (optional) We change to use GPU to speed things up.
# if you want to use CPU, change DEVICE="cpu"
DEVICE = "cuda:0"
PATH = "./cifar_net.pth"


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()

    # (2) initializes NVFlare client API
    flare.init()

    # (3) decorates with flare.train and load model from the first argument
    # wraps training logic into a method
    @flare.train
    def train(input_model=None, total_epochs=2, lr=0.001):
        net.load_state_dict(input_model.params)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        # (optional) use GPU to speed things up
        net.to(DEVICE)
        # (optional) calculate total steps
        steps = total_epochs * len(trainloader)

        for epoch in range(total_epochs):  # loop over the dataset multiple times

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

        torch.save(net.state_dict(), PATH)

        # (4) construct trained FL model
        output_model = flare.FLModel(params=net.cpu().state_dict(), meta={"NUM_STEPS_CURRENT_ROUND": steps})
        return output_model

    # (5) decorates with flare.evaluate and load model from the first argument
    @flare.evaluate
    def fl_evaluate(input_model=None):
        return {"accuracy": evaluate(input_weights=input_model.params)}

    # wraps evaluate logic into a method
    def evaluate(input_weights):
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

        # return evaluation metrics
        return 100 * correct // total

    while flare.is_running():
        # (6) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        # (7) call fl_evaluate method before training
        #       to evaluate on the received/aggregated model
        global_metric = fl_evaluate(input_model)
        print(f"Accuracy of the global model on the 10000 test images: {global_metric} %")
        # call train method
        train(input_model, total_epochs=2, lr=0.001)
        # call evaluate method
        metric = evaluate(input_weights=torch.load(PATH))
        print(f"Accuracy of the trained model on the 10000 test images: {metric} %")


if __name__ == "__main__":
    main()
