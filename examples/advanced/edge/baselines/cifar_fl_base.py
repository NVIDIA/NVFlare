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
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.edge.models.model import Cifar10ConvNet

CIFAR10_ROOT = "/tmp/nvflare/datasets/cifar10"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    # Data loading code
    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 4
    train_set = datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=True, transform=transform)

    net = Cifar10ConvNet()
    tb_writer = SummaryWriter()

    # wraps evaluation logic into a method to re-use for
    #       evaluation on both trained and received model
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

    # (2) initialize NVFlare client API
    flare.init()
    client_id = flare.get_site_name()
    # Indices according to client_id number
    # find the number in client_id string
    client_id = int(client_id.split("-")[-1]) - 1
    increment = 3125
    indices = list(range(client_id * increment, (client_id + 1) * increment))
    train_subset = Subset(train_set, indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # (3) run continously when launch_once=true
    while flare.is_running():
        # (4) receive FLModel from NVFlare
        input_model = flare.receive()
        cur_round = input_model.current_round
        total_rounds = input_model.total_rounds
        print(f"({client_id}) current_round={cur_round}, total_rounds={total_rounds}")

        # Evaluate global model
        global_acc = evaluate(input_model.params)
        tb_writer.add_scalar("accuracy", global_acc, cur_round)

        # (5.1) loads model from NVFlare
        net.load_state_dict(input_model.params)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        net.to(DEVICE)

        for epoch in range(4):
            local_base_step = (cur_round * 4 + epoch) * len(train_loader)
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
                    tb_writer.add_scalar("loss", running_loss / 250, local_base_step + i)
                    running_loss = 0.0

        print(f"({client_id}) Finished Training")
        # Save the final model
        model_name = "cifar_net.pth"
        torch.save(input_model.params, model_name)

        # (5.4) construct trained FL model
        output_model = flare.FLModel(
            params=net.cpu().state_dict(),
            metrics={"accuracy": global_acc},
            meta={"NUM_STEPS_CURRENT_ROUND": len(train_loader)},
        )

        # (5.5) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
