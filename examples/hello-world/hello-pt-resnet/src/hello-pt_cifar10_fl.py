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

import os

import torch
from resnet_18 import Resnet18
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

DATASET_PATH = "/tmp/nvflare/data"


def main():
    batch_size = 4
    epochs = 5
    lr = 0.01
    model = Resnet18(num_classes=10)
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
        root=os.path.join(DATASET_PATH, client_name),
        transform=transforms,
        download=True,
        train=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    summary_writer = SummaryWriter()
    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        model.load_state_dict(input_model.params)
        model.to(device)

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

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if i % 3000 == 0:
                    print(f"Epoch: {epoch}/{epochs}, Iteration: {i}, Loss: {running_loss / 3000}")
                    global_step = input_model.current_round * steps + epoch * len(train_loader) + i
                    summary_writer.add_scalar(
                        tag="loss_for_each_batch",
                        scalar=running_loss,
                        global_step=global_step,
                    )
                    running_loss = 0.0

        print("Finished Training")

        PATH = "./cifar_net.pth"
        torch.save(model.state_dict(), PATH)

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()
