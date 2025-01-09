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

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net
from torch.nn.parallel import DistributedDataParallel as DDP

# (1) import nvflare client API
import nvflare.client as flare

DATASET_PATH = "/tmp/nvflare/data"
PATH = "./cifar_net.pth"


# wraps evaluation logic into a method to re-use for
#       evaluation on both trained and received model
def evaluate(input_weights, device, dataloader):
    net = Net()
    net.load_state_dict(input_weights)
    # (optional) use GPU to speed things up
    net.to(device)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
    return 100 * correct // total


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    print(f"Start running basic DDP example on rank {rank}.")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    epochs = 2

    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # (2) initializes NVFlare client API
    flare.init(rank=f"{rank}")

    # (3) gets FLModel from NVFlare
    while flare.is_running():
        input_model = flare.receive()
        if rank == 0:
            print(f"\n[Current Round={input_model.current_round}, Site = {flare.get_site_name()}]\n")

            # (4) loads model from NVFlare
            net.load_state_dict(input_model.params)

        # (optional) use GPU to speed things up
        net.to(device)
        ddp_model = DDP(net, device_ids=[device])

        # From https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#save-and-load-checkpoints
        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            print(f"Saving DDP model on rank {rank}.")
            torch.save(ddp_model.state_dict(), PATH)

        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()
        # configure map_location properly
        map_location = f"cuda:{rank}"
        print(f"Loading DDP model on rank {rank}.")
        ddp_model.load_state_dict(torch.load(PATH, map_location=map_location))

        # (optional) calculate total steps
        steps = epochs * len(trainloader)
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # (optional) use GPU to speed things up
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if rank == 0 and i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        print("Finished Training")

        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(ddp_model.state_dict(), PATH)

            # (5) evaluate on received model for model selection
            accuracy = evaluate(input_model.params, device, testloader)

            # (6) construct trained FL model
            output_model = flare.FLModel(
                params=net.cpu().state_dict(),
                metrics={"accuracy": accuracy},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )
            # (7) send model back to NVFlare
            flare.send(output_model)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
