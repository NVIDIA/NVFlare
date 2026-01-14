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
PyTorch DDP (DistributedDataParallel) client for multi-GPU federated learning.

Launch with:
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=7777 client.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Net
from torch.nn.parallel import DistributedDataParallel as DDP

# (1) import nvflare client API
import nvflare.client as flare

DATASET_PATH = "/tmp/nvflare/data"
CHECKPOINT_PATH = "./cifar_net.pth"


# wraps evaluation logic into a method to re-use for
#       evaluation on both trained and received model
def evaluate(input_weights, device, dataloader):
    """Evaluate model accuracy."""
    net = Net()
    net.load_state_dict(input_weights)
    net.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")
    return accuracy


def main():
    # Initialize distributed process group
    dist.init_process_group("nccl")
    rank = int(dist.get_rank())
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    print(f"DDP rank {rank} initialized on {device}")

    # Data setup
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    epochs = 2

    trainset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model setup
    net = Net()
    criterion = nn.CrossEntropyLoss()

    # (2) initializes NVFlare client API
    flare.init(rank=f"{rank}")

    print(f"flare init DDP rank {rank} initialized on {device}")
    # (3) gets FLModel from NVFlare
    while flare.is_running():
        input_model = flare.receive()
        if rank == 0:
            print(f"\n[Round={input_model.current_round}, Site={flare.get_site_name()}]")
            # (4) loads model from NVFlare
            net.load_state_dict(input_model.params)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # Wrap model with DDP
        net.to(device)
        ddp_model = DDP(net, device_ids=[rank])

        # Sync model across ranks
        if rank == 0:
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

        dist.barrier()
        ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

        # Training loop
        steps = epochs * len(trainloader)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if rank == 0 and i % 2000 == 1999:
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        print(f"Rank {rank}: Finished Training")

        # Only rank 0 sends model back
        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

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
