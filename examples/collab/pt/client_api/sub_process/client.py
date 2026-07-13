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

"""DDP Training with checkpoint-based sync.

This example uses checkpoint files to sync model parameters across ranks.
Simple and works everywhere, but has disk I/O overhead.

Launch with:
    torchrun --nproc_per_node=2 -m nvflare.collab.runtime.flare.worker ...
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import nvflare.client as flare
from collab.pt.client_api.sub_process.ddp_utils import receive_with_checkpoint


class SimpleModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def main():
    # Initialize DDP
    dist.init_process_group("gloo")
    rank = dist.get_rank()

    # Data setup
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
    trainloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # Model setup
    net = SimpleModel()
    criterion = nn.MSELoss()

    # Initialize NVFlare Client API
    flare.init(rank=f"{rank}")
    site_name = flare.get_site_name()
    if rank == 0:
        print(f"[{site_name}] DDP initialized with {dist.get_world_size()} ranks")

    # Training loop using helper function
    while True:
        # Use helper function - handles all DDP sync
        input_model, running = receive_with_checkpoint(rank, net)
        if not running:
            break

        if rank == 0:
            print(f"\n[Round={input_model.current_round}, Site={site_name}]")

        optimizer = optim.SGD(net.parameters(), lr=0.01)
        ddp_model = DDP(net)

        # Training
        epochs = 2
        running_loss = 0.0
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            for inputs, labels in trainloader:
                optimizer.zero_grad()
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        # Only rank 0 sends result
        if rank == 0:
            output_model = flare.FLModel(
                params=net.state_dict(),
                metrics={"loss": running_loss},
            )
            flare.send(output_model)
            print(f"  [{site_name}] Sent model, loss={running_loss:.4f}")

    dist.destroy_process_group()
    if rank == 0:
        print(f"[{site_name}] Training complete")


if __name__ == "__main__":
    main()
