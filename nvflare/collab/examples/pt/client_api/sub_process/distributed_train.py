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

"""DDP-based Local Training (Single Node, Multi-GPU).

This script demonstrates standard PyTorch Distributed Data Parallel (DDP)
training on a single node with multiple GPUs.

Supports two modes:
1. Standalone mode: Run directly with torchrun for pure DDP training
2. FL mode: Called by sim_distributed_fedavg_train.py with input/output weights

Usage (standalone):
    torchrun --nproc_per_node=2 distributed_train.py

Usage (FL mode - called by sim_distributed_fedavg_train.py):
    FL_WEIGHTS_FILE=input.pt FL_OUTPUT_FILE=output.pt torchrun ... distributed_train.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# Simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def train(global_weights=None, num_epochs=5, client_id="local"):
    """Local DDP training on single node with multiple GPUs.

    Args:
        global_weights: Optional initial weights (for FL mode)
        num_epochs: Number of training epochs
        client_id: Client identifier for logging

    Returns:
        Tuple of (final_weights, final_loss)
    """
    # Initialize distributed process group
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"[{client_id}] Starting DDP training with {world_size} processes")

    # Create dataset (in real scenario, each rank would load different data shards)
    # Use client_id to seed different data for each client
    torch.manual_seed(hash(client_id) % 2**32 + rank)
    inputs = torch.randn(100, 10)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(inputs, labels)

    # DistributedSampler ensures each GPU gets different data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # Create model and load global weights if provided
    model = SimpleModel()
    if global_weights is not None and len(global_weights) > 0:
        model.load_state_dict(global_weights)

    # Wrap model with DDP for gradient synchronization
    model = DDP(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    final_loss = 0.0
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        epoch_loss = 0.0
        num_batches = 0

        for batch_inputs, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()  # DDP automatically syncs gradients
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        final_loss = epoch_loss / num_batches
        if rank == 0:
            print(f"[{client_id}]   Epoch {epoch + 1}/{num_epochs}, Loss: {final_loss:.4f}")

    # Get final model weights (from rank 0)
    final_weights = model.module.state_dict()

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        print(f"[{client_id}] DDP training completed")

    return final_weights, final_loss, rank


def main():
    """Main entry point - handles both standalone and FL modes."""
    # Check for FL mode (called by sim_distributed_fedavg_train.py)
    weights_file = os.environ.get("FL_WEIGHTS_FILE")
    output_file = os.environ.get("FL_OUTPUT_FILE")
    client_id = os.environ.get("FL_CLIENT_ID", "local")
    num_epochs = int(os.environ.get("FL_NUM_EPOCHS", "5"))

    fl_mode = weights_file is not None and output_file is not None

    # Load global weights if in FL mode
    global_weights = None
    if fl_mode and os.path.exists(weights_file):
        global_weights = torch.load(weights_file)
        if not global_weights:  # Empty dict means no weights
            global_weights = None

    # Run training
    final_weights, final_loss, rank = train(
        global_weights=global_weights,
        num_epochs=num_epochs,
        client_id=client_id,
    )

    # Save results if in FL mode (only rank 0)
    if fl_mode and rank == 0:
        torch.save({"weights": final_weights, "loss": final_loss}, output_file)


if __name__ == "__main__":
    main()


# To run standalone:
#   torchrun --nproc_per_node=2 distributed_train.py
#
# This runs 2 processes on the same node, each simulating a GPU.
# In production, you would have actual GPUs and use NCCL backend.
