import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 1. Define Model, Data, and Training Function
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def train(weights=None):
    """Local training - assumes process group already initialized."""
    # Setup data (each process has its own random data)
    inputs = torch.randn(100, 10)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=10)

    # Load global model weights if provided
    model = SimpleModel()
    if weights is not None:
        model.load_state_dict(weights)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(5):
        for batch in dataloader:
            batch_inputs, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Return updated weights and loss
    return model.state_dict(), loss.item()


def weighted_avg(local_weights, world_size):
    """Distributed aggregation - uses existing process group."""
    avg_weights = {}
    for key in local_weights.keys():
        tensor = local_weights[key].clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        avg_weights[key] = tensor / world_size
    return avg_weights


def fed_avg(clients, num_rounds=5):
    """Entry point for each client process."""
    # Initialize process group ONCE at start (client-side initialization)
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Use provided clients list or generate based on world_size
    if len(clients) >= world_size:
        # Use first world_size clients from the list
        client_id = clients[rank]
    else:
        # Generate client_id if not enough clients provided
        client_id = f"site-{rank + 1}"

    if rank == 0:
        print(f"Starting Distributed FedAvg with {world_size} clients for {num_rounds} rounds")

    global_weights = None

    for round_num in range(num_rounds):
        if rank == 0:
            print(f"\n=== Round {round_num + 1} ===")

        # Each process (client) trains locally
        local_weights, loss = train(global_weights)
        print(f"  [{client_id}] Round {round_num + 1}, Loss: {loss:.4f}")

        # Synchronize before aggregation
        dist.barrier()

        # Aggregate weights across all processes using all_reduce
        global_weights = weighted_avg(local_weights, world_size)

        # Aggregate loss for reporting (only rank 0 prints)
        loss_tensor = torch.tensor([loss])
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_loss = loss_tensor.item() / world_size

        if rank == 0:
            print(f"  Global average loss: {global_loss:.4f}")

    # Cleanup process group ONCE at end
    dist.destroy_process_group()

    if rank == 0:
        print(f"\nDistributed FedAvg completed after {num_rounds} rounds")
        print("\nJob Status:", "Completed")

    return global_weights


# 2. Execute the training
if __name__ == "__main__":
    clients = ["site-1", "site-2", "site-3", "site-4", "site-5"]
    result = fed_avg(clients, num_rounds=5)


# To run:
# torchrun --nproc_per_node=5 sim_distributed_fedavg_train.py
