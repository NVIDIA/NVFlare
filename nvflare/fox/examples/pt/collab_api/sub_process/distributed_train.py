import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# 1. Define Model, Data, and Training Function
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def train():
    # Initialize distributed process group
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Setup data with DistributedSampler
    inputs = torch.randn(100, 10)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(inputs, labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # Setup model with DDP
    model = SimpleModel()
    model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    loss = None  # Initialize for linter

    # Training loop
    for epoch in range(5):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            batch_inputs, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Cleanup
    dist.destroy_process_group()


# 2. Execute the training
if __name__ == "__main__":
    train()


# To run:
# torchrun --nproc_per_node=2 distributed_train.py
