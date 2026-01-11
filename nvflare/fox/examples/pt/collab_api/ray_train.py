import torch
import torch.nn as nn
import torch.optim as optim
from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
from ray import train
from torch.utils.data import DataLoader, TensorDataset
# Import ray data if using Ray Data, otherwise keep torch utils
import ray.data

# 1. Define Model (same as above)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 2. Wrap the training logic in a function
def train_loop_per_worker():
    # Setup data
    inputs = torch.randn(100, 10)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(inputs, labels)
    # Use prepare_data_loader to automatically shard data across workers and place on correct device
    dataloader = prepare_data_loader(DataLoader(dataset, batch_size=10))

    # Setup model and optimizer
    model = SimpleModel()
    # Use prepare_model to wrap the model in DDP and place on the correct device
    model = prepare_model(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(5):
        for batch in dataloader:
            # Data is automatically moved to GPU/CPU by prepare_data_loader
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Report metrics back to Ray Train for tracking/checkpointing
        train.report({"loss": loss.item(), "epoch": epoch+1})

# 3. Execute the distributed training
if __name__ == "__main__":
    # Configure scaling (e.g., 2 workers, use GPUs if available)
    scaling_config = train.ScalingConfig(num_workers=2, use_gpu=False)

    # Initialize the TorchTrainer with the training function and config
    trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=scaling_config,
    )

    # Launch the training job
    result = trainer.fit()
    print(f"Training result: {result}")
