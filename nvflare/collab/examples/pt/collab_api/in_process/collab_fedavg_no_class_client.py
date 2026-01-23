"""Client-side training logic for Federated Averaging.

This module contains only the @fox.collab decorated functions
that run on client sites.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nvflare.collab import fox


class SimpleModel(nn.Module):
    """Simple linear model for demonstration."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


@fox.collab
def train(weights=None):
    """Train a local model - runs on each client site."""
    # Setup data (in real case, load from local dataset)
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
    loss = None

    # Training loop
    for epoch in range(5):
        for batch in dataloader:
            batch_inputs, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    print(f"  [{fox.site_name}] Loss: {loss.item():.4f}")

    # Return updated weights and loss
    return model.state_dict(), loss.item()
