"""
CIFAR-10 model used by the FedSCS NVFlare research example.
"""

import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """Small CNN for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def create_model(num_classes: int = 10) -> CIFAR10CNN:
    """Create a new CIFAR-10 model."""
    return CIFAR10CNN(num_classes=num_classes)
