import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import CIFAR10CNN

logger = logging.getLogger(__name__)


def get_dataloaders(site_id: str, batch_size: int = 128):
    data_root = Path(__file__).resolve().parent / "data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=transform,
    )

    # For the initial NVFlare example, both clients use the same dataset.
    # The site_id is retained so the example can later be extended to
    # client-specific/non-IID partitions.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def train_one_round(
    model,
    train_loader,
    device,
    epochs=1,
    learning_rate=0.001,
):
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
    )

    for epoch in range(epochs):
        if len(train_loader) == 0:
            raise ValueError("Training data loader is empty")

        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        logger.info(
            "Epoch %d/%d - loss: %.4f",
            epoch + 1,
            epochs,
            avg_loss,
        )


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if total == 0:
        raise ValueError("Test data loader is empty")

    accuracy = 100.0 * correct / total

    logger.info(
        "Test accuracy: %.2f%% (%d/%d)",
        accuracy,
        correct,
        total,
    )

    return accuracy


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    site_id = os.environ.get("NVFLARE_SITE_ID", "site-1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Site: %s", site_id)
    logger.info("Device: %s", device)

    train_loader, test_loader = get_dataloaders(site_id)

    model = CIFAR10CNN().to(device)

    train_one_round(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=1,
        learning_rate=0.001,
    )

    evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    return model


if __name__ == "__main__":
    main()
