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

import argparse
import hashlib
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from e2e_net import E2ENet
from torch.utils.data import DataLoader, Subset

import nvflare.client as flare


def _site_index(site_name: str, num_clients: int) -> int:
    matches = re.findall(r"\d+", site_name)
    if matches:
        return (int(matches[-1]) - 1) % num_clients
    digest = hashlib.sha1(site_name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_clients


def _subset(dataset, site_index: int, num_clients: int, max_samples: int):
    indices = list(range(site_index, len(dataset), num_clients))[:max_samples]
    if not indices:
        raise RuntimeError("empty CIFAR10 subset; check num_clients and dataset contents")
    return Subset(dataset, indices)


def _evaluate(model, loader, criterion):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += float(loss.item()) * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += int((predicted == labels).sum().item())
    if total == 0:
        raise ValueError("evaluation data loader produced no samples; check the DataLoader and dataset subset")
    accuracy = correct / total
    avg_loss = loss_sum / total
    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    flare.init()
    site_name = flare.get_site_name()
    site_index = _site_index(site_name, args.num_clients)
    print(f"E2E_DATA site={site_name} root={args.data_root} download={args.download}", flush=True)

    train_data = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=args.download, transform=transform
    )
    val_data = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=args.download, transform=transform
    )
    train_subset = _subset(train_data, site_index, args.num_clients, args.max_train_samples)
    val_subset = _subset(val_data, site_index, args.num_clients, args.max_val_samples)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = E2ENet().to(device)
    criterion = nn.CrossEntropyLoss()

    while flare.is_running():
        input_model = flare.receive()
        current_round = input_model.current_round
        print(f"E2E_ROUND current_round={current_round} site={site_name}", flush=True)
        model.load_state_dict(input_model.params)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()
        steps = 0
        for _epoch in range(args.epochs):
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                steps += 1

        accuracy, val_loss = _evaluate(model, val_loader, criterion)
        print(
            f"E2E_METRIC current_round={current_round} site={site_name} "
            f"accuracy={accuracy:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )
        output_model = flare.FLModel(
            params={k: v.cpu() for k, v in model.state_dict().items()},
            metrics={"accuracy": accuracy, "val_loss": val_loss},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
