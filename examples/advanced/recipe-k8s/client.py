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

import torch
from model import Cifar10Net
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

import nvflare.client as flare


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-index", type=int, required=True)
    parser.add_argument("--num-sites", type=int, required=True)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", default="/tmp/nvflare/cifar10")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def partition_dataset(dataset, site_index: int, num_sites: int, max_samples: int) -> Subset:
    indices = list(range(site_index, len(dataset), num_sites))
    if max_samples:
        indices = indices[:max_samples]
    return Subset(dataset, indices)


def select_device(require_gpu: bool) -> torch.device:
    if not require_gpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU was requested for this client, but CUDA is not available")
    return torch.device("cuda:0")


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            predictions = model(images.to(device)).argmax(dim=1)
            labels = labels.to(device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    local_epochs: int,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    total_loss = 0.0
    total_batches = 0

    for _ in range(local_epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def main() -> None:
    args = define_parser().parse_args()
    device = select_device(args.require_gpu)
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR10(root=args.data_dir, train=True, download=args.download, transform=transform)
    test_dataset = CIFAR10(root=args.data_dir, train=False, download=args.download, transform=transform)
    site_train_dataset = partition_dataset(train_dataset, args.site_index, args.num_sites, args.max_train_samples)
    site_test_dataset = partition_dataset(test_dataset, args.site_index, args.num_sites, args.max_test_samples)
    train_loader = DataLoader(
        site_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(site_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    flare.init()
    site_name = flare.system_info()["site_name"]
    print(f"site={site_name} device={device}", flush=True)
    model = Cifar10Net()

    while flare.is_running():
        input_model = flare.receive()
        model.load_state_dict(input_model.params)
        model.to(device)

        global_accuracy = evaluate(model, test_loader, device)
        loss = train(model, train_loader, device, args.local_epochs)
        local_accuracy = evaluate(model, test_loader, device)
        print(
            f"site={site_name} round={input_model.current_round} loss={loss:.4f} "
            f"global_accuracy={global_accuracy:.4f} local_accuracy={local_accuracy:.4f}",
            flush=True,
        )

        flare.send(
            flare.FLModel(
                params=model.cpu().state_dict(),
                params_type=flare.ParamsType.FULL,
                # Model selection evaluates the received global model, not the
                # local weights that will be aggregated after this round.
                metrics={"accuracy": global_accuracy},
                meta={"NUM_STEPS_CURRENT_ROUND": args.local_epochs * len(train_loader)},
                current_round=input_model.current_round,
            )
        )


if __name__ == "__main__":
    main()
