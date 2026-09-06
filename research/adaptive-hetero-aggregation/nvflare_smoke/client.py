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

"""Small deterministic CPU client for end-to-end NVFlare smoke validation."""

import argparse
import math
import re

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType

NUM_FEATURES = 6
NUM_CLASSES = 3
DEVICE = torch.device("cpu")


def _site_index(site_name: str) -> int:
    match = re.search(r"(\d+)$", site_name)
    if not match:
        raise ValueError(f"expected simulator site name ending in an integer, got {site_name!r}")
    return int(match.group(1)) - 1


def _class_probabilities(site_index: int) -> torch.Tensor:
    peak = site_index % NUM_CLASSES
    probabilities = torch.full((NUM_CLASSES,), 0.10, dtype=torch.float32)
    probabilities[peak] = 0.70
    probabilities[(peak + 1) % NUM_CLASSES] = 0.20
    return probabilities


def _make_dataset(site_index: int, count: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed + site_index * 97)
    probabilities = _class_probabilities(site_index)
    labels = torch.multinomial(probabilities, count, replacement=True, generator=generator)
    class_means = torch.tensor(
        [
            [2.0, 0.0, 0.0, 1.0, -0.5, 0.5],
            [0.0, 2.0, 0.0, -0.5, 1.0, 0.5],
            [0.0, 0.0, 2.0, 0.5, -0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    direction = torch.tensor([0.4, -0.3, 0.2, 0.1, -0.2, 0.3], dtype=torch.float32)
    covariate_shift = float(site_index - 1) * 0.35 * direction
    features = class_means[labels] + covariate_shift
    features = features + 0.85 * torch.randn(count, NUM_FEATURES, generator=generator)
    return TensorDataset(features, labels)


def _evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(DEVICE))
            labels = labels.to(DEVICE)
            total_loss += float(criterion(logits, labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return total_loss / total, correct / total


def main(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    model = nn.Linear(NUM_FEATURES, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    flare.init()
    site_name = flare.get_site_name()
    site_index = _site_index(site_name)
    train_dataset = _make_dataset(site_index, args.train_samples, args.seed)
    valid_dataset = _make_dataset(site_index, args.valid_samples, args.seed + 10000)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    labels = train_dataset.tensors[1]
    descriptor = torch.bincount(labels, minlength=NUM_CLASSES).to(dtype=torch.float64).tolist()

    while flare.is_running():
        input_model = flare.receive()
        model.load_state_dict(input_model.params, strict=True)
        global_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        before_loss, before_accuracy = _evaluate(model, valid_loader)
        model.train()
        for _ in range(args.local_epochs):
            for features, target in train_loader:
                optimizer.zero_grad()
                logits = model(features.to(DEVICE))
                loss = criterion(logits, target.to(DEVICE))
                loss.backward()
                optimizer.step()

        after_loss, after_accuracy = _evaluate(model, valid_loader)
        local_state = model.state_dict()
        model_diff = {name: local_state[name].detach().cpu() - global_state[name] for name in global_state}
        steps = args.local_epochs * math.ceil(len(train_dataset) / args.batch_size)

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={"accuracy": before_accuracy, "local_post_accuracy": after_accuracy},
            meta={
                "NUM_STEPS_CURRENT_ROUND": steps,
                "adaptive_distribution_descriptor": descriptor,
                "adaptive_client_metric": before_accuracy,
                "adaptive_quality_improvement": before_loss - after_loss,
            },
        )
        flare.send(output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=240)
    parser.add_argument("--valid_samples", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260906)
    main(parser.parse_args())
