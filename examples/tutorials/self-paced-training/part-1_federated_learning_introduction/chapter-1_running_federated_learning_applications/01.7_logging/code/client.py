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

"""Train and evaluate the model for this lesson."""

import argparse
import logging

import torch
from data import SyntheticImageDataset, stable_seed
from model import create_model
from torch.utils.data import DataLoader

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel

logger = logging.getLogger(__name__)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def training_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=positive_int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--local_epochs", type=positive_int, default=1)
    return parser


def make_loaders(site_name, batch_size):
    # Keep this CPU exercise small and avoid oversubscribing a laptop with two clients.
    torch.set_num_threads(1)
    train_loader = DataLoader(
        SyntheticImageDataset(site_name, "train", 200),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(stable_seed(site_name, "shuffle")),
        num_workers=0,
    )
    eval_loader = DataLoader(SyntheticImageDataset(site_name, "eval", 100), batch_size=batch_size, num_workers=0)
    return train_loader, eval_loader


def train(model, loader, epochs, learning_rate):
    # This lesson deliberately resets SGD state at each round.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    steps = 0
    total_loss = 0.0
    examples_seen = 0
    for _ in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            steps += 1
            total_loss += loss.item() * len(labels)
            examples_seen += len(labels)
    return steps, total_loss / examples_seen


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
    return 100.0 * correct / total


def copy_weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def main():
    args = training_parser().parse_args()
    flare.init()
    model = create_model()
    site = flare.get_site_name()
    train_loader, eval_loader = make_loaders(site, args.batch_size)
    logger.info(
        f"{site}: batch_size={args.batch_size}, learning_rate={args.learning_rate}, local_epochs={args.local_epochs}"
    )

    while flare.is_running():
        incoming = flare.receive()
        # The configured workflows issue train and evaluate, not submit_model.
        if not (flare.is_train() or flare.is_evaluate()):
            raise RuntimeError("This client supports train and evaluate tasks")
        model.load_state_dict(incoming.params)
        accuracy = evaluate(model, eval_loader)
        if flare.is_evaluate():
            flare.send(FLModel(metrics={"accuracy": accuracy}, meta={"num_eval_samples": len(eval_loader.dataset)}))
            continue

        steps, loss = train(model, train_loader, args.local_epochs, args.learning_rate)
        local_accuracy = evaluate(model, eval_loader)
        logger.info(
            f"Round {incoming.current_round}: received accuracy={accuracy:.1f}%, local accuracy={local_accuracy:.1f}%"
        )
        flare.send(
            FLModel(
                params=copy_weights(model),
                metrics={"accuracy": accuracy},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )
        )


if __name__ == "__main__":
    main()
