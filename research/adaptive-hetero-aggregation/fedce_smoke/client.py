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

"""Small deterministic client validating NVIDIA's real FedCE protocol."""

import argparse
import json
import math
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_opt.pt.fedce import PTFedCEHelper

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
    labels = torch.multinomial(_class_probabilities(site_index), count, replacement=True, generator=generator)
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


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(DEVICE))
            labels = labels.to(DEVICE)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return correct / total


def _model_difference(trained_state: dict, initial_state: dict) -> dict:
    return {name: trained_state[name].detach().cpu() - initial_state[name].detach().cpu() for name in initial_state}


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

    previous_local_state = None
    minus_scores = {}

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        current_round = input_model.current_round
        model.load_state_dict(input_model.params, strict=True)
        initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        global_accuracy = _evaluate(model, valid_loader)
        contribution_weight = PTFedCEHelper.get_contribution_weight(input_model, site_name)
        if current_round == 0:
            minus_accuracy = None
            minus_scores[current_round] = 0.0
        else:
            if previous_local_state is None:
                raise RuntimeError("FedCE previous local state is unavailable after round 0")
            minus_model = PTFedCEHelper.make_minus_model(model, previous_local_state, contribution_weight)
            minus_accuracy = _evaluate(minus_model, valid_loader)
            minus_scores[current_round] = minus_accuracy

        model.train()
        for _ in range(args.local_epochs):
            for features, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(features.to(DEVICE)), labels.to(DEVICE))
                loss.backward()
                optimizer.step()

        trained_state = model.state_dict()
        model_diff = _model_difference(trained_state, initial_state)
        previous_local_state = {name: value.detach().cpu().clone() for name, value in trained_state.items()}
        historical_minus_score = 1.0 - float(np.mean([minus_scores[index] for index in range(current_round + 1)]))
        steps = args.local_epochs * math.ceil(len(train_dataset) / args.batch_size)

        result = FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={"accuracy": global_accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        PTFedCEHelper.set_minus_model_score(result, historical_minus_score)
        print(
            "FEDCE_SMOKE_METRIC "
            + json.dumps(
                {
                    "client": site_name,
                    "round": current_round,
                    "global_accuracy": global_accuracy,
                    "minus_accuracy": minus_accuracy,
                    "contribution_weight": contribution_weight,
                    "historical_minus_score": historical_minus_score,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        flare.send(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=180)
    parser.add_argument("--valid_samples", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260906)
    main(parser.parse_args())
