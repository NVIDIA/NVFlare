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

"""FedCE CIFAR-10 client using the same training protocol as the matched benchmark."""

import argparse
import copy
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.cifar10_data_utils import create_data_loaders, create_datasets
from model import ModerateCNN
from torch.utils.data import Subset
from train_utils import compute_model_diff, evaluate, get_lr_values

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_opt.pt.fedce import PTFedCEHelper
from nvflare.client.tracking import SummaryWriter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _site_seed(base_seed: int, site_name: str) -> int:
    match = re.search(r"(\d+)$", site_name)
    if not match:
        raise ValueError(f"expected site name ending in an integer, got {site_name!r}")
    return base_seed + int(match.group(1)) - 1


def _local_validation_dataset(valid_dataset, eval_idx_root: str, site_name: str):
    path = os.path.join(eval_idx_root, f"{site_name}.npy")
    if not os.path.isfile(path):
        raise ValueError(f"missing local evaluation split for {site_name}: {path}")
    indices = np.load(path).astype(np.int64)
    if indices.size == 0:
        raise ValueError(f"local evaluation split for {site_name} is empty")
    return Subset(valid_dataset, indices.tolist())


def main(args):
    flare.init()
    client_name = flare.get_site_name()
    seed = _site_seed(args.seed, client_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = ModerateCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = None

    train_dataset, full_valid_dataset = create_datasets(client_name, train_idx_root=args.train_idx_root)
    local_valid_dataset = _local_validation_dataset(full_valid_dataset, args.eval_idx_root, client_name)
    train_loader, valid_loader = create_data_loaders(
        train_dataset, local_valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    if len(train_loader) == 0 or len(valid_loader) == 0:
        raise ValueError("FedCE CIFAR-10 evaluation requires non-empty training and validation loaders")

    writer = SummaryWriter()
    previous_local_state = None
    minus_scores = {}

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        current_round = input_model.current_round
        if scheduler is None and not args.no_lr_scheduler:
            eta_min = args.lr * args.cosine_lr_eta_min_factor
            t_max = input_model.total_rounds * args.aggregation_epochs
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

        model.load_state_dict(input_model.params, strict=True)
        initial_model = copy.deepcopy(model)
        for parameter in initial_model.parameters():
            parameter.requires_grad = False
        model.to(DEVICE)
        initial_model.to(DEVICE)

        global_accuracy = evaluate(initial_model, valid_loader)
        writer.add_scalar("val_acc_global_model", global_accuracy, global_step=current_round)
        contribution_weight = PTFedCEHelper.get_contribution_weight(input_model, client_name)

        minus_accuracy = None
        if current_round == 0:
            minus_scores[current_round] = 0.0
        else:
            if previous_local_state is None:
                raise RuntimeError("FedCE previous local state is unavailable after round 0")
            minus_model = PTFedCEHelper.make_minus_model(initial_model, previous_local_state, contribution_weight)
            minus_model.to(DEVICE)
            minus_accuracy = evaluate(minus_model, valid_loader)
            writer.add_scalar("val_acc_minus_model", minus_accuracy, global_step=current_round)
            minus_scores[current_round] = minus_accuracy
            del minus_model
        writer.add_scalar("FedCE_Coef", contribution_weight, global_step=current_round)

        steps = args.aggregation_epochs * len(train_loader)
        for epoch in range(args.aggregation_epochs):
            model.train()
            running_loss = 0.0
            for data in train_loader:
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            global_epoch = current_round * args.aggregation_epochs + epoch
            writer.add_scalar("global_round", current_round, global_step=global_epoch)
            writer.add_scalar("global_epoch", global_epoch, global_step=global_epoch)
            writer.add_scalar("train_loss", running_loss / len(train_loader), global_step=global_epoch)
            writer.add_scalar("learning_rate", get_lr_values(optimizer)[0], global_step=global_epoch)
            if scheduler is not None:
                scheduler.step()

        model_diff, diff_norm = compute_model_diff(model, initial_model)
        writer.add_scalar("diff_norm", diff_norm.item(), global_step=current_round)
        previous_local_state = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        historical_minus_score = 1.0 - float(np.mean([minus_scores[i] for i in range(current_round + 1)]))

        result = FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={"accuracy": global_accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        PTFedCEHelper.set_minus_model_score(result, historical_minus_score)
        print(
            "FEDCE_CIFAR_METRIC "
            + json.dumps(
                {
                    "client": client_name,
                    "round": current_round,
                    "local_validation_accuracy": global_accuracy,
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
    parser.add_argument("--train_idx_root", required=True)
    parser.add_argument("--eval_idx_root", required=True)
    parser.add_argument("--aggregation_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--cosine_lr_eta_min_factor", type=float, default=0.01)
    main(parser.parse_args())
