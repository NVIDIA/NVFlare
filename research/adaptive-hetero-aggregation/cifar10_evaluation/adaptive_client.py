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

"""CIFAR-10 client for the adaptive aggregation evaluation.

The training loop follows NVIDIA FLARE's standard PyTorch CIFAR-10 FedAvg
example. Adaptive performance metadata is computed only on a deterministic
site-local validation subset held out from CIFAR-10's training set. The official
CIFAR-10 test set is reserved for post-training evaluation.
"""

import argparse
import copy
import json
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data.cifar10_data_utils import create_data_loaders
from local_data import create_local_datasets
from model import ModerateCNN
from train_utils import compute_model_diff, evaluate, get_lr_values

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.client.tracking import SummaryWriter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


def _site_seed(base_seed: int, site_name: str) -> int:
    match = re.search(r"(\d+)$", site_name)
    if not match:
        raise ValueError(f"expected site name ending in an integer, got {site_name!r}")
    return base_seed + int(match.group(1)) - 1


def _class_descriptor(train_dataset) -> list[float]:
    targets = np.asarray(train_dataset.target, dtype=np.int64)
    if targets.size == 0:
        raise ValueError("training dataset must contain at least one example")
    return np.bincount(targets, minlength=NUM_CLASSES).astype(np.float64).tolist()


def main(args):
    flare.init()
    site_name = flare.get_site_name()
    seed = _site_seed(args.seed, site_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = ModerateCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = None

    print(f"Create datasets for site {site_name}")
    train_dataset, validation_dataset = create_local_datasets(
        site_name,
        train_idx_root=args.train_idx_root,
        validation_idx_root=args.validation_idx_root,
    )
    train_loader, valid_loader = create_data_loaders(
        train_dataset, validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    if len(train_loader) == 0 or len(valid_loader) == 0:
        raise ValueError("CIFAR-10 evaluation requires non-empty training and validation loaders")

    descriptor = _class_descriptor(train_dataset)
    sample_count = len(train_dataset)
    validation_count = len(validation_dataset)
    summary_writer = SummaryWriter()

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        current_round = input_model.current_round
        print(f"\n[Current Round={current_round}, Site={site_name}]\n")

        if scheduler is None and not args.no_lr_scheduler:
            eta_min = args.lr * args.cosine_lr_eta_min_factor
            t_max = input_model.total_rounds * args.aggregation_epochs
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

        model.load_state_dict(input_model.params, strict=True)
        global_model = copy.deepcopy(model)
        for parameter in global_model.parameters():
            parameter.requires_grad = False
        model.to(DEVICE)
        global_model.to(DEVICE)

        val_acc_global_model = evaluate(global_model, valid_loader)
        summary_writer.add_scalar("val_acc_global_model", val_acc_global_model, global_step=current_round)
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
            summary_writer.add_scalar("global_round", current_round, global_step=global_epoch)
            summary_writer.add_scalar("global_epoch", global_epoch, global_step=global_epoch)
            summary_writer.add_scalar("train_loss", running_loss / len(train_loader), global_step=global_epoch)
            summary_writer.add_scalar("learning_rate", get_lr_values(optimizer)[0], global_step=global_epoch)
            if scheduler is not None:
                scheduler.step()

        model_diff, diff_norm = compute_model_diff(model, global_model)
        summary_writer.add_scalar("diff_norm", diff_norm.item(), global_step=current_round)

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={"accuracy": val_acc_global_model},
            meta={
                "NUM_STEPS_CURRENT_ROUND": steps,
                "adaptive_sample_count": sample_count,
                "adaptive_distribution_descriptor": descriptor,
                "adaptive_client_metric": val_acc_global_model,
            },
        )
        print(
            "ADAPTIVE_CIFAR_METRIC "
            + json.dumps(
                {
                    "client": site_name,
                    "round": current_round,
                    "local_validation_accuracy": val_acc_global_model,
                    "sample_count": sample_count,
                    "validation_count": validation_count,
                    "steps": steps,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        flare.send(output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_idx_root", required=True)
    parser.add_argument("--validation_idx_root", required=True)
    parser.add_argument("--aggregation_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--cosine_lr_eta_min_factor", type=float, default=0.01)
    main(parser.parse_args())
