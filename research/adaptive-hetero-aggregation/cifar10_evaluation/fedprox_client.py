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

"""Matched CIFAR-10 FedProx client with held-out training validation data."""

import argparse
import copy
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
from nvflare.app_common.utils.fedprox_utils import get_fedprox_mu
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from nvflare.client.tracking import SummaryWriter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _site_seed(base_seed: int, site_name: str) -> int:
    match = re.search(r"(\d+)$", site_name)
    if not match:
        raise ValueError(f"expected site name ending in an integer, got {site_name!r}")
    return base_seed + int(match.group(1)) - 1


def _seed_client(base_seed: int, site_name: str):
    seed = _site_seed(base_seed, site_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main(args):
    flare.init()
    site_name = flare.get_site_name()
    _seed_client(args.seed, site_name)

    model = ModerateCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = None

    train_dataset, validation_dataset = create_local_datasets(
        site_name,
        train_idx_root=args.train_idx_root,
        validation_idx_root=args.validation_idx_root,
    )
    train_loader, valid_loader = create_data_loaders(
        train_dataset, validation_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    if len(train_loader) == 0 or len(valid_loader) == 0:
        raise ValueError("FedProx CIFAR-10 client requires non-empty training and validation loaders")

    summary_writer = SummaryWriter()
    last_trained_params = None
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        if flare.is_evaluate():
            model.load_state_dict(input_model.params, strict=True)
            model.to(DEVICE)
            accuracy = evaluate(model, valid_loader)
            flare.send(flare.FLModel(metrics={"accuracy": accuracy}))
            continue

        if flare.is_submit_model():
            if last_trained_params is None:
                raise RuntimeError("Cannot submit a local model before completing a training round")
            flare.send(flare.FLModel(params=last_trained_params, params_type=ParamsType.FULL))
            continue

        if not flare.is_train():
            raise RuntimeError(f"Unsupported task: {flare.get_task_name()}")

        fedprox_mu = get_fedprox_mu(input_model)
        criterion_prox = PTFedProxLoss(mu=fedprox_mu)
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
        summary_writer.add_scalar(
            "val_acc_global_model", val_acc_global_model, global_step=input_model.current_round
        )
        steps = args.aggregation_epochs * len(train_loader)

        for epoch in range(args.aggregation_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels) + criterion_prox(model, global_model)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            global_epoch = input_model.current_round * args.aggregation_epochs + epoch
            summary_writer.add_scalar("global_round", input_model.current_round, global_step=global_epoch)
            summary_writer.add_scalar("global_epoch", global_epoch, global_step=global_epoch)
            summary_writer.add_scalar("train_loss", running_loss / len(train_loader), global_step=global_epoch)
            summary_writer.add_scalar("learning_rate", get_lr_values(optimizer)[0], global_step=global_epoch)
            if scheduler is not None:
                scheduler.step()

        model_diff, diff_norm = compute_model_diff(model, global_model)
        last_trained_params = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        summary_writer.add_scalar("diff_norm", diff_norm.item(), global_step=input_model.current_round)
        flare.send(
            flare.FLModel(
                params=model_diff,
                params_type=ParamsType.DIFF,
                metrics={"accuracy": val_acc_global_model},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_idx_root", required=True)
    parser.add_argument("--validation_idx_root", required=True)
    parser.add_argument("--aggregation_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--cosine_lr_eta_min_factor", type=float, default=0.01)
    main(parser.parse_args())
