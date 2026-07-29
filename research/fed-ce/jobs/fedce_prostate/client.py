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
import json
from pathlib import Path

import numpy as np
import torch
from model import UNet
from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
from monai.inferers import SimpleInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    Resized,
    ScaleIntensityRanged,
)
from monai.utils import set_determinism

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_opt.pt.fedce import PTFedCEHelper
from nvflare.client.tracking import SummaryWriter

CLIENTS = [
    "client_I2CVB",
    "client_MSD",
    "client_NCI_ISBI_3T",
    "client_NCI_ISBI_Dx",
    "client_Promise12",
    "client_PROSTATEx",
]


def _build_loaders(args, client_name):
    datalist_path = Path(args.data_root) / "datalist_2D" / f"{client_name}.json"
    dataset_dir = Path(args.data_root) / "dataset_2D"
    if not datalist_path.is_file():
        raise ValueError(f"Missing client datalist: {datalist_path}")
    if not dataset_dir.is_dir():
        raise ValueError(f"Missing 2D dataset directory: {dataset_dir}")

    train_list = load_decathlon_datalist(
        data_list_file_path=str(datalist_path),
        is_segmentation=True,
        data_list_key="training",
        base_dir=str(dataset_dir),
    )
    valid_list = load_decathlon_datalist(
        data_list_file_path=str(datalist_path),
        is_segmentation=True,
        data_list_key="validation",
        base_dir=str(dataset_dir),
    )
    if not train_list or not valid_list:
        raise ValueError(
            f"{client_name} requires non-empty training and validation splits, "
            f"got training={len(train_list)}, validation={len(valid_list)}"
        )

    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image", "label"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
            ),
            Resized(
                keys=["image", "label"],
                spatial_size=(256, 256),
                mode="bilinear",
                align_corners=True,
            ),
            AsDiscreted(keys=["label"], threshold=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    dataset_type = CacheDataset if args.cache_rate > 0.0 else Dataset
    dataset_args = {"transform": transform}
    if dataset_type is CacheDataset:
        dataset_args.update(cache_rate=args.cache_rate, num_workers=args.num_workers)

    train_dataset = dataset_type(data=train_list, **dataset_args)
    valid_dataset = dataset_type(data=valid_list, **dataset_args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, valid_loader


def _validate(model, loader, device, post_transform):
    inferer = SimpleInferer()
    metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    score = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            predictions = post_transform(inferer(images, model))
            score += metric(y_pred=predictions, y=labels).item()
    return score / len(loader)


def _train(model, loader, optimizer, criterion, device, local_epochs, round_number, writer):
    steps_per_epoch = len(loader)
    for epoch in range(local_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        mean_loss = epoch_loss / steps_per_epoch
        global_epoch = round_number * local_epochs + epoch
        writer.add_scalar("train_loss", mean_loss, global_step=global_epoch)
        print(f"FedCE local epoch {epoch + 1}/{local_epochs}: loss={mean_loss:.6f}", flush=True)


def _model_difference(trained_state, initial_state):
    difference = {}
    for name, value in trained_state.items():
        initial_value = initial_state[name]
        if value.is_floating_point() or value.is_complex():
            difference[name] = value.detach().cpu() - initial_value.detach().cpu()
        else:
            difference[name] = torch.zeros_like(value, device="cpu")
    return difference


def main(args):
    flare.init()
    client_name = flare.get_site_name()
    if client_name not in CLIENTS:
        raise ValueError(f"Unexpected FedCE client {client_name!r}; expected one of {CLIENTS}")

    seed = args.seed + CLIENTS.index(client_name)
    set_determinism(seed=seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader = _build_loaders(args, client_name)
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = DiceLoss(sigmoid=True)
    post_transform = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    writer = SummaryWriter()
    previous_local_state = None
    minus_scores = {}

    print(
        f"FedCE client ready: site={client_name}, device={device}, "
        f"training={len(train_loader.dataset)}, validation={len(valid_loader.dataset)}",
        flush=True,
    )
    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break
        current_round = input_model.current_round
        model.load_state_dict(input_model.params)
        model.to(device)
        initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

        global_dice = _validate(model, valid_loader, device, post_transform)
        writer.add_scalar("val_metric_global_model", global_dice, global_step=current_round)

        contribution_weight = PTFedCEHelper.get_contribution_weight(input_model, client_name)
        if current_round == 0:
            minus_scores[current_round] = 0.0
            minus_dice = None
        else:
            if previous_local_state is None:
                raise RuntimeError("FedCE previous local state is unavailable after round 0")
            minus_model = PTFedCEHelper.make_minus_model(model, previous_local_state, contribution_weight)
            minus_dice = _validate(minus_model, valid_loader, device, post_transform)
            writer.add_scalar("val_metric_minus_model", minus_dice, global_step=current_round)
            minus_scores[current_round] = minus_dice
            del minus_model
        writer.add_scalar("FedCE_Coef", contribution_weight, global_step=current_round)

        _train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            local_epochs=args.local_epochs,
            round_number=current_round,
            writer=writer,
        )
        trained_state = model.state_dict()
        model_diff = _model_difference(trained_state, initial_state)
        previous_local_state = {name: value.detach().cpu().clone() for name, value in trained_state.items()}
        historical_minus_score = 1.0 - float(np.mean([minus_scores[i] for i in range(current_round + 1)]))

        result = FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={"dice": global_dice},
            meta={"NUM_STEPS_CURRENT_ROUND": args.local_epochs * len(train_loader)},
        )
        PTFedCEHelper.set_minus_model_score(result, historical_minus_score)
        record = {
            "client": client_name,
            "round": current_round,
            "global_dice": global_dice,
            "minus_dice": minus_dice,
            "contribution_weight": contribution_weight,
            "historical_minus_score": historical_minus_score,
        }
        print(f"FEDCE_METRIC {json.dumps(record, sort_keys=True)}", flush=True)
        flare.send(result)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-rate", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(_parse_args())
