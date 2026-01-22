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
"""
Client-side training script for BraTS18 using NVFlare Client API.
"""
import argparse
import copy
import os
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
from model import create_brats_model
from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)

import nvflare.client as flare
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from nvflare.client.tracking import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS18 client training with NVFlare Client API.")
    parser.add_argument("--aggregation_epochs", type=int, default=1, help="Local epochs per round.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fedproxloss_mu", type=float, default=0.0)
    parser.add_argument("--cache_dataset", type=float, default=0.0)
    parser.add_argument("--dataset_base_dir", type=str, required=True)
    parser.add_argument("--datalist_json_path", type=str, required=True)
    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=(224, 224, 144),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--infer_roi_size",
        type=int,
        nargs=3,
        default=(240, 240, 160),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--centralized", action="store_true", help="Use all data for centralized training")
    return parser.parse_args()


def custom_client_datalist_json_path(datalist_json_path: str, client_id: str, centralized: bool = False) -> str:
    """Customize datalist_json_path for each client.

    Args:
        datalist_json_path: Root path containing all json files
        client_id: Client identifier (e.g., site-1, site-2, etc.)
        centralized: If True, use site-All.json for centralized training with all data

    Returns:
        Path to the appropriate datalist json file
    """
    if centralized:
        # Use site-All.json for centralized training with all data
        all_data_path = os.path.join(datalist_json_path, "site-All.json")
        if os.path.exists(all_data_path):
            return all_data_path
    return os.path.join(datalist_json_path, client_id + ".json")


def build_dataloaders(
    *,
    client_id: str,
    cache_rate: float,
    dataset_base_dir: str,
    datalist_json_path: str,
    roi_size: Sequence[int],
    infer_roi_size: Sequence[int],
    centralized: bool = False,
) -> Tuple[DataLoader, DataLoader, SlidingWindowInferer, Compose, DiceMetric]:
    datalist_json_path = custom_client_datalist_json_path(datalist_json_path, client_id, centralized)

    print(f"[{client_id}] Loading datalist from: {datalist_json_path}")

    train_list = load_decathlon_datalist(
        data_list_file_path=datalist_json_path,
        is_segmentation=True,
        data_list_key="training",
        base_dir=dataset_base_dir,
    )
    valid_list = load_decathlon_datalist(
        data_list_file_path=datalist_json_path,
        is_segmentation=True,
        data_list_key="validation",
        base_dir=dataset_base_dir,
    )

    print(f"[{client_id}] Training samples: {len(train_list)}, Validation samples: {len(valid_list)}")

    transform_train = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    transform_valid = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            DivisiblePadd(keys=["image", "label"], k=32),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    if cache_rate > 0.0:
        train_dataset = CacheDataset(data=train_list, transform=transform_train, cache_rate=cache_rate, num_workers=1)
        valid_dataset = CacheDataset(data=valid_list, transform=transform_valid, cache_rate=cache_rate, num_workers=1)
    else:
        train_dataset = Dataset(data=train_list, transform=transform_train)
        valid_dataset = Dataset(data=valid_list, transform=transform_valid)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    inferer = SlidingWindowInferer(roi_size=infer_roi_size, sw_batch_size=1, overlap=0.5)
    transform_post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    valid_metric = DiceMetric(include_background=True, reduction="mean")

    return train_loader, valid_loader, inferer, transform_post, valid_metric


def validate(model, valid_loader, inferer, transform_post, valid_metric, device):
    model.eval()
    with torch.no_grad():
        metric = 0.0
        ct = 0
        for batch_data in valid_loader:
            val_images = batch_data["image"].to(device)
            val_labels = batch_data["label"].to(device)
            val_outputs = inferer(val_images, model)
            val_outputs = transform_post(val_outputs)
            metric_score = valid_metric(y_pred=val_outputs, y=val_labels)
            for sub_region in range(3):
                metric_score_single = metric_score[0][sub_region].item()
                if not np.isnan(metric_score_single):
                    metric += metric_score_single
                    ct += 1
        if ct == 0:
            return 0.0
        return metric / ct


def main():
    args = parse_args()

    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    summary_writer = SummaryWriter()

    train_loader, valid_loader, inferer, transform_post, valid_metric = build_dataloaders(
        client_id=client_name,
        cache_rate=args.cache_dataset,
        dataset_base_dir=args.dataset_base_dir,
        datalist_json_path=args.datalist_json_path,
        roi_size=args.roi_size,
        infer_roi_size=args.infer_roi_size,
        centralized=args.centralized,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_brats_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    criterion_prox = PTFedProxLoss(mu=args.fedproxloss_mu) if args.fedproxloss_mu > 0 else None

    while flare.is_running():
        input_model = flare.receive()
        model.load_state_dict(input_model.params, strict=True)
        model.to(device)

        global_metric = validate(model, valid_loader, inferer, transform_post, valid_metric, device)
        summary_writer.add_scalar("val_metric_global_model", global_metric, input_model.current_round)

        model_global = None
        if args.fedproxloss_mu > 0:
            model_global = copy.deepcopy(model)
            for param in model_global.parameters():
                param.requires_grad = False

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.aggregation_epochs

        for epoch in range(args.aggregation_epochs):
            model.train()
            running_loss = 0.0
            for batch_data in train_loader:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if args.fedproxloss_mu > 0:
                    loss += criterion_prox(model, model_global)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if len(train_loader) == 0:
                raise ValueError("Training data loader is empty. Check dataset preparation and datalist configuration.")
            avg_loss = running_loss / len(train_loader)
            global_step = input_model.current_round * total_steps + epoch
            summary_writer.add_scalar("train_loss", avg_loss, global_step)

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"val_dice": global_metric},
            meta={"NUM_STEPS_CURRENT_ROUND": total_steps},
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
