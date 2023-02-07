# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.networks.nets.segresnet import SegResNet
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
    Spacingd,
)


def main():
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_base_dir", default="../dataset_brats18/dataset", type=str)
    parser.add_argument("--datalist_json_path", default="../dataset_brats18/datalist/site-All.json", type=str)
    args = parser.parse_args()

    # Set basic settings and paths
    dataset_base_dir = args.dataset_base_dir
    datalist_json_path = args.datalist_json_path
    model_path = args.model_path
    infer_roi_size = (240, 240, 160)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set datalists
    test_list = load_decathlon_datalist(
        data_list_file_path=datalist_json_path,
        is_segmentation=True,
        data_list_key="validation",
        base_dir=dataset_base_dir,
    )
    print(f"Testing Size: {len(test_list)}")

    # Network, optimizer, and loss
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
    model_weights = torch.load(model_path)
    model_weights = model_weights["model"]
    model.load_state_dict(model_weights)

    # Inferer, evaluation metric
    inferer = SlidingWindowInferer(roi_size=infer_roi_size, sw_batch_size=1, overlap=0.5)
    valid_metric = DiceMetric(include_background=True, reduction="mean")

    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            DivisiblePadd(keys=["image", "label"], k=32),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    transform_post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Set dataset
    test_dataset = Dataset(data=test_list, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    # Train
    model.eval()
    with torch.no_grad():
        metric = 0
        metric_tc = 0
        metric_wt = 0
        metric_et = 0
        ct = 0
        ct_tc = 0
        ct_wt = 0
        ct_et = 0
        for i, batch_data in enumerate(test_loader):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            # Inference
            outputs = inferer(images, model)
            outputs = transform_post(outputs)
            # Compute metric
            metric_score = valid_metric(y_pred=outputs, y=labels)
            if not np.isnan(metric_score[0][0].item()):
                metric += metric_score[0][0].item()
                ct += 1
                metric_tc += metric_score[0][0].item()
                ct_tc += 1
            if not np.isnan(metric_score[0][1].item()):
                metric += metric_score[0][1].item()
                ct += 1
                metric_wt += metric_score[0][1].item()
                ct_wt += 1
            if not np.isnan(metric_score[0][2].item()):
                metric += metric_score[0][2].item()
                ct += 1
                metric_et += metric_score[0][2].item()
                ct_et += 1
        # compute mean dice over whole validation set
        metric_tc /= ct_tc
        metric_wt /= ct_wt
        metric_et /= ct_et
        metric /= ct
        print(f"Test Dice: {metric:.4f}, Valid count: {ct}")
        print(f"Test Dice TC: {metric_tc:.4f}, Valid count: {ct_tc}")
        print(f"Test Dice WT: {metric_wt:.4f}, Valid count: {ct_wt}")
        print(f"Test Dice ET: {metric_et:.4f}, Valid count: {ct_et}")


if __name__ == "__main__":
    main()
