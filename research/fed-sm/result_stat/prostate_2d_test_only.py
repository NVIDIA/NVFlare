# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.inferers import SimpleInferer
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
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
from vgg import vgg11

model_postfix = "_best_FL_global_model.pt"
client_id_labels = ["client_I2CVB", "client_MSD", "client_NCI_ISBI_3T"]


def main():
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("--models_dir", type=str)
    parser.add_argument("--cache_rate", default=0.0, type=float)
    parser.add_argument("--select_threshold", default=0.9, type=float)
    parser.add_argument("--dataset_base_dir", default="DATASET_ROOT/dataset_2D", type=str)
    parser.add_argument(
        "--datalist_json_path",
        default="DATASET_ROOT/datalist_2D/client_All.json",
        type=str,
    )
    args = parser.parse_args()

    # Set basic settings and paths
    dataset_base_dir = args.dataset_base_dir
    datalist_json_path = args.datalist_json_path
    models_dir = args.models_dir
    cache_rate = args.cache_rate
    select_threshold = args.select_threshold
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set datalists
    test_list = load_decathlon_datalist(
        data_list_file_path=datalist_json_path,
        is_segmentation=True,
        data_list_key="testing",
        base_dir=dataset_base_dir,
    )
    print(f"Testing Size: {len(test_list)}")

    # Network, optimizer, and loss
    num_site = len(client_id_labels)
    model_select = vgg11(num_classes=num_site).to(device)
    model_global = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model_person = []
    for site in range(num_site):
        model_person.append(
            UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(device)
        )

    model_path = models_dir + "global_weights" + model_postfix
    model_stat_dict = torch.load(model_path)
    for var_name in model_stat_dict:
        model_stat_dict[var_name] = torch.as_tensor(model_stat_dict[var_name])
    model_global.load_state_dict(model_stat_dict)

    model_global.eval()
    model_path = models_dir + "select_weights" + model_postfix
    model_stat_dict = torch.load(model_path)
    for var_name in model_stat_dict:
        model_stat_dict[var_name] = torch.as_tensor(model_stat_dict[var_name])
    model_select.load_state_dict(model_stat_dict)

    model_select.eval()
    for site in range(num_site):
        model_path = models_dir + client_id_labels[site] + model_postfix
        model_stat_dict = torch.load(model_path)
        for var_name in model_stat_dict:
            model_stat_dict[var_name] = torch.as_tensor(model_stat_dict[var_name])
        model_person[site].load_state_dict(model_stat_dict)
        model_person[site].eval()

    # Inferer, evaluation metric
    inferer_select = SimpleInferer()
    inferer_segment = SimpleInferer()
    valid_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image", "label"], a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            Resized(
                keys=["image", "label"],
                spatial_size=(256, 256),
                mode=("bilinear"),
                align_corners=True,
            ),
            AsDiscreted(keys=["label"], threshold=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    transform_post = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Set dataset
    test_dataset = CacheDataset(
        data=test_list,
        transform=transform,
        cache_rate=cache_rate,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    model_select.eval()
    with torch.no_grad():
        metric = 0
        for i, batch_data in enumerate(test_loader):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            # Inference
            # get the selector result
            outputs_select = inferer_select(images, model_select)
            score = torch.nn.functional.softmax(outputs_select).cpu().numpy()
            score = np.squeeze(score)
            max_index = np.argmax(score)
            max_score = score[max_index]
            # get max score and determine which model to use
            if max_score > select_threshold:
                model_segment = model_person[max_index]
            else:
                model_segment = model_global
            # segmentation inference
            outputs_segment = inferer_segment(images, model_segment)
            outputs_segment = transform_post(outputs_segment)
            # Compute metric
            metric_score = valid_metric(y_pred=outputs_segment, y=labels)
            metric += metric_score.item()
        # compute mean dice over whole validation set
        metric /= len(test_loader)
        print(f"Test Dice: {metric:.4f}")


if __name__ == "__main__":
    main()
