# Copyright (c) 2021, NVIDIA CORPORATION.
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

import os
import json

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

import monai
from monai.data import (
    Dataset,
    DataLoader,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.losses import DiceLoss
from monai.networks.nets.unet import UNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    SplitChannel,
)
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

from pt.validators.supervised_validator import SupervisedValidator
from pt.trainers.brats_trainer import custom_client_datalist_json_path


class BratsValidator(SupervisedValidator):
    def __init__(
        self,
        train_config_filename,
        validate_task_name=AppConstants.TASK_VALIDATION,
    ):
        """Simple Brats Validator. It inherits from Supervised Validator.

        Args:
            train_config_filename: directory of config file.
            validate_task_name: name of the task to validate the model.

        Returns:
            a Shareable with the validation metrics.
        """
        super().__init__(
            train_config_filename=train_config_filename,
            validate_task_name=validate_task_name,
        )
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point

    def _validation_config(self, fl_ctx: FLContext, train_config_file_path: str):
        # Load training configurations
        with open(train_config_file_path) as file:
            config_info = json.load(file)
        # Get the config_info
        dataset_base_dir = config_info["dataset_base_dir"]
        datalist_json_path = config_info["datalist_json_path"]

        datalist_json_path = custom_client_datalist_json_path(
            datalist_json_path, self.client_id, prefix="config_brats18_datalist"
        )

        self.roi_size = config_info.get("roi_size", (224, 224, 144))
        self.infer_roi_size = config_info.get("infer_roi_size", (240, 240, 160))

        # Set datalist
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
        self.log_info(
            fl_ctx,
            f"Training Size: {len(train_list)}, Validation Size: {len(valid_list)}",
        )

        # Set the training-related context
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = monai.networks.nets.SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        ).to(self.device)
        self.model.to(self.device)

        self.transform_valid = Compose(
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
        self.transform_post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.transform_post_splitchannel = SplitChannel(channel_dim=1)

        # Set dataset
        self.train_dataset = Dataset(
            data=train_list,
            transform=self.transform_valid,
        )
        self.valid_dataset = Dataset(
            data=valid_list,
            transform=self.transform_valid,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

        # Set inferer and evaluation metric
        self.inferer = SlidingWindowInferer(roi_size=self.infer_roi_size, sw_batch_size=1, overlap=0.5)
        self.valid_metric = DiceMetric(include_background=True, reduction="mean")  # metrics for validation
        self.valid_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    def local_valid(self, val_loader, abort_signal: Signal) -> dict:
        """
        Return a Dictionary that may contain multiple validation metrics.
        """
        self.model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                if abort_signal.triggered:
                    return
                val_inputs, val_labels = (
                    val_data["image"].to(self.device),
                    val_data["label"].to(self.device),
                )
                val_outputs = self.inferer(val_inputs, self.model)
                val_outputs = [self.transform_post(i) for i in decollate_batch(val_outputs)]
                self.valid_metric(y_pred=val_outputs, y=val_labels)
                self.valid_metric_batch(y_pred=val_outputs, y=val_labels)

            # metric for all the 3 labels, and 3 labels separately
            metric = self.valid_metric.aggregate().item()
            metric_batch = self.valid_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_wt = metric_batch[1].item()
            metric_et = metric_batch[2].item()

            self.valid_metric.reset()
            self.valid_metric_batch.reset()

        metric_dict = {
            "dice": metric,
            "dice_tc": metric_tc,
            "dice_wt": metric_wt,
            "dice_et": metric_et,
        }
        for k in metric_dict.keys():
            print(f"valid {k}: {metric_dict[k]:.4f}")

        return metric_dict
