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
import torch
import torch.optim as optim

from typing import Tuple

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
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric

from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.signal import Signal
from nvflare.apis.fl_context import FLContext

from pt.trainers.supervised_trainer import SupervisedTrainer


def custom_client_datalist_json_path(datalist_json_path: str, client_id: str, prefix: str) -> str:
    """
    Customize datalist_json_path for each client
    Args:
         datalist_json_path: default datalist_json_path
         client_id: e.g., site-2
    """
    # Customize datalist_json_path for each client
    # - client_id: e.g. site-5
    head, tail = os.path.split(datalist_json_path)
    datalist_json_path = os.path.join(
        head,
        prefix + "_" + str(client_id) + ".json",
    )
    return datalist_json_path


class BratsTrainer(SupervisedTrainer):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        """Simple Brats Trainer. It inherits from Supervised trainer.

        Args:
            train_config_filename: directory of config file.
            aggregation_epochs: the number of training epochs for a round.
                This parameter only works when `aggregation_iters` is 0. Defaults to 1.
            train_task_name: name of the task to train the model.
            submit_model_task_name: name of the task to submit the best local model.

        Returns:
            a Shareable with the updated local model after running `execute()`
            or the best local model depending on the specified task.
        """
        super().__init__(
            train_config_filename=train_config_filename,
            aggregation_epochs=aggregation_epochs,
            train_task_name=train_task_name,
            submit_model_task_name=submit_model_task_name,
        )

    def _extra_train_config(self, fl_ctx: FLContext, config_info: dict):
        # Get the config_info
        self.lr = config_info["learning_rate"]
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = DiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
        )  # criterion for training

        self.transform_train = Compose(
            [
                # load Nifti image
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                RandSpatialCropd(keys=["image", "label"], roi_size=self.roi_size, random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
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

        # Set dataset
        self.train_dataset = Dataset(
            data=train_list,
            transform=self.transform_train,
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

    # Use a custom `local_valid` routine. It is possible to use other non-monai validation metrics.
    def local_valid(self, val_loader, tb_id, abort_signal: Signal):
        """
        Return a scalar as the validation metric that is used for best model selection during training
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
            metric = self.valid_metric.aggregate().item()
            self.valid_metric.reset()
        self.writer.add_scalar(tb_id, metric, self.epoch_of_start_time)
        print(f"valid_dice: {metric:.4f}")
        return metric
