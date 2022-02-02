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


import torch
import torch.optim as optim
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets.unet import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)
from pt.learners.supervised_learner import SupervisedLearner
from pt.utils.custom_client_datalist_json_path import custom_client_datalist_json_path

from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fedproxloss import PTFedProxLoss


class ProstateLearner(SupervisedLearner):
    def __init__(
        self,
        train_config_filename,
        aggregation_epochs: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        """Trainer for prostate segmentation task. It inherits from MONAI trainer.

        Args:
            train_config_filename: directory of config file.
            aggregation_epochs: the number of training epochs for a round.
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

    def _extra_train_config(self, fl_ctx: FLContext, config_info: str):
        # Get the config_info
        self.lr = config_info["learning_rate"]
        self.fedproxloss_mu = config_info["fedproxloss_mu"]
        self.roi_size = config_info.get("roi_size", (224, 224, 32))
        self.infer_roi_size = config_info.get("infer_roi_size", (224, 224, 32))
        dataset_base_dir = config_info["dataset_base_dir"]
        datalist_json_path = config_info["datalist_json_path"]

        # Get datalist json
        datalist_json_path = custom_client_datalist_json_path(datalist_json_path, self.client_id)

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
        self.model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = DiceLoss(sigmoid=True)
        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx, f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)

        self.transform_train = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.3, 0.3, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                DivisiblePadd(keys=["image", "label"], k=32),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.roi_size,
                    pos=1,
                    neg=1,
                    num_samples=4,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        self.transform_valid = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.3, 0.3, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                DivisiblePadd(keys=["image", "label"], k=32),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        self.transform_post = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # Set dataset
        self.train_dataset = Dataset(
            data=train_list,
            transform=self.transform_train,
        )
        self.train_dataset_for_valid = Dataset(
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
        self.train_for_valid_loader = DataLoader(
            self.train_dataset_for_valid,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

        # Set inferer and evaluation metric
        self.inferer = SlidingWindowInferer(roi_size=self.infer_roi_size, sw_batch_size=4, overlap=0.25)
        self.valid_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    def local_valid(self, valid_loader, abort_signal: Signal, tb_id=None):
        self.model.eval()
        with torch.no_grad():
            metric_score = 0
            for i, batch_data in enumerate(valid_loader):
                if abort_signal.triggered:
                    return self._abort_execution()
                val_images = batch_data["image"].to(self.device)
                val_labels = batch_data["label"].to(self.device)
                # Inference
                val_outputs = self.inferer(val_images, self.model)
                val_outputs = self.transform_post(val_outputs)
                # Compute metric
                metric = self.valid_metric(y_pred=val_outputs, y=val_labels)
                metric_score += metric.item()
            # compute mean dice over whole validation set
            metric_score /= len(valid_loader)
            if tb_id:
                self.writer.add_scalar(tb_id, metric_score, self.epoch_of_start_time)
        return metric_score
