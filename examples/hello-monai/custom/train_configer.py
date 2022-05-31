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

import json
import os

import torch
from monai.apps.utils import download_and_extract
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.optimizers import Novograd
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    FgBgToIndicesd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)


class TrainConfiger:
    """
    This class is used to config the necessary components of train and evaluate engines
    for MONAI trainer.
    Please check the implementation of `SupervisedEvaluator` and `SupervisedTrainer`
    from `monai.engines` and determine which components can be used.
    Args:
        app_root: root folder path of config files.
        wf_config_file_name: json file name of the workflow config file.
    """

    def __init__(
        self,
        app_root: str,
        dataset_root: str,
        wf_config_file_name: str,
        dataset_folder_name: str = "Task09_Spleen",
        max_epochs: int = 100,
    ):
        with open(os.path.join(app_root, wf_config_file_name)) as file:
            wf_config = json.load(file)

        self.wf_config = wf_config
        """
        config Args:
            max_epochs: the total epoch number for trainer to run.
            learning_rate: the learning rate for optimizer.
            data_list_json_file: the data list json file.
            val_interval: the interval (number of epochs) to do validation.
            ckpt_dir: the directory to save the checkpoint.
            amp: whether to enable auto-mixed-precision training.
            use_gpu: whether to use GPU in training.

        """
        self.max_epochs = max_epochs
        self.learning_rate = wf_config["learning_rate"]
        self.data_list_json_file = wf_config["data_list_json_file"]
        self.val_interval = wf_config["val_interval"]
        self.ckpt_dir = wf_config["ckpt_dir"]
        self.amp = wf_config["amp"]
        self.use_gpu = wf_config["use_gpu"]
        self.app_root = app_root
        self.dataset_root = dataset_root
        self.dataset_folder_name = dataset_folder_name

        dataset_path = os.path.join(dataset_root, self.dataset_folder_name)
        if not os.path.exists(dataset_path):
            self.download_spleen_dataset(dataset_path)

    def set_device(self):
        device = torch.device("cuda" if self.use_gpu else "cpu")
        self.device = device

    def download_spleen_dataset(self, dataset_path: str):
        url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
        tarfile_name = f"{dataset_path}.tar"
        download_and_extract(
            url=url, filepath=tarfile_name, output_dir=self.dataset_root
        )

    def configure(self):
        self.set_device()
        network = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(self.device)

        train_transforms = Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys="image",
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=("image", "label"), source_key="image"),
                FgBgToIndicesd(
                    keys="label",
                    fg_postfix="_fg",
                    bg_postfix="_bg",
                    image_key="image",
                ),
                RandCropByPosNegLabeld(
                    keys=("image", "label"),
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    fg_indices_key="label_fg",
                    bg_indices_key="label_bg",
                ),
                ToTensord(keys=("image", "label")),
            ]
        )
        # set datalist
        train_datalist = load_decathlon_datalist(
            os.path.join(self.app_root, self.data_list_json_file),
            is_segmentation=True,
            data_list_key="training",
            base_dir=os.path.join(self.dataset_root, self.dataset_folder_name),
        )
        val_datalist = load_decathlon_datalist(
            os.path.join(self.app_root, self.data_list_json_file),
            is_segmentation=True,
            data_list_key="validation",
            base_dir=os.path.join(self.dataset_root, self.dataset_folder_name),
        )
        train_ds = CacheDataset(
            data=train_datalist,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=4,
        )
        train_data_loader = DataLoader(
            train_ds,
            batch_size=4,
            shuffle=True,
            num_workers=4,
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=("image", "label")),
                EnsureChannelFirstd(keys=("image", "label")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys="image",
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=("image", "label"), source_key="image"),
                ToTensord(keys=("image", "label")),
            ]
        )

        val_ds = CacheDataset(
            data=val_datalist, transform=val_transforms, cache_rate=0.0, num_workers=4
        )
        val_data_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )
        post_transform = Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(
                    keys=["pred", "label"],
                    argmax=[True, False],
                    to_onehot=2,
                ),
            ]
        )
        # metric
        key_val_metric = {
            "val_mean_dice": MeanDice(
                include_background=False,
                output_transform=from_engine(["pred", "label"]),
            )
        }
        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
            CheckpointSaver(
                save_dir=self.ckpt_dir,
                save_dict={"model": network},
                save_key_metric=True,
            ),
            TensorBoardStatsHandler(
                log_dir=self.ckpt_dir, output_transform=lambda x: None
            ),
        ]
        self.eval_engine = SupervisedEvaluator(
            device=self.device,
            val_data_loader=val_data_loader,
            network=network,
            inferer=SlidingWindowInferer(
                roi_size=[160, 160, 160],
                sw_batch_size=4,
                overlap=0.5,
            ),
            postprocessing=post_transform,
            key_val_metric=key_val_metric,
            val_handlers=val_handlers,
            amp=self.amp,
        )

        optimizer = Novograd(network.parameters(), self.learning_rate)
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, batch=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5000, gamma=0.1
        )
        train_handlers = [
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
            ValidationHandler(
                validator=self.eval_engine, interval=self.val_interval, epoch_level=True
            ),
            StatsHandler(
                tag_name="train_loss", output_transform=from_engine("loss", first=True)
            ),
            TensorBoardStatsHandler(
                log_dir=self.ckpt_dir,
                tag_name="train_loss",
                output_transform=from_engine("loss", first=True),
            ),
        ]

        self.train_engine = SupervisedTrainer(
            device=self.device,
            max_epochs=self.max_epochs,
            train_data_loader=train_data_loader,
            network=network,
            optimizer=optimizer,
            loss_function=loss_function,
            inferer=SimpleInferer(),
            postprocessing=post_transform,
            key_train_metric=None,
            train_handlers=train_handlers,
            amp=self.amp,
        )
