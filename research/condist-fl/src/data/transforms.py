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

from typing import Literal

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandZoomd,
    Spacingd,
    SpatialPadd,
)

from .augmentations import (
    RandAdjustBrightnessAndContrastd,
    RandFlipAxes3Dd,
    RandInverseIntensityGammad,
    SimulateLowResolutiond,
)
from .normalize import NormalizeIntensityRanged


def get_train_transforms(num_samples: int = 1):
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], as_closest_canonical=True),
            Spacingd(
                keys=["image", "label"], pixdim=[1.44423774, 1.44423774, 2.87368553], mode=["bilinear", "nearest"]
            ),
            RandRotated(
                keys=["image", "label"],
                range_x=0.5236,
                range_y=0.5236,
                range_z=0.5236,
                prob=0.2,
                mode=["bilinear", "nearest"],
                keep_size=False,
            ),
            RandZoomd(
                keys=["image", "label"],
                prob=0.2,
                min_zoom=0.7,
                max_zoom=1.4,
                mode=["trilinear", "nearest"],
                keep_size=False,
            ),
            NormalizeIntensityRanged(keys=["image"], a_min=-54.0, a_max=258.0, subtrahend=100.0, divisor=50.0),
            SpatialPadd(keys=["image", "label"], spatial_size=[224, 224, 64]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[224, 224, 64],
                pos=2.0,
                neg=1.0,
                num_samples=num_samples,
                image_key="image",
            ),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
            RandGaussianSmoothd(keys=["image"], sigma_x=[0.5, 1.5], sigma_y=[0.5, 1.5], sigma_z=[0.5, 1.5], prob=0.15),
            RandAdjustBrightnessAndContrastd(
                keys=["image"], probs=[0.15, 0.15], brightness_range=[0.7, 1.3], contrast_range=[0.65, 1.5]
            ),
            SimulateLowResolutiond(keys=["image"], prob=0.25, zoom_range=[0.5, 1.0]),
            RandAdjustContrastd(keys=["image"], prob=0.15, gamma=[0.8, 1.2]),
            RandInverseIntensityGammad(keys=["image"], prob=0.15, gamma=[0.8, 1.2]),
            RandFlipAxes3Dd(keys=["image", "label"], prob_x=0.50, prob_y=0.50, prob_z=0.50),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return transforms


def get_validate_transforms():
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], as_closest_canonical=True),
            Spacingd(
                keys=["image", "label"], pixdim=[1.44423774, 1.44423774, 2.87368553], mode=["bilinear", "nearest"]
            ),
            NormalizeIntensityRanged(keys=["image"], a_min=-54.0, a_max=258.0, subtrahend=100.0, divisor=50.0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return transforms


def get_infer_transforms():
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], as_closest_canonical=True),
            Spacingd(
                keys=["image", "label"], pixdim=[1.44423774, 1.44423774, 2.87368553], mode=["bilinear", "nearest"]
            ),
            NormalizeIntensityRanged(keys=["image"], a_min=-54.0, a_max=258.0, subtrahend=100.0, divisor=50.0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return transforms


def get_transforms(mode: Literal["train", "validate", "infer"], num_samples: int = 1):
    if mode == "train":
        return get_train_transforms(num_samples=num_samples)
    if mode == "validate":
        return get_validate_transforms()
    if mode == "infer":
        return get_infer_transforms()
    raise ValueError(f"Unsupported transform mode {mode}.")
