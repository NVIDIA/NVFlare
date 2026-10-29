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

"""Replaceable image-generation and augmentation defaults for data-free KD."""

import torch
from torch import nn
from torch.nn import functional as F


class ImageGenerator(nn.Module):
    """Map [batch, latent_size] noise to [batch, C, H, W] images in [0, 1].

    The default shape is RGB 32x32; channels and spatial dimensions are
    configurable. This small convolutional generator is a starting image prior,
    not a task-independent data model or a FedRevive reproduction preset.
    """

    def __init__(self, latent_size=128, input_shape=(3, 32, 32), features=32):
        super().__init__()
        if len(input_shape) != 3 or any(not isinstance(value, int) or value < 1 for value in input_shape):
            raise ValueError("input_shape must contain positive C, H, W dimensions")
        if latent_size < 1 or features < 1:
            raise ValueError("latent_size and features must be positive")
        self.input_shape = tuple(input_shape)
        self.project = nn.Linear(latent_size, features * 2 * 4 * 4)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(features * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features * 2, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(features, input_shape[0], 3, padding=1),
            nn.Sigmoid(),
        )
        self.features = features

    def forward(self, noise):
        images = self.layers(self.project(noise).reshape(len(noise), self.features * 2, 4, 4))
        return F.interpolate(images, size=self.input_shape[1:], mode="bilinear", align_corners=False)


class ImageTransform(nn.Module):
    """Differentiable crop/flip, with optional per-channel normalization.

    A crop offset is sampled per batch and flip decisions per image. Defaults
    preserve [0, 1] image values. Supply mean/std for the teacher's normalization
    or replace the transform for another input domain. Randomness uses the
    caller's CPU generator, or the global CPU RNG; its state is not serialized.
    """

    def __init__(self, padding=2, flip_probability=0.5, mean=None, std=None, generator=None):
        super().__init__()
        if not isinstance(padding, int) or padding < 0 or not 0 <= flip_probability <= 1:
            raise ValueError("padding must be nonnegative and flip_probability must be in [0, 1]")
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be supplied together")
        self.padding = padding
        self.flip_probability = flip_probability
        self.generator = generator
        mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1, 1)
        std = None if std is None else torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1, 1)
        if std is not None and (
            mean.shape != std.shape
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("mean/std must be finite, matching channel vectors, with positive std")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("ImageTransform requires [batch, channels, height, width] inputs")
        height, width = images.shape[-2:]
        if self.padding:
            images = F.pad(images, (self.padding,) * 4, mode="replicate")
            top, left = torch.randint(2 * self.padding + 1, (2,), generator=self.generator).tolist()
            images = images[:, :, top : top + height, left : left + width]
        flip = torch.rand((len(images), 1, 1, 1), generator=self.generator).to(images.device) < self.flip_probability
        images = torch.where(flip, images.flip(-1), images)
        if self.mean is not None:
            if images.shape[1] != self.mean.shape[1]:
                raise ValueError("normalization channel count does not match images")
            images = (images - self.mean) / self.std
        return images
