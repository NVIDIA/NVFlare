# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import random

import cv2
import numpy as np
import torch
import torchvision.transforms


def get_transformations(size, train=True):
    """
    Get data transforms and augmentations
    :param tuple size: Image size
    :param bool train: Training or validation
    :return Compose transform object
    """
    if train:
        augs = torchvision.transforms.Compose(
            [
                RandomScaling(min_scale_factor=0.5, max_scale_factor=2.0),
                RandomCrop(size, cat_max_ratio=0.75),
                RandomHorizontallyFlip(),
                PhotoMetricDistortion(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PadImage(size),
                AddIgnoreRegions(),
                ToTensor(),
            ]
        )
    else:
        augs = torchvision.transforms.Compose(
            [
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PadImage(size),
                AddIgnoreRegions(),
                ToTensor(),
            ]
        )

    return augs


class RandomScaling(object):
    """
    Randomly scale the image and labels
    :param float min_scale_factor: Minimum scaling value
    :param float max_scale_factor: Maximum scaling value
    """

    def __init__(self, min_scale_factor, max_scale_factor):
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor

    def get_random_scale(self):
        if self.min_scale_factor < 0 or self.min_scale_factor > self.max_scale_factor:
            raise ValueError("Unexpected value of 'min_scale_factor'!")

        if self.min_scale_factor == self.max_scale_factor:
            min_scale_factor = float(self.min_scale_factor)
            return min_scale_factor

        # Uniformly sampling of the value from [min, max)
        return np.random.uniform(low=self.min_scale_factor, high=self.max_scale_factor)

    def scale(self, key, unscaled, scale):
        # No random scaling if scale == 1.
        if scale == 1.0:
            return unscaled
        image_shape = unscaled.shape[0:2]
        new_dim = tuple([int(x * scale) for x in image_shape])

        unscaled = np.squeeze(unscaled)
        if key == 'image':  # float value, linear interpolation
            scaled = cv2.resize(unscaled, new_dim[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            scaled = cv2.resize(unscaled, new_dim[::-1], interpolation=cv2.INTER_NEAREST)
        if scaled.ndim == 2:
            scaled = np.expand_dims(scaled, axis=2)

        # Adjust depth maps with rescaling
        if key == 'depth':
            scaled /= scale

        return scaled

    def __call__(self, sample):
        random_scale = self.get_random_scale()

        for key, target in sample.items():
            sample[key] = self.scale(key, target, random_scale)

        return sample


class RandomCrop(object):
    """
    Randomly crop image and labels if it exceeds desired size
    :param int/tuple size: Desired size
    """

    def __init__(self, size, cat_max_ratio=1):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError('Crop size must be an int or a tuple!')
        self.cat_max_ratio = cat_max_ratio

    def get_random_crop_loc(self, uncropped):
        uncropped_shape = uncropped.shape
        img_height = uncropped_shape[0]
        img_width = uncropped_shape[1]

        desired_height = self.size[0]
        desired_width = self.size[1]
        if img_height == desired_height and img_width == desired_width:
            return None

        # Get random offset uniformly from [0, max_offset)
        max_offset_height = max(img_height - desired_height, 0)
        max_offset_width = max(img_width - desired_width, 0)

        offset_height = random.randint(0, max_offset_height)
        offset_width = random.randint(0, max_offset_width)
        crop_loc = [offset_height, offset_height + desired_height, offset_width, offset_width + desired_width]

        return crop_loc

    def random_crop(self, uncropped, crop_loc):
        if not crop_loc:
            return uncropped

        cropped = uncropped[crop_loc[0] : crop_loc[1], crop_loc[2] : crop_loc[3], :]

        return cropped

    def __call__(self, sample):
        crop_location = self.get_random_crop_loc(sample['image'])

        if self.cat_max_ratio < 1.0 and 'semseg' in sample.keys():
            # Repeat 10 times
            for _ in range(10):
                seg_tmp = self.random_crop(sample['semseg'], crop_location)
                labels, cnt = np.unique(seg_tmp, return_counts=True)
                cnt = cnt[labels != 255]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_location = self.get_random_crop_loc(sample['image'])

        for key, target in sample.items():
            sample[key] = self.random_crop(target, crop_location)

        return sample


class RandomHorizontallyFlip(object):
    """
    Randomly horizontally flip image and labels with probability of 0.5
    """

    def __call__(self, sample):
        if random.random() < 0.5:
            for key, val in sample.items():
                sample[key] = np.fliplr(val).copy()
                # Flip the normal direction
                if key == 'normals':
                    sample[key][:, :, 0] *= -1

        return sample


class PadImage(object):
    """
    Pad image and labels to have dimensions >= [size_height, size_width]
    :param int/tuple size: Desired size
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError('Padding size must be an int or a tuple!')

        self.fill_index = {
            'image': [0, 0, 0],
            'edge': 255,
            'semseg': 255,
            'human_parts': 255,
            'sal': 255,
            'normals': [0.0, 0.0, 0.0],
            'depth': 0,
        }

    def pad(self, key, unpadded):
        unpadded_shape = unpadded.shape

        if unpadded_shape[0] >= self.size[0] and unpadded_shape[1] >= self.size[1]:
            return unpadded

        delta_height = max(self.size[0] - unpadded_shape[0], 0)
        delta_width = max(self.size[1] - unpadded_shape[1], 0)

        # Location to place image
        height_location = [delta_height // 2, (delta_height // 2) + unpadded_shape[0]]
        width_location = [delta_width // 2, (delta_width // 2) + unpadded_shape[1]]

        pad_value = self.fill_index[key]
        max_height = max(self.size[0], unpadded_shape[0])
        max_width = max(self.size[1], unpadded_shape[1])

        padded = np.full((max_height, max_width, unpadded_shape[2]), pad_value, dtype=unpadded.dtype)
        padded[height_location[0] : height_location[1], width_location[0] : width_location[1], :] = unpadded

        return padded

    def __call__(self, sample):
        for key, val in sample.items():
            sample[key] = self.pad(key, val)

        return sample


class Normalize:
    """
    Normalize image by first mapping from [0, 255] to [0, 1] and then applying standardization.
    :param list mean: Mean values for each channel
    :param list std: Standard deviation values for each channel
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def normalize_img(self, img):
        assert img.dtype == np.float32
        scaled = img.copy() / 255.0
        scaled -= self.mean
        scaled /= self.std

        return scaled

    def __call__(self, sample):
        sample['image'] = self.normalize_img(sample['image'])

        return sample


class AddIgnoreRegions:
    """
    Add ignore regions to labels
    """

    def __call__(self, sample):
        for key in sample.keys():
            tmp = sample[key]
            if key == 'normals':
                # Check areas with norm 0
                norm = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2 + tmp[:, :, 2] ** 2)
                tmp[norm == 0, :] = 255
                sample[key] = tmp
            elif key == 'human_parts':
                # Check for images without human part annotations
                if ((tmp == 0) | (tmp == 255)).all():
                    tmp = np.full(tmp.shape, 255, dtype=tmp.dtype)
                    sample[key] = tmp
            elif key == 'depth':
                tmp[tmp == 0] = 255
                sample[key] = tmp

        return sample


class PhotoMetricDistortion:
    """
    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from RGB to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to RGB
    7. random contrast (mode 1)

    :param int brightness_delta: Delta of brightness, defaults to 32
    :param tuple contrast_range: Range of contrast, defaults to (0.5, 1.5)
    :param tuple saturation_range: Range of saturation, defaults to (0.5, 1.5)
    :param int hue_delta: Delta of hue, defaults to 18
    """

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.random() < 0.5:
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        if random.random() < 0.5:
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        # image in HSV color
        if random.random() < 0.5:
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(self.saturation_lower, self.saturation_upper)
            )
        return img

    def hue(self, img):
        # image in HSV color
        if random.random() < 0.5:
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta - 1)) % 180
        return img

    def __call__(self, sample):
        img = sample['image']
        img = img.astype(np.uint8)  # functions need a uint8 image

        # f_mode == True -> do random contrast first, False -> do random contrast last
        f_mode = random.random() < 0.5

        img = self.brightness(img)

        if f_mode:
            img = self.contrast(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = self.saturation(img)
        img = self.hue(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        if not f_mode:
            img = self.contrast(img)

        sample['image'] = img.astype(np.float32)

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        for key, val in sample.items():
            sample[key] = torch.from_numpy(val.transpose((2, 0, 1))).float()

        return sample
