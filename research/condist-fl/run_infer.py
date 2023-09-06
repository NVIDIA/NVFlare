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

import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from monai.transforms import AsDiscrete, EnsureChannelFirst, LoadImage, SaveImage, Spacing


class ImageDataset(object):
    def __init__(self, data_root: str, data_list: str, data_list_key: str):
        with open(data_list) as f:
            data = json.load(f).get(data_list_key, [])
        self.data = [os.path.join(data_root, d["image"]) for d in data]
        self.index = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class DataProcessor(object):
    def __init__(self, i_min: float, i_max: float, mean: float, std: float, output_dir: str) -> None:
        self.i_min = i_min
        self.i_max = i_max
        self.mean = mean
        self.std = std

        self.reader = LoadImage(image_only=True)
        self.channel = EnsureChannelFirst()
        self.spacing = Spacing(pixdim=[1.44, 1.44, 2.87])
        self.post = AsDiscrete(argmax=True)
        self.writer = SaveImage(
            output_dir=output_dir, output_postfix="seg", output_dtype=np.uint8, separate_folder=False, resample=True
        )

    def preprocess(self, input_file_name: str) -> torch.Tensor:
        image = self.reader(input_file_name)
        image = self.channel(image)
        image = image.cuda()
        image = self.spacing(image)
        # Use inplace operations to avoid Tensor creation
        image = image.clip_(self.i_min, self.i_max)
        image.add_(-self.mean)
        image.div_(self.std)
        return image

    def postprocess(self, seg: torch.Tensor) -> None:
        seg = self.post(seg)
        self.writer(seg)


def main(args):
    data = ImageDataset(args.data_root, args.data_list, args.data_list_key)

    dp = DataProcessor(i_min=-54.0, i_max=258.0, mean=100.0, std=50.0, output_dir=args.output)

    inferer = SlidingWindowInferer(roi_size=[224, 224, 64], mode="gaussian", sw_batch_size=1, overlap=0.50)

    model = torch.jit.load(args.model)
    model = model.eval().cuda()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for image in iter(data):
                image = dp.preprocess(input_file_name=image)
                image = image.view(1, *image.shape)

                output = inferer(image, model)
                output = torch.squeeze(output, dim=0)
                dp.postprocess(output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-r", type=str, help="Path to data root directory.")
    parser.add_argument("--data_list", "-l", type=str, help="Path to data list file.")
    parser.add_argument("--data_list_key", "-k", type=str, help="Target data split key in data list.")
    parser.add_argument("--model", "-m", type=str, help="Path to model torchscript file.")
    parser.add_argument("--output", "-o", type=str, help="Output directory.")
    args = parser.parse_args()

    main(args)
