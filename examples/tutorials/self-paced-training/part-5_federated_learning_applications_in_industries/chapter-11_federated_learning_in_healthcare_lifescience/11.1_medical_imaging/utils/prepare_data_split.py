# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import glob
import json
import os

import numpy as np

np.random.seed(0)

# Prostate data is arranged as ${data_dir}/${site_name}/Image/ and  ${data_dir}/${site_name}/Mask/
# Image-Mask pairs have identical filename, and stored separately
# output json file is named client_${site_id}.json

parser = argparse.ArgumentParser(description="generate multi-site train/valid splits for dataset")
parser.add_argument("--data_dir", type=str, help="Path to data folder")
parser.add_argument("--site_num", type=int, default=4, help="Number of sites, default 4")
parser.add_argument("--train", type=float, default=0.8, help="Portion of training set, default 80%")
parser.add_argument("--valid", type=float, default=0.2, help="Portion of validation set, default 20%")
parser.add_argument("--out_path", type=str, help="Path to datalist json file")


def partition_data(data_path, site_num, train, valid, out_path):
    print(f"Generate data split for {data_path}, with train:validation {train}:{valid} to {site_num} sites")
    print(f"Save json to {out_path}")

    image_file_path = os.path.join(data_path, "Image", "*")
    mask_file_path = os.path.join(data_path, "Mask", "*")
    image_files = glob.glob(image_file_path)
    mask_files = glob.glob(mask_file_path)

    assert len(image_files) == len(mask_files), "The number of image and mask files should be the same."
    # sort will produce the same sequence since filenames are identical for image and masks
    image_files.sort()
    mask_files.sort()

    # produce random index for split
    length = len(image_files)
    rand_idx = np.arange(length)
    np.random.shuffle(rand_idx)

    # check the ratio sum
    assert (train + valid) == 1, "Sum of all two splits should be 1."

    train_num = round(length * train)

    # split the training data index into site_num parts
    # use same validation data for all sites
    train_split = np.array_split(rand_idx[:train_num], site_num)
    valid_split = rand_idx[train_num:]

    # generate json data for each site
    # json data item is a list of image and mask paths
    for site_id in range(site_num):
        json_data = {"training": [], "validation": []}
        for idx in train_split[site_id]:
            new_item = {}
            new_item["image"] = image_files[idx].replace(data_path + "/", "")
            new_item["label"] = mask_files[idx].replace(data_path + "/", "")
            json_data["training"].append(new_item)
        for idx in valid_split:
            new_item = {}
            new_item["image"] = image_files[idx].replace(data_path + "/", "")
            new_item["label"] = mask_files[idx].replace(data_path + "/", "")
            json_data["validation"].append(new_item)
        # save json data for each site
        with open(out_path + f"/client_{site_id}.json", "w") as f:
            json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    partition_data(
        data_path=args.data_dir,
        site_num=args.site_num,
        train=args.train,
        valid=args.valid,
        out_path=args.out_path,
    )
