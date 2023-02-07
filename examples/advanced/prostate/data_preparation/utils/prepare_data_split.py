# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
# output json file is named client_${site_name}.json

parser = argparse.ArgumentParser(description="generate train/valid/test splits for datasets")
parser.add_argument(
    "--mode", type=str, help="Split mode, mode can either be 'folder' or 'file', controlling the split level"
)
parser.add_argument("--data_dir", type=str, help="Path to data folder")
parser.add_argument("--site_name", type=str, help="Path to particular set")
parser.add_argument("--train", type=float, default=0.5, help="Portion of training set, default 50%")
parser.add_argument("--valid", type=float, default=0.25, help="Portion of validation set, default 25%")
parser.add_argument("--test", type=float, default=0.25, help="Portion of testing set, default 25%")
parser.add_argument("--out_path", type=str, help="Path to datalist json file")


def partition_data(mode, data_path, site_name, train, valid, test, out_path):
    assert mode in ["folder", "file"], "mode should either be 'folder' or 'file'"
    print(f"Generate data split for {data_path}/{site_name}, with train:validation:test {train}:{valid}:{test}")
    print(f"Save json to {out_path}")
    print(f"Mode: {mode}")
    tra = 0
    val = 0
    tst = 0
    tra_i = 0
    val_i = 0
    tst_i = 0
    total_file = 0
    json_data = {"training": [], "validation": [], "testing": []}

    image_file_path = os.path.join(data_path, site_name, "Image", "*")
    mask_file_path = os.path.join(data_path, site_name, "Mask", "*")
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
    assert (train + valid + test) == 1, "Sum of all three splits should be 1."

    tra_cut = round(length * train)
    val_cut = round(length * train) + round(length * valid)

    for count in range(length):
        # if folder, add all images inside it
        if mode == "folder":
            image_file_name = glob.glob(os.path.join(image_files[rand_idx[count]], "*"))
            mask_file_name = glob.glob(os.path.join(mask_files[rand_idx[count]], "*"))
            image_file_name.sort()
            mask_file_name.sort()
        elif mode == "file":
            image_file_name = [image_files[rand_idx[count]]]
            mask_file_name = [mask_files[rand_idx[count]]]

        if count < tra_cut:
            to_append = "training"
            tra = tra + 1
            tra_i = tra_i + len(image_file_name)
        elif count < val_cut:
            to_append = "validation"
            val = val + 1
            val_i = val_i + len(image_file_name)
        else:
            to_append = "testing"
            tst = tst + 1
            tst_i = tst_i + len(image_file_name)
        total_file = total_file + len(image_file_name)

        for idx in range(len(image_file_name)):
            new_item = {}
            # collect the paths, excluding the common data_root
            new_item["image"] = image_file_name[idx].replace(data_path + "/", "")
            new_item["label"] = mask_file_name[idx].replace(data_path + "/", "")
            temp = json_data[to_append]
            temp.append(new_item)

    print(f"In total {length} cases, {tra} for training, {val} for validation, and {tst} for testing")
    if mode == "folder":
        print(
            f"In total {total_file} samples, split at case level, {tra_i} for training, {val_i} for validation, and {tst_i} for testing"
        )
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    partition_data(
        mode=args.mode,
        data_path=args.data_dir,
        site_name=args.site_name,
        train=args.train,
        valid=args.valid,
        test=args.test,
        out_path=args.out_path,
    )
