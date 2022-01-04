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

import argparse
import json
import math
import os

import numpy as np

np.random.seed(0)

# Data is arranged as ${data_dir}/${site_name}/Image/ and  ${data_dir}/${site_name}/Mask/
# Image-Mask pairs have identical filename, and stored separately
# output json file is named client_${site_name}.json

parser = argparse.ArgumentParser(description="generate train/valid/test splits for datasets")
parser.add_argument("--data_dir", type=str, help="Path to image folder")
parser.add_argument("--site_name", type=str, help="Path to image folder")
parser.add_argument("--train", type=int, default=8, help="Portion of training set, default 8")
parser.add_argument("--valid", type=int, default=2, help="Portion of validation set, default 2")
parser.add_argument("--test", type=int, default=0, help="Portion of validation set, default 0 (no test set)")
parser.add_argument("--out_path", type=str, help="Path to datalise json file")


def partition_data(data_path, site_name, train, valid, test, out_path):
    print(f"Generate data split for {data_path}/{site_name}, with train:validation:test {train}:{valid}:{test}")
    print(f"Save json to {out_path}/client_{site_name}.json")

    tra = 0
    val = 0
    tst = 0

    json_data = {"training": [], "validation": [], "testing": []}

    image_file_folder = os.path.join(data_path, site_name + "/Image")
    files = [d for d in os.listdir(image_file_folder) if os.path.isfile(os.path.join(image_file_folder, d))]

    rand_idx = np.random.permutation(len(files))
    tra_cut = math.ceil(len(files) * train / (train + valid + test))
    val_cut = math.ceil(len(files) * (train + valid) / (train + valid + test))

    for count in range(len(files)):
        file_id = files[rand_idx[count]]
        new_item = {}
        if count < tra_cut:
            to_append = "training"
            tra = tra + 1
        elif count < val_cut:
            to_append = "validation"
            val = val + 1
        else:
            to_append = "testing"
            tst = tst + 1

        new_item["image"] = os.path.join(site_name, "Image", file_id)
        new_item["label"] = os.path.join(site_name, "Mask", file_id)
        temp = json_data[to_append]
        temp.append(new_item)

    print(f"In total {len(files)} cases, {tra} for training, {val} for validation, and {tst} for testing")
    with open(os.path.join(out_path, "client_" + site_name + ".json"), "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    data_path = os.path.join(args.data_dir)
    out_path = args.out_path
    partition_data(
        data_path=data_path,
        site_name=args.site_name,
        train=args.train,
        valid=args.valid,
        test=args.test,
        out_path=out_path,
    )
