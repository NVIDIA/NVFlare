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
import json

import numpy as np

parser = argparse.ArgumentParser(description="generate train configs for HIGGS dataset")
parser.add_argument("--site_num", type=int, default=5, help="Total number of sites")
parser.add_argument("--nthread", type=int, default=16, help="nthread for xgboost")
parser.add_argument("--site_name", type=str, default="site-", help="Site name prefix")
parser.add_argument("--size_total", type=int, default=11000000, help="Total number of instances, default 11 million")
parser.add_argument("--size_valid", type=int, default=1000000, help="Validation size, default 1 million")
parser.add_argument("--split_method", type=str, default="uniform", help="How to split the dataset")
parser.add_argument("--lr_mode", type=str, default="uniform", help="Whether to use uniform or scaled shrinkage")
parser.add_argument("--out_path", type=str, default="train_configs/config_train_uniform.json", help="Path to json file")


def split_num_proportion(n, site_num, option: str):
    split = []
    if option == "uniform":
        ratio_vec = np.ones(site_num)
    elif option == "linear":
        ratio_vec = np.linspace(1, site_num, num=site_num)
    elif option == "square":
        ratio_vec = np.square(np.linspace(1, site_num, num=site_num))
    elif option == "exponential":
        ratio_vec = np.exp(np.linspace(1, site_num, num=site_num))
    else:
        raise ValueError("Split method not implemented!")

    total = sum(ratio_vec)
    left = n
    for site in range(site_num - 1):
        x = int(n * ratio_vec[site] / total)
        left = left - x
        split.append(x)
    split.append(left)
    return split


def train_config_gen(site_num, nthread, site_name, size_total, size_valid, split_method, out_path):
    json_data = {}
    json_data["data_path"] = "./dataset/HIGGS_UCI.csv"
    json_data["local_model_path"] = "model.json"
    json_data["global_model_path"] = "model_global.json"
    json_data["learning_rate"] = 0.1
    json_data["objective"] = "binary:logistic"
    json_data["max_depth"] = 8
    json_data["eval_metric"] = "auc"
    json_data["nthread"] = nthread
    json_data["data_index"] = {}
    json_data["data_index"]["valid"] = {}
    json_data["data_index"]["valid"]["start"] = 0
    json_data["data_index"]["valid"]["end"] = size_valid

    site_size = split_num_proportion((size_total - size_valid), site_num, split_method)

    for site in range(site_num):
        site_id = site_name + str(site + 1)
        idx_start = size_valid + sum(site_size[:site])
        idx_end = size_valid + sum(site_size[: site + 1])
        json_data["data_index"][site_id] = {}
        json_data["data_index"][site_id]["start"] = idx_start
        json_data["data_index"][site_id]["end"] = idx_end
        json_data["data_index"][site_id]["lr_scale"] = site_size[site] / sum(site_size)

    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    train_config_gen(
        site_num=args.site_num,
        nthread=args.nthread,
        site_name=args.site_name,
        size_total=args.size_total,
        size_valid=args.size_valid,
        split_method=args.split_method,
        out_path=args.out_path,
    )
