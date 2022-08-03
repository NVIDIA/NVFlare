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

parser = argparse.ArgumentParser(description="generate splits for HIGGS dataset")
parser.add_argument("--site_num", type=int, default=5, help="Total number of sites")
parser.add_argument("--site_name", type=str, default="site-", help="Site name prefix")
parser.add_argument("--size_total", type=int, default=11000000, help="Total number of instances, default 11 million")
parser.add_argument("--size_valid", type=int, default=1000000, help="Validation size, default 1 million")
parser.add_argument("--out_path", type=str, default="config_train.json", help="Path to datalist json file")


def train_config_gen(site_num, site_name, size_total, size_valid, out_path):
    json_data = {}
    json_data["data_path"] = "/media/ziyuexu/Data/HIGGS/HIGGS_UCI.csv"
    json_data["model_path"] = "model.json"
    json_data["learning_rate"] = 0.1
    json_data["objective"] = "binary:logistic"
    json_data["max_depth"] = 8
    json_data["eval_metric"] = "auc"
    json_data["nthread"] = 16
    json_data["data_index"] = {}
    json_data["data_index"]["valid"] = {}
    json_data["data_index"]["valid"]["start"] = 0
    json_data["data_index"]["valid"]["end"] = size_valid

    site_size = int((size_total - size_valid) / site_num)
    for site in range(site_num):
        site_id = site_name + str(site + 1)
        idx_start = size_valid + site_size * site
        idx_end = size_valid + site_size * (site + 1)
        json_data["data_index"][site_id] = {}
        json_data["data_index"][site_id]["start"] = idx_start
        json_data["data_index"][site_id]["end"] = idx_end

    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    train_config_gen(
        site_num=args.site_num,
        site_name=args.site_name,
        size_total=args.size_total,
        size_valid=args.size_valid,
        out_path=args.out_path,
    )
