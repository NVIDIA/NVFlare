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
import os

import numpy as np
import torchvision.datasets as datasets


def main():
    parser = argparse.ArgumentParser(description="generate train_config for brats")
    parser.add_argument("--dataset_base_dir", type=str, default=None)
    parser.add_argument("--datalist_json_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--config_dir", "-o", type=str)
    parser.add_argument("--num_sites", type=int)
    parser.add_argument("--site_pre", type=str, default=None)
    args = parser.parse_args()

    if os.path.isfile(os.path.join(args.config_dir, "config_train.json")):
        with open(os.path.join(args.config_dir, "config_train.json")) as outfile:
            config_train = json.load(outfile)
    else:
        config_train = {}
    if args.lr:
        config_train["learning_rate"] = args.lr
    if args.dataset_base_dir:
        config_train["dataset_base_dir"] = args.dataset_base_dir
    if args.datalist_json_path:
        config_train["datalist_json_path"] = args.datalist_json_path

    with open(os.path.join(args.config_dir, "config_train.json"), "w") as outfile:
        json.dump(config_train, outfile, indent=2)

    with open(os.path.join(args.config_dir, "config_fed_server.json")) as config_fed_server_file:
        config_fed_server = json.load(config_fed_server_file)
    config_fed_server["min_clients"] = args.num_sites

    # assign equal weights for each client
    if args.site_pre is not None:
        for i in range(len(config_fed_server["components"])):
            if config_fed_server["components"][i]["id"] == "aggregator":
                config_fed_server["components"][i]["args"]["aggregation_weights"] = {
                    args.site_pre + str(site_id): 1.0 for site_id in range(1, args.num_sites + 1)
                }

    with open(os.path.join(args.config_dir, "config_fed_server.json"), "w") as outfile:
        json.dump(config_fed_server, outfile, indent=2)


if __name__ == "__main__":
    main()
