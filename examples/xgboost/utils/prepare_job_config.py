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
import os
import shutil

parser = argparse.ArgumentParser(description="generate train configs for HIGGS dataset")
parser.add_argument("--train_config_path", type=str, default="./train_configs", help="Path to train config folder")
parser.add_argument("--job_config_path_root", type=str, default="./job_configs", help="Path to job config folder")
parser.add_argument("--base_job_name", type=str, default="higgs_base", help="Job name of base config")
parser.add_argument("--site_num", type=int, default=5, help="Total number of sites")
parser.add_argument("--round_num", type=int, default=100, help="Total number of training rounds")
parser.add_argument("--train_mode", type=str, default="bagging", help="Training mode")
parser.add_argument("--split_method", type=str, default="uniform", help="How to split the dataset")
parser.add_argument("--lr_mode", type=str, default="uniform", help="Whether to use uniform or scaled shrinkage")


def read_json(filename):
    assert os.path.isfile(filename), f"{filename} does not exist!"
    with open(filename, "r") as f:
        return json.load(f)


def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def main():
    args = parser.parse_args()
    job_name = (
        "higgs_"
        + str(args.site_num)
        + "_"
        + args.train_mode
        + "_"
        + args.split_method
        + "_split"
        + "_"
        + args.lr_mode
        + "_lr"
    )
    train_config_name = "config_train_" + str(args.site_num) + "_" + args.split_method + ".json"
    target_path = os.path.join(args.job_config_path_root, job_name)
    target_config_path = os.path.join(target_path, job_name, "config")
    base_path = os.path.join(args.job_config_path_root, args.base_job_name)

    # make target config folders
    if not os.path.exists(target_config_path):
        os.makedirs(target_config_path)
    # copy files
    meta_config_filename = os.path.join(target_path, "meta.json")
    client_config_filename = os.path.join(target_config_path, "config_fed_client.json")
    server_config_filename = os.path.join(target_config_path, "config_fed_server.json")
    train_config_filename = os.path.join(target_config_path, train_config_name)
    shutil.copyfile(os.path.join(base_path, "meta.json"), meta_config_filename)
    shutil.copyfile(os.path.join(base_path, "higgs_base/config", "config_fed_client.json"), client_config_filename)
    shutil.copyfile(
        os.path.join(base_path, "higgs_base/config", "config_fed_server_" + args.train_mode + ".json"),
        server_config_filename,
    )
    shutil.copyfile(os.path.join(args.train_config_path, train_config_name), train_config_filename)

    # adjust file contents according to each job's specs
    meta_config = read_json(meta_config_filename)
    client_config = read_json(client_config_filename)
    server_config = read_json(server_config_filename)
    # update meta
    meta_config["name"] = job_name
    meta_config["deploy_map"][job_name] = meta_config["deploy_map"][args.base_job_name]
    del meta_config["deploy_map"][args.base_job_name]
    # update client config
    client_config["components"][0]["args"]["train_config_filename"] = train_config_name
    client_config["components"][0]["args"]["lr_mode"] = args.lr_mode
    if args.train_mode == "bagging":
        client_config["components"][0]["args"]["num_parallel_tree"] = args.site_num
        server_config["workflows"][0]["args"]["num_rounds"] = args.round_num + 1
        # update server config
        server_config["workflows"][0]["args"]["min_clients"] = args.site_num
    elif args.train_mode == "cyclic":
        client_config["components"][0]["args"]["num_parallel_tree"] = 1
        server_config["workflows"][0]["args"]["num_rounds"] = int(args.round_num / args.site_num)
    else:
        print(f"Training mode {args.train_mode} not supported")
        return False
    # write jsons
    write_json(meta_config, meta_config_filename)
    write_json(client_config, client_config_filename)
    write_json(server_config, server_config_filename)


if __name__ == "__main__":
    main()
