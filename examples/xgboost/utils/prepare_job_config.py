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
import pathlib
import shutil

from nvflare.apis.fl_constant import JobConstants


def job_config_args_parser():
    parser = argparse.ArgumentParser(description="generate train configs for HIGGS dataset")
    parser.add_argument("--data_split_path", type=str, default="./data_splits", help="Path to data split folder")
    parser.add_argument(
        "--job_configs_root", type=str, default="./job_configs", help="Path to root folder of all job configs"
    )
    parser.add_argument("--site_num", type=int, default=5, help="Total number of sites")
    parser.add_argument("--round_num", type=int, default=100, help="Total number of training rounds")
    parser.add_argument(
        "--training_mode", type=str, default="bagging", choices=["bagging", "cyclic", "histogram"], help="Training mode"
    )
    parser.add_argument("--split_method", type=str, default="uniform", help="How to split the dataset")
    parser.add_argument("--lr_mode", type=str, default="uniform", help="Whether to use uniform or scaled shrinkage")
    parser.add_argument("--nthread", type=int, default=16, help="nthread for xgboost")
    parser.add_argument(
        "--tree_method", type=str, default="hist", help="tree_method for xgboost - use hist or gpu_hist for best perf"
    )
    return parser


def read_json(filename):
    if not os.path.isfile(filename):
        raise ValueError(f"{filename} does not exist!")
    with open(filename, "r") as f:
        return json.load(f)


def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def _get_job_name(args) -> str:
    return (
        "higgs_"
        + str(args.site_num)
        + "_"
        + args.training_mode
        + "_"
        + args.split_method
        + "_split"
        + "_"
        + args.lr_mode
        + "_lr"
    )


def _get_data_split_name(args) -> str:
    return "data_split_" + str(args.site_num) + "_" + args.split_method + ".json"


def update_meta(meta: dict, args):
    name = _get_job_name(args)
    meta["name"] = name


def _update_client_config(config: dict, args):
    data_split_name = _get_data_split_name(args)
    if args.training_mode == "bagging" or args.training_mode == "cyclic":
        # update client config
        config["executors"][0]["executor"]["args"]["data_split_filename"] = data_split_name
        config["executors"][0]["executor"]["args"]["lr_mode"] = args.lr_mode
        config["executors"][0]["executor"]["args"]["nthread"] = args.nthread
        config["executors"][0]["executor"]["args"]["tree_method"] = args.tree_method
        config["executors"][0]["executor"]["args"]["training_mode"] = args.training_mode
        if args.training_mode == "bagging":
            config["executors"][0]["executor"]["args"]["num_tree_bagging"] = args.site_num
        elif args.training_mode == "cyclic":
            config["executors"][0]["executor"]["args"]["num_tree_bagging"] = 1
    else:
        config["executors"][0]["executor"]["args"]["data_split_filename"] = data_split_name
        config["executors"][0]["executor"]["args"]["xgboost_params"]["nthread"] = args.nthread
        config["executors"][0]["executor"]["args"]["xgboost_params"]["tree_method"] = args.tree_method


def _update_server_config(config: dict, args):
    if args.training_mode == "bagging":
        config["workflows"][0]["args"]["num_rounds"] = args.round_num + 1
        config["workflows"][0]["args"]["min_clients"] = args.site_num
    elif args.training_mode == "cyclic":
        config["workflows"][0]["args"]["num_rounds"] = int(args.round_num / args.site_num)


def copy_custom_files(src_job_path, dst_job_path):
    dst_path = dst_job_path / "app" / "custom"
    os.makedirs(dst_path, exist_ok=True)
    src_path = src_job_path / "app" / "custom"
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


def main():
    parser = job_config_args_parser()
    args = parser.parse_args()
    job_name = _get_job_name(args)
    data_split_name = _get_data_split_name(args)

    ref_job_map = {"bagging": "tree-based", "cyclic": "tree-based", "histogram": "histogram-based"}
    src_job_path = pathlib.Path(ref_job_map[args.training_mode]) / args.job_configs_root / "base"
    meta_config = read_json(src_job_path / JobConstants.META_FILE)
    client_config = read_json(src_job_path / "app" / "config" / JobConstants.CLIENT_JOB_CONFIG)
    server_config = read_json(src_job_path / "app" / "config" / JobConstants.SERVER_JOB_CONFIG)

    # adjust file contents according to each job's specs
    update_meta(meta_config, args)
    _update_client_config(client_config, args)
    _update_server_config(server_config, args)

    # create a new job
    dst_job_path = pathlib.Path(ref_job_map[args.training_mode]) / args.job_configs_root / job_name
    app_config_path = dst_job_path / "app" / "config"

    # make target config folders
    if not os.path.exists(app_config_path):
        os.makedirs(app_config_path)

    meta_config_filename = dst_job_path / JobConstants.META_FILE
    client_config_filename = app_config_path / JobConstants.CLIENT_JOB_CONFIG
    server_config_filename = app_config_path / JobConstants.SERVER_JOB_CONFIG

    write_json(meta_config, meta_config_filename)
    write_json(client_config, client_config_filename)
    write_json(server_config, server_config_filename)

    # copy data split
    data_split_filename = app_config_path / data_split_name
    shutil.copyfile(os.path.join(args.data_split_path, data_split_name), data_split_filename)

    # copy custom file
    copy_custom_files(src_job_path, dst_job_path)


if __name__ == "__main__":
    main()
