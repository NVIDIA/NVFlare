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

import argparse
import json
import os
import pathlib
import shutil
from enum import Enum
from typing import List

import numpy as np

from nvflare.apis.fl_constant import JobConstants

JOBS_ROOT = "jobs"


class SplitMethod(Enum):
    UNIFORM = "uniform"
    LINEAR = "linear"
    SQUARE = "square"
    EXPONENTIAL = "exponential"


def job_config_args_parser():
    parser = argparse.ArgumentParser(description="generate train configs with data split")
    parser.add_argument("--task_name", type=str, help="Task name for the config")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--site_num", type=int, help="Total number of sites")
    parser.add_argument("--site_name_prefix", type=str, default="site-", help="Site name prefix")
    parser.add_argument(
        "--data_size",
        type=int,
        default=0,
        help="Total data size, use if specified, in order to use partial data"
        "If not specified, use the full data size fetched from file.",
    )
    parser.add_argument(
        "--valid_frac",
        type=float,
        help="Validation fraction of the total size, N = round(total_size* valid_frac), "
        "the first N to be treated as validation data. "
        "special case valid_frac = 1, where all data will be used"
        "in validation, e.g. for evaluating unsupervised clustering with known ground truth label.",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="uniform",
        choices=["uniform", "linear", "square", "exponential"],
        help="How to split the dataset",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sklearn",
        choices=["sklearn", "cuml"],
        help="Backend library used",
    )
    return parser


def get_split_ratios(site_num: int, split_method: SplitMethod):
    if split_method == SplitMethod.UNIFORM:
        ratio_vec = np.ones(site_num)
    elif split_method == SplitMethod.LINEAR:
        ratio_vec = np.linspace(1, site_num, num=site_num)
    elif split_method == SplitMethod.SQUARE:
        ratio_vec = np.square(np.linspace(1, site_num, num=site_num))
    elif split_method == SplitMethod.EXPONENTIAL:
        ratio_vec = np.exp(np.linspace(1, site_num, num=site_num))
    else:
        raise ValueError(f"Split method {split_method.name} not implemented!")

    return ratio_vec


def split_num_proportion(n, site_num, split_method: SplitMethod) -> List[int]:
    split = []
    ratio_vec = get_split_ratios(site_num, split_method)
    total = sum(ratio_vec)
    left = n
    for site in range(site_num - 1):
        x = int(n * ratio_vec[site] / total)
        left = left - x
        split.append(x)
    split.append(left)
    return split


def assign_data_index_to_sites(
    data_size: int,
    valid_fraction: float,
    num_sites: int,
    site_name_prefix: str,
    split_method: SplitMethod = SplitMethod.UNIFORM,
) -> dict:
    if valid_fraction > 1.0:
        raise ValueError("validation percent should be less than or equal to 100% of the total data")
    elif valid_fraction < 1.0:
        valid_size = int(round(data_size * valid_fraction, 0))
        train_size = data_size - valid_size
    else:
        valid_size = data_size
        train_size = data_size

    site_sizes = split_num_proportion(train_size, num_sites, split_method)
    split_data_indices = {
        "valid": {"start": 0, "end": valid_size},
    }
    for site in range(num_sites):
        site_id = site_name_prefix + str(site + 1)
        if valid_fraction < 1.0:
            idx_start = valid_size + sum(site_sizes[:site])
            idx_end = valid_size + sum(site_sizes[: site + 1])
        else:
            idx_start = sum(site_sizes[:site])
            idx_end = sum(site_sizes[: site + 1])
        split_data_indices[site_id] = {"start": idx_start, "end": idx_end}

    return split_data_indices


def get_file_line_count(input_path: str) -> int:
    count = 0
    with open(input_path, "r") as fp:
        for i, _ in enumerate(fp):
            count += 1
    return count


def split_data(
    data_path: str,
    site_num: int,
    data_size: int,
    valid_frac: float,
    site_name_prefix: str = "site-",
    split_method: SplitMethod = SplitMethod.UNIFORM,
):
    size_total_file = get_file_line_count(data_path)
    if data_size > 0:
        if data_size > size_total_file:
            raise ValueError("data_size should be less than or equal to the true data size")
        else:
            size_total = data_size
    else:
        size_total = size_total_file
    site_indices = assign_data_index_to_sites(size_total, valid_frac, site_num, site_name_prefix, split_method)
    return site_indices


def _read_json(filename):
    if not os.path.isfile(filename):
        raise ValueError(f"{filename} does not exist!")
    with open(filename, "r") as f:
        return json.load(f)


def _write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def _get_job_name(args) -> str:
    return args.task_name + "_" + str(args.site_num) + "_" + args.split_method + "_" + args.backend


def _gen_deploy_map(num_sites: int, site_name_prefix: str) -> dict:
    deploy_map = {"app_server": ["server"]}
    for i in range(1, num_sites + 1):
        deploy_map[f"app_{site_name_prefix}{i}"] = [f"{site_name_prefix}{i}"]
    return deploy_map


def _update_meta(meta: dict, args):
    name = _get_job_name(args)
    meta["name"] = name
    meta["deploy_map"] = _gen_deploy_map(args.site_num, args.site_name_prefix)
    meta["min_clients"] = args.site_num


def _update_client_config(config: dict, args, site_name: str, site_indices):
    # update client config
    # data path and training/validation row indices
    config["components"][0]["args"]["backend"] = args.backend
    config["components"][0]["args"]["data_path"] = args.data_path
    config["components"][0]["args"]["train_start"] = site_indices[site_name]["start"]
    config["components"][0]["args"]["train_end"] = site_indices[site_name]["end"]
    config["components"][0]["args"]["valid_start"] = site_indices["valid"]["start"]
    config["components"][0]["args"]["valid_end"] = site_indices["valid"]["end"]


def _update_server_config(config: dict, args):
    config["min_clients"] = args.site_num


def _copy_custom_files(src_job_path, src_app_name, dst_job_path, dst_app_name):
    dst_path = dst_job_path / dst_app_name / "custom"
    os.makedirs(dst_path, exist_ok=True)
    src_path = src_job_path / src_app_name / "custom"
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


def create_server_app(src_job_path, src_app_name, dst_job_path, site_name, args):
    dst_app_name = f"app_{site_name}"
    server_config = _read_json(src_job_path / src_app_name / "config" / JobConstants.SERVER_JOB_CONFIG)
    dst_config_path = dst_job_path / dst_app_name / "config"

    # make target config folders
    if not os.path.exists(dst_config_path):
        os.makedirs(dst_config_path)

    _update_server_config(server_config, args)
    server_config_filename = dst_config_path / JobConstants.SERVER_JOB_CONFIG
    _write_json(server_config, server_config_filename)

    # copy custom file
    _copy_custom_files(src_job_path, src_app_name, dst_job_path, dst_app_name)


def create_client_app(src_job_path, src_app_name, dst_job_path, site_name, site_indices, args):
    dst_app_name = f"app_{site_name}"
    client_config = _read_json(src_job_path / src_app_name / "config" / JobConstants.CLIENT_JOB_CONFIG)
    dst_config_path = dst_job_path / dst_app_name / "config"

    # make target config folders
    if not os.path.exists(dst_config_path):
        os.makedirs(dst_config_path)

    # adjust file contents according to each job's specs
    _update_client_config(client_config, args, site_name, site_indices)
    client_config_filename = dst_config_path / JobConstants.CLIENT_JOB_CONFIG
    _write_json(client_config, client_config_filename)

    # copy custom file
    _copy_custom_files(src_job_path, src_app_name, dst_job_path, dst_app_name)


def main():
    parser = job_config_args_parser()
    args = parser.parse_args()
    job_name = _get_job_name(args)
    src_name = args.task_name + "_base"
    src_job_path = pathlib.Path(JOBS_ROOT) / src_name

    # create a new job
    dst_job_path = pathlib.Path(JOBS_ROOT) / job_name
    if not os.path.exists(dst_job_path):
        os.makedirs(dst_job_path)

    # update meta
    meta_config_dst = dst_job_path / JobConstants.META_FILE
    meta_config = _read_json(src_job_path / JobConstants.META_FILE)
    _update_meta(meta_config, args)
    _write_json(meta_config, meta_config_dst)

    # create server side app
    create_server_app(
        src_job_path=src_job_path,
        src_app_name="app",
        dst_job_path=dst_job_path,
        site_name="server",
        args=args,
    )

    # generate data split
    site_indices = split_data(
        args.data_path,
        args.site_num,
        args.data_size,
        args.valid_frac,
        args.site_name_prefix,
    )

    # create client side app
    for i in range(1, args.site_num + 1):
        create_client_app(
            src_job_path=src_job_path,
            src_app_name="app",
            dst_job_path=dst_job_path,
            site_name=f"{args.site_name_prefix}{i}",
            site_indices=site_indices,
            args=args,
        )


if __name__ == "__main__":
    main()
