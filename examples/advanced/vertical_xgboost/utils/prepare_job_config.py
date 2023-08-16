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

from nvflare.apis.fl_constant import JobConstants, WorkspaceConstants

JOB_CONFIGS_ROOT = "jobs"
CONFIG_FOLDER_NAME = "config"


def job_config_args_parser():
    parser = argparse.ArgumentParser(description="Generate job configs for PSI and Vertical XGBoost")
    parser.add_argument("--job_name", type=str, default="vertical_xgb", help="job name")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/tmp/nvflare/vertical_xgb_data",
        help="Path to dataset files for each site",
    )
    parser.add_argument("--site_num", type=int, default=5, help="Total number of sites")
    parser.add_argument("--label_owner", type=int, default=1, help="Site that contains the label column")
    parser.add_argument("--site_name_prefix", type=str, default="site-", help="Site name prefix")
    parser.add_argument("--round_num", type=int, default=100, help="Total number of training rounds")

    return parser


def _read_json(filename):
    if not os.path.isfile(filename):
        raise ValueError(f"{filename} does not exist!")
    with open(filename, "r") as f:
        return json.load(f)


def _write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def _get_job_name(args) -> str:
    return args.job_name + "_" + str(args.site_num)


def _get_src_job_dir(args):
    return pathlib.Path(JOB_CONFIGS_ROOT) / (args.job_name + "_base")


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


def _update_client_config(config: dict, args, site_name: str):
    if args.job_name == "vertical_xgb_psi":
        config["components"][1]["args"]["data_split_path"] = config["components"][1]["args"]["data_split_path"].replace(
            "{site-x}", site_name
        )
    elif args.job_name == "vertical_xgb":
        config["components"][0]["args"]["data_split_path"] = config["components"][0]["args"]["data_split_path"].replace(
            "{site-x}", site_name
        )
        config["components"][0]["args"]["psi_path"] = config["components"][0]["args"]["psi_path"].replace(
            "{site-x}", site_name
        )
        config["components"][0]["args"]["label_owner"] = f"{args.site_name_prefix}{args.label_owner}"


def _copy_custom_files(src_job_path, src_app_name, dst_job_path, dst_app_name):
    dst_path = dst_job_path / dst_app_name / WorkspaceConstants.CUSTOM_FOLDER_NAME
    os.makedirs(dst_path, exist_ok=True)
    src_path = src_job_path / src_app_name / WorkspaceConstants.CUSTOM_FOLDER_NAME
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


def create_server_app(src_job_path, src_app_name, dst_job_path, site_name, args):
    dst_app_name = f"app_{site_name}"
    server_config = _read_json(src_job_path / src_app_name / CONFIG_FOLDER_NAME / JobConstants.SERVER_JOB_CONFIG)
    dst_config_path = dst_job_path / dst_app_name / CONFIG_FOLDER_NAME

    # make target config folders
    if not os.path.exists(dst_config_path):
        os.makedirs(dst_config_path)

    server_config_filename = dst_config_path / JobConstants.SERVER_JOB_CONFIG
    _write_json(server_config, server_config_filename)


def create_client_app(src_job_path, src_app_name, dst_job_path, site_name, args):
    dst_app_name = f"app_{site_name}"
    client_config = _read_json(src_job_path / src_app_name / CONFIG_FOLDER_NAME / JobConstants.CLIENT_JOB_CONFIG)
    dst_config_path = dst_job_path / dst_app_name / CONFIG_FOLDER_NAME

    # make target config folders
    if not os.path.exists(dst_config_path):
        os.makedirs(dst_config_path)

    # adjust file contents according to each job's specs
    _update_client_config(client_config, args, site_name)
    client_config_filename = dst_config_path / JobConstants.CLIENT_JOB_CONFIG
    _write_json(client_config, client_config_filename)

    # copy custom file
    _copy_custom_files(src_job_path, src_app_name, dst_job_path, dst_app_name)


def main():
    parser = job_config_args_parser()
    args = parser.parse_args()
    job_name = _get_job_name(args)
    src_job_path = _get_src_job_dir(args)

    # create a new job
    dst_job_path = pathlib.Path(JOB_CONFIGS_ROOT) / job_name
    if not os.path.exists(dst_job_path):
        os.makedirs(dst_job_path)

    # update meta
    meta_config_dst = dst_job_path / JobConstants.META_FILE
    meta_config = _read_json(src_job_path / JobConstants.META_FILE)
    _update_meta(meta_config, args)
    _write_json(meta_config, meta_config_dst)

    # create server side app
    create_server_app(
        src_job_path=src_job_path, src_app_name="app", dst_job_path=dst_job_path, site_name="server", args=args
    )

    # create client side app
    for i in range(1, args.site_num + 1):
        create_client_app(
            src_job_path=src_job_path,
            src_app_name="app",
            dst_job_path=dst_job_path,
            site_name=f"{args.site_name_prefix}{i}",
            args=args,
        )


if __name__ == "__main__":
    main()
