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
import os
from typing import List

from nvflare.tool.job.config.configer import DEFAULT_APP_NAME, build_config_file_indices, get_cli_config, merge_configs


def _create_test_args(config_file: List, job_name: str = "launch_once"):
    args = argparse.Namespace()
    args.config_file = config_file
    args.debug = False
    args.force = True
    args.job_folder = os.path.join(os.path.dirname(__file__), f"../../../data/jobs/{job_name}")
    args.job_sub_cmd = "create"
    args.sub_command = "job"
    return args


def _get_merged_configs(args):
    cli_configs = get_cli_config(args, [DEFAULT_APP_NAME])
    app_indices = build_config_file_indices(args.job_folder, [DEFAULT_APP_NAME])
    merged_configs = merge_configs(app_indices_configs=app_indices, app_cli_file_configs=cli_configs)
    return merged_configs


class TestConfiger:
    def test_get_cli_config(self):
        args = _create_test_args(
            config_file=[["config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=False"]],
            job_name="launch_once",
        )
        result = get_cli_config(args, [DEFAULT_APP_NAME])
        assert result == {
            DEFAULT_APP_NAME: {"config_fed_client.conf": {"app_script": "cifar10_fl.py", "launch_once": "False"}}
        }

    def test_merge_configs(self):
        args = _create_test_args(
            config_file=[["config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=False"]],
            job_name="launch_once",
        )
        result_merged = _get_merged_configs(args)

        args = _create_test_args(config_file=[["config_fed_client.conf"]], job_name="launch_everytime")
        expected_merged = _get_merged_configs(args)

        assert result_merged == expected_merged

    def test_merge_configs_lower_case_false(self):
        args = _create_test_args(
            config_file=[["config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=false"]],
            job_name="launch_once",
        )
        result_merged = _get_merged_configs(args)

        args = _create_test_args(config_file=[["config_fed_client.conf"]], job_name="launch_everytime")
        expected_merged = _get_merged_configs(args)

        assert result_merged == expected_merged
