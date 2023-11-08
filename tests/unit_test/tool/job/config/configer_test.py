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

import pytest

from nvflare.tool.job.config.configer import build_config_file_indices, get_cli_config, merge_configs, split_array_key
from nvflare.tool.job.job_client_const import DEFAULT_APP_NAME, META_APP_NAME

MERGE_CONFIG_TEST_CASES = [
    (
        "launch_once",
        [
            ["app/config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=False"],
            ["meta.conf", "min_clients=3"],
        ],
        "launch_everytime",
        [["app/config_fed_client.conf"], ["meta.conf"]],
    ),
    (
        "launch_once",
        [
            ["app/config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=false"],
            ["meta.conf", "min_clients=3"],
        ],
        "launch_everytime",
        [["app/config_fed_client.conf"], ["meta.conf"]],
    ),
    (
        "launch_once",
        [["app/config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=False"]],
        "launch_everytime",
        [["app/config_fed_client.conf"], ["meta.conf", "min_clients=2"]],
    ),
    (
        "launch_once",
        [["app/config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=false"]],
        "launch_everytime",
        [["app/config_fed_client.conf"], ["meta.conf", "min_clients=2"]],
    ),
]

GET_CLI_USE_CASES = [
    (
        "launch_once",
        [["config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=False"]],
        {DEFAULT_APP_NAME: {"config_fed_client.conf": {"app_script": "cifar10_fl.py", "launch_once": "False"}}},
    ),
    ("launch_once", [["meta.conf", "min_clients=3"]], {META_APP_NAME: {"meta.conf": {"min_clients": "3"}}}),
]


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
    @pytest.mark.parametrize("job_name, config_file, expected", GET_CLI_USE_CASES)
    def test_get_cli_config(self, job_name, config_file, expected):
        args = _create_test_args(
            config_file=config_file,
            job_name=job_name,
        )
        result = get_cli_config(args, [DEFAULT_APP_NAME])
        assert result == expected

    @pytest.mark.parametrize("origin_job, origin_config, expect_job, expect_config", MERGE_CONFIG_TEST_CASES)
    def test_merge_configs(self, origin_job, origin_config, expect_job, expect_config):
        args = _create_test_args(
            config_file=origin_config,
            job_name=origin_job,
        )
        result_merged = _get_merged_configs(args)

        args = _create_test_args(config_file=expect_config, job_name=expect_job)
        expected_merged = _get_merged_configs(args)

        assert result_merged == expected_merged

    def test_split_key(self):
        assert split_array_key("components[1].args.model.path") == ("components", 1, "args.model.path")
        assert split_array_key("args.model.path") == (None, None, "args.model.path")
        try:
            assert split_array_key("components1].args.model.path")
        except ValueError:
            assert True
        try:
            assert split_array_key("components[1.args.model.path")
        except ValueError:
            assert True
