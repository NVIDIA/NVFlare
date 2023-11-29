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

from nvflare.tool.job.config.configer import (
    build_config_file_indices,
    convert_to_number,
    get_app_name_from_path,
    get_cli_config,
    get_config_file_path,
    merge_configs,
    split_array_key,
)
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
            ["config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=false"],
            ["meta.conf", "min_clients=3"],
        ],
        "launch_everytime",
        [["app/config_fed_client.conf"], ["meta.conf"]],
    ),
    (
        "launch_once",
        [
            ["app/config/config_fed_client.conf", "app_script=cifar10_fl.py", "launch_once=False"],
            ["meta.conf", "min_clients=3"],
        ],
        "launch_everytime",
        [["app/config_fed_client.conf"], ["meta.conf"]],
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

GET_CONFIG_FILE_PATH_TEST_CASES = [
    ("meta.conf", "meta.conf"),
    ("config_fed_client.conf", "app/config/config_fed_client.conf"),
    ("app/config/config_fed_client.conf", "app/config/config_fed_client.conf"),
    ("app/custom/config.yaml", "app/custom/config.yaml"),
    ("app/custom/config.yml", "app/custom/config.yml"),
    ("app/custom/code/config.yml", "app/custom/code/config.yml"),
]


def _create_test_args(config_file: List, job_name: str = "launch_once"):
    args = argparse.Namespace()
    args.config_file = config_file
    args.debug = False
    args.force = True
    dir_name = os.path.join(os.getcwd(), os.path.dirname(__file__))
    args.job_folder = os.path.realpath(os.path.join(dir_name, f"../../../data/jobs/{job_name}"))
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
        updated_expected = {}
        for app in expected:
            app_expected = {}
            for file in expected.get(app):
                basename = os.path.basename(file)
                if basename == "meta.conf":
                    full_path = os.path.realpath(os.path.join(args.job_folder, basename))
                else:
                    full_path = os.path.realpath(os.path.join(args.job_folder, "app/config", basename))
                app_expected[full_path] = expected.get(app).get(file)
            updated_expected[app] = app_expected

        assert result == updated_expected

    @pytest.mark.parametrize("origin_job, origin_config, expect_job, expect_config", MERGE_CONFIG_TEST_CASES)
    def test_merge_configs(self, origin_job, origin_config, expect_job, expect_config):
        args = _create_test_args(
            config_file=origin_config,
            job_name=origin_job,
        )
        result_merged = _get_merged_configs(args)

        expected_args = _create_test_args(config_file=expect_config, job_name=expect_job)
        expected_merged = _get_merged_configs(expected_args)

        result = {}
        for app in result_merged:
            app_result = {}
            result_file_config = result_merged.get(app)
            for file in result_file_config:
                rel_file_path = os.path.relpath(file, args.job_folder)
                app_result[rel_file_path] = result_file_config.get(file)
            result[app] = app_result

        expected = {}
        for app in expected_merged:
            app_expected = {}
            expected_file_config = expected_merged.get(app)
            for file in expected_file_config:
                rel_file_path = os.path.relpath(file, expected_args.job_folder)
                app_expected[rel_file_path] = expected_file_config.get(file)
                assert app_expected[rel_file_path] == result[app][rel_file_path]
            expected[app] = app_expected

        assert result == expected

    def test_add_and_remove_config_keys(self):
        # remove config executors[0].executor.args.training = true
        # add config executors[0].executor.args.evaluation = true
        config_file = [
            [
                "config_fed_client.conf",
                "executors[0].executor.args.training-",
                "executors[0].executor.args.train_with_evaluation=true",
            ]
        ]
        args = _create_test_args(
            config_file=config_file,
            job_name="launch_once",
        )
        result_merged = _get_merged_configs(args)
        file_merged = result_merged.get("app")
        file, merged = list(file_merged.items())[0]
        config, excluded_key_list, key_indices = merged

        assert config.get("executors")[0].get("executor.args.training", None) is None
        assert key_indices.get("training", None) is None

        assert config.get("executors")[0].get("executor.args.train_with_evaluation", None) == "true"
        assert key_indices.get("train_with_evaluation", None) is not None

    def test_split_key(self):
        assert split_array_key("components[1].args.model.path") == ("components", 1, "args.model.path")
        assert split_array_key("args.model.path") == (None, None, "args.model.path")
        with pytest.raises(ValueError):
            split_array_key("components[].args.model.path")

        with pytest.raises(ValueError):
            split_array_key("components1].args.model.path")

        with pytest.raises(ValueError):
            split_array_key("components[1.args.model.path")

    def test_get_app_name_from_path(self):
        assert get_app_name_from_path("a.conf") == DEFAULT_APP_NAME
        assert get_app_name_from_path("app_1/a.conf") == "app_1"
        assert get_app_name_from_path("app_server/bbb.conf") == "app_server"
        assert get_app_name_from_path("app_server/custom/bbb.conf") == "app_server"
        with pytest.raises(ValueError):
            get_app_name_from_path("/")
        with pytest.raises(ValueError):
            get_app_name_from_path("/app_1/a.conf")

    @pytest.mark.parametrize("input_file_path, expected_config_file_path", GET_CONFIG_FILE_PATH_TEST_CASES)
    def test_get_config_file_path(self, input_file_path, expected_config_file_path):
        job_folder = "/tmp/nvflare/job_folder"
        config_file_path = get_config_file_path(app_name="app", input_file_path=input_file_path, job_folder=job_folder)
        assert config_file_path == os.path.join(job_folder, expected_config_file_path)

        with pytest.raises(ValueError) as excinfo:
            config_file_path = get_config_file_path(
                app_name="app", input_file_path="/custom/my.conf", job_folder=job_folder
            )
        assert str(excinfo.value) == "invalid config_file, /custom/my.conf"

    def test_get_config_file_path2(self):
        job_folder = "/tmp/nvflare/job_folder"

        with pytest.raises(ValueError) as excinfo:
            config_file_path = get_config_file_path(
                app_name="app", input_file_path="/custom/my.conf", job_folder=job_folder
            )
        assert str(excinfo.value) == "invalid config_file, /custom/my.conf"

    def test_convert_to_number(self):
        text = "I am a str"
        assert text == convert_to_number(text)

        text = "1"
        assert 1 == convert_to_number(text)

        text = "1.0"
        assert 1.0 == convert_to_number(text)

        text = "0.1"
        assert 0.1 == convert_to_number(text)

        text = "0.01"
        assert 0.01 == convert_to_number(text)

        text = "0.0.1"
        assert "0.0.1" == convert_to_number(text)
