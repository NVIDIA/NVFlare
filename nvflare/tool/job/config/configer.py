# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import shutil
from typing import Any, Dict, List, Optional

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.tool.job.config.config_indexer import build_reverse_order_index


def merge_configs_from_cli(cmd_args) -> Dict[str, ConfigTree]:
    cli_config_dict: Dict[str, Dict[str, str]] = get_cli_config(cmd_args)
    copy_app_config_file(cli_config_dict, cmd_args)
    config_dir = os.path.join(cmd_args.job_folder, "app", "config")
    indices: Dict[str, (Dict, Dict)] = build_config_file_indexers(config_dir)
    return merge_configs(indices, cli_config_dict)


def copy_app_config_file(cli_config_dict, cmd_args):
    config_dir = os.path.join(cmd_args.job_folder, "app/config")
    for cli_config_file in cli_config_dict:
        print("cli_config_file =", cli_config_file)
        base_config_filename = os.path.basename(cli_config_file)
        target_file = os.path.join(config_dir, base_config_filename)
        if not os.path.exists(target_file):
            shutil.copyfile(cli_config_file, target_file)


def extract_string_with_index(input_string):
    """
    Extract the string before '[', the index within '[', and the string after ']'.

    Args:
        input_string (str): The input string containing the pattern '[index]'.

    Returns:
        list: A list of tuples containing the extracted components: (string_before, index, string_after).
    """
    result = []
    while True:
        opening_bracket_index = input_string.find("[")
        closing_bracket_index = input_string.find("]")

        if opening_bracket_index == -1 or closing_bracket_index == -1:
            break

        string_before = input_string[:opening_bracket_index]
        index = int(input_string[opening_bracket_index + 1: closing_bracket_index])
        string_after = input_string[closing_bracket_index + 1:]

        result.append((string_before.strip("."), index, string_after.strip(".")))
        input_string = f"{string_before}{string_after}"

    result = [elm for elm in result if len(elm) > 0]
    return result


def merge_configs(indices_configs: Dict[str, tuple], cli_file_configs: Dict[str, Dict]) -> Dict[str, ConfigTree]:
    """
    Merge configurations from indices_configs and cli_file_configs.

    Args:
        indices_configs (Dict[str, tuple]): A dictionary containing indices and configurations.
        cli_file_configs (Dict[str, Dict]): A dictionary containing CLI configurations.

    Returns:
        Dict[str, CF]: A dictionary containing merged configurations.
    """
    merged = {}
    for file, (indices_dict, configs_dict) in indices_configs.items():
        basename = os.path.basename(file)
        cli_configs = cli_file_configs.get(basename, None)
        if cli_configs:
            conf = CF.from_dict(configs_dict)
            if len(indices_dict) > 0:
                for key, value in cli_configs.items():
                    if key not in indices_dict:
                        raise ValueError(f"Invalid config key: '{key}' for file '{basename}'")

                    key_path_list = indices_dict[key]
                    if len(key_path_list) > 1:
                        raise ValueError(
                            f"Ambiguity config key: '{key}' for file '{basename}', "
                            f"more than one key paths with such key: {key_path_list}"
                        )

                    key_path = key_path_list[0]
                    results = extract_string_with_index(key_path)
                    if len(results) > 0:
                        target_conf = conf
                        for idx, (before, index, after) in enumerate(results):
                            before_configs = target_conf.get_list(before)
                            after_config = before_configs[index]
                            if idx == len(results) - 1:
                                value_type = type(after_config.get(after))
                                after_config.put(after, value_type(value))
                            else:
                                target_conf = after_config
                    else:
                        conf.put(key_path, value)

            merged[basename] = conf

    return merged


def get_cli_config(cmd_args: Any) -> Dict[str, Dict[str, str]]:
    """
    Extract configurations from command-line arguments and return them in a dictionary.

    Args:
        cmd_args: Command-line arguments containing configuration data.

    Returns:
        A dictionary containing the configurations extracted from the command-line arguments.
    """
    cli_config_dict = {}
    if cmd_args.config_file:
        cli_configs = cmd_args.config_file
        return parse_cli_config(cli_configs)
    return cli_config_dict


def parse_cli_config(cli_configs: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Extract configurations from command-line arguments and return them in a dictionary.

    Args:
        cli_configs: Array of CLI config option in the format of
           filename  key1=v1 key2=v2
        separated by space
    Returns:
        A dictionary containing the configurations extracted from the command-line arguments.
    """

    cli_config_dict = {}
    if cli_configs:
        for arr in cli_configs:
            config_file = os.path.basename(arr[0])
            config_data = arr[1:]
            config_dict = {}

            for conf in config_data:
                conf_key_value = conf.split("=")
                if len(conf_key_value) != 2:
                    raise ValueError(f"Invalid config data: {conf}")
                conf_key, conf_value = conf_key_value
                config_dict[conf_key] = conf_value
            cli_config_dict[config_file] = config_dict

    return cli_config_dict


def build_config_file_indexers(config_dir: str, excluded: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    Build a dictionary of config file indexers for the given job folder.

    Args:
        excluded: if not None, we will excluded from the file list
        config_dir:  Job config directory

    Returns:
        Dict[str, dict]: A dictionary where keys are absolute paths of config files
                         and values are their corresponding reverse order indexers.
    """
    config_extensions = ConfigFormat.extensions()

    config_file_index = {}
    for root, _, files in os.walk(config_dir):
        config_files = [f for f in files if os.path.splitext(f)[1] in config_extensions and not f.startswith("._")]
        if excluded:
            config_files = [f for f in config_files if f not in excluded]

        for f in config_files:
            f = str(os.path.abspath(os.path.join(root, f)))
            config_file_index[f] = build_reverse_order_index(f)

    return config_file_index
