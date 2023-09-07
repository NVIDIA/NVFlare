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
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from pyhocon import ConfigTree

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.tool.job.config.config_indexer import KeyIndex, build_reverse_order_index


def merge_configs_from_cli(cmd_args) -> Tuple[Dict[str, tuple], bool]:
    indices: Dict[str, Tuple] = build_config_file_indices(cmd_args.job_folder)
    cli_config_dict: Dict[str, Dict[str, str]] = get_cli_config(cmd_args)
    config_modified = False
    if cli_config_dict:
        config_modified = True
        copy_app_config_file(cli_config_dict, cmd_args)
        return merge_configs(indices, cli_config_dict), config_modified
    else:
        return indices, config_modified


def copy_app_config_file(cli_config_dict, cmd_args):
    config_dir = os.path.join(cmd_args.job_folder, "app/config")
    for cli_config_file in cli_config_dict:
        base_config_filename = os.path.basename(cli_config_file)
        if base_config_filename.startswith("meta."):
            target_dir = cmd_args.job_folder
        else:
            target_dir = config_dir

        target_file = os.path.join(target_dir, base_config_filename)
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
    if not input_string.strip(" "):
        return result

    opening_bracket_index = input_string.find("[")
    closing_bracket_index = input_string.find("]")
    if opening_bracket_index > 0 and closing_bracket_index > 0:
        string_before = input_string[:opening_bracket_index]
        index = int(input_string[opening_bracket_index + 1 : closing_bracket_index])
        string_after = input_string[closing_bracket_index + 1 :].strip(". ")
        if string_after:
            r = (string_before.strip("."), index, extract_string_with_index(string_after.strip(".")))
            if r:
                result.append(r)
        else:
            r = (string_before.strip("."), index, string_after)
            result.append(r)
    else:
        result.append(input_string)

    result = [elm for elm in result if len(elm) > 0]
    return result


def filter_indices(indices_configs: Dict[str, Tuple]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for file, (config, excluded_key_list, key_indices) in indices_configs.items():
        result[file] = filter_config_name_and_values(excluded_key_list, key_indices)

    return result


def filter_config_name_and_values(excluded_key_list, key_indices):
    temp_results = {}
    for key, key_index_list in key_indices.items():
        for key_index in key_index_list:
            if key not in excluded_key_list and key_index.value not in excluded_key_list:
                temp_results[key] = key_index

    return temp_results


def merge_configs(indices_configs: Dict[str, tuple], cli_file_configs: Dict[str, Dict]) -> Dict[str, tuple]:
    """
    Merge configurations from indices_configs and cli_file_configs.

    Args:
        indices_configs (Dict[str, tuple]): A dictionary containing indices and configurations.
        cli_file_configs (Dict[str, Dict]): A dictionary containing CLI configurations.

    Returns:
        Dict[str, tuple]: A dictionary containing merged configurations.
    """
    merged = {}
    for file, (config, excluded_key_list, key_indices) in indices_configs.items():
        basename = os.path.basename(file)
        if len(key_indices) > 0:
            # CLI could be use absolute path as well, try that first, not found, then use base name
            cli_configs = cli_file_configs.get(file, None)
            if not cli_configs:
                cli_configs = cli_file_configs.get(basename, None)

            if cli_configs:
                for key, cli_value in cli_configs.items():
                    if key not in key_indices:
                        # not every client has app_config, app_script
                        if key not in ["app_script", "app_config"]:
                            raise ValueError(f"Invalid config key: '{key}' for file '{file}'")
                    else:
                        indices = key_indices.get(key)
                        for key_index in indices:
                            value_type = type(key_index.value)
                            new_value = value_type(cli_value) if key_index.value is not None else cli_value
                            key_index.value = new_value
                            parent_key = key_index.parent_key
                            if parent_key and isinstance(parent_key.value, ConfigTree):
                                parent_key.value.put(key_index.key, new_value)

        merged[basename] = (config, excluded_key_list, key_indices)

    return merged


def get_root_index(key_index: KeyIndex) -> Optional[KeyIndex]:
    if key_index is None or key_index.parent_key is None:
        return key_index

    if key_index.parent_key is not None:
        if key_index.parent_key.parent_key is None or key_index.parent_key.parent_key.key == "":
            return key_index.parent_key
        else:
            return get_root_index(key_index.parent_key)

    return None


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
        cli_config_dict = parse_cli_config(cli_configs)

    if "script" in cmd_args and cmd_args.script:
        script = os.path.basename(cmd_args.script)
        if "config_fed_client.conf" in cli_config_dict:
            cli_config_dict["config_fed_client.conf"].update({"app_script": script})
        else:
            cli_config_dict["config_fed_client.conf"] = {"app_script": script}

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


def build_config_file_indices(config_dir: str) -> Dict[str, Tuple]:
    excluded = ["info"]
    included = ["config_fed_client", "config_fed_server", "config_exchange", "meta"]
    config_extensions = ConfigFormat.extensions()

    config_file_index = {}
    config_files = []
    for root, _, files in os.walk(config_dir):
        for f in files:
            tokens = os.path.splitext(f)
            name_wo_ext = tokens[0]
            ext = tokens[1]
            if (
                ext in config_extensions
                and not f.startswith("._")
                and name_wo_ext in included
                and name_wo_ext not in excluded
            ):
                config_files.append(f)
        for f in config_files:
            f = str(os.path.abspath(os.path.join(root, f)))
            if os.path.isfile(f):
                real_path, config, excluded_key_list, key_indices = build_reverse_order_index(str(f))
                config_file_index[real_path] = (config, excluded_key_list, key_indices)

    return config_file_index
