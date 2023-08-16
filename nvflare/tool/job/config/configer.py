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
import inspect
import os
import shutil
from typing import Any, Dict, List, Tuple

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel_opt.utils.pyhocon_loader import PyhoconConfig
from nvflare.tool.job.config.config_indexer import build_reverse_order_index, KeyIndex


def merge_configs_from_cli(cmd_args) -> Dict[str, tuple]:
    cli_config_dict: Dict[str, Dict[str, str]] = get_cli_config(cmd_args)
    copy_app_config_file(cli_config_dict, cmd_args)
    indices: Dict[str, Tuple] = build_config_file_indices(cmd_args.job_folder)
    return merge_configs(indices, cli_config_dict)


def copy_app_config_file(cli_config_dict, cmd_args):
    config_dir = os.path.join(cmd_args.job_folder, "app/config")
    target_dir = config_dir
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
        index = int(input_string[opening_bracket_index + 1: closing_bracket_index])
        string_after = input_string[closing_bracket_index + 1:].strip(". ")
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


def extract_value_from_index(indices_configs: Dict[str, Tuple]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for file, (excluded_key_list, key_indices) in indices_configs.items():
        result[file] = extract_value_from_dict_by_index(excluded_key_list, key_indices)

    return result


def extract_value_from_dict_by_index(excluded_key_list, key_indices):
    temp_results = {}
    for key, key_index in key_indices.items():
        value = key_index.value
        if key == "path":
            last_dot_index = value.rindex(".")
            class_path = value[:last_dot_index]
            class_name = value[last_dot_index + 1:]
            module, import_flag = optional_import(module=class_path, name=class_name)
            if import_flag:
                params = inspect.signature(module.__init__).parameters
                for v in params.values():
                    if v.name != "self" and \
                            v.default is not None and \
                            v.name not in excluded_key_list and \
                            v.default not in excluded_key_list:
                        if isinstance(v.default, str):
                            if len(v.default) > 0:
                                temp_results[v.name] = \
                                    KeyIndex(key=v.name,
                                             value=v.default,
                                             parent_key=key_index.parent_key,
                                             component_name=key_index.component_name)
                        else:
                            temp_results[v.name] = \
                                KeyIndex(key=v.name,
                                         value=v.default,
                                         parent_key=key_index.parent_key,
                                         component_name=key_index.component_name)

    for key, key_index in key_indices.items():
        if key not in excluded_key_list and key_index.value not in excluded_key_list:
            temp_results[key] = key_index

    return temp_results


def extract_value_from_list_index(conf, input_str):
    target_conf = conf

    if isinstance(input_str, Tuple):
        before, index, after = input_str
        before_configs = target_conf.get_list(before)
        after_config = before_configs[index]
        if after:
            if isinstance(after, List):
                return extract_value_from_list_index(after_config, after[0])
            else:
                if isinstance(after_config, ConfigTree):
                    return after_config.get(after)
                else:
                    return after_config
        else:
            return after_config
    else:
        return conf.get(input_str)


def replace_value_from_list_index(conf, input_str, new_value):
    if isinstance(input_str, Tuple):
        before, index, after = input_str
        before_configs = conf.get_list(before)
        after_config = before_configs[index]
        if after:
            if isinstance(after, List):
                replace_value_from_list_index(after_config, after[0], new_value)
            else:
                if isinstance(after_config, ConfigTree):
                    old_value_type = type(after_config.get(after))
                    after_config.put(after, old_value_type(new_value))
                else:
                    # this kind of like this: a = [1, 2], a[0] -> 1, a[1] = 2
                    # unlike be able to specify this in CLI.
                    raise ValueError(
                        f"unable to substitute value {new_value} for {input_str}," f" suggest direct edit config file"
                    )
        else:
            # this kind of like this: a = [1, 2], a[0] -> 1, a[1] = 2
            # unlike be able to specify this in CLI.
            raise ValueError(
                f"unable to substitute value {new_value} for {input_str}," f" suggest direct edit config file"
            )

    else:
        old_value_type = type(conf.get(input_str))
        conf.put(input_str, old_value_type(new_value))


def merge_configs(indices_configs: Dict[str, tuple], cli_file_configs: Dict[str, Dict]) -> Dict[str, tuple]:
    """
    Merge configurations from indices_configs and cli_file_configs.

    Args:
        indices_configs (Dict[str, tuple]): A dictionary containing indices and configurations.
        cli_file_configs (Dict[str, Dict]): A dictionary containing CLI configurations.

    Returns:
        Dict[str, CF]: A dictionary containing merged configurations.
    """
    merged = {}
    for file, (excluded_key_list, key_indices) in indices_configs.items():
        # TODO: actually merged
        basename = os.path.basename(file)
        # conf = CF.from_dict(configs_dict)
        # if len(indices_dict) > 0:
        #     # CLI could be use absolute path as well, try that first, not found, then use base name
        #     cli_configs = cli_file_configs.get(file, None)
        #     if not cli_configs:
        #         cli_configs = cli_file_configs.get(basename, None)
        #
        #     if cli_configs:
        #         for key, value in cli_configs.items():
        #             if key not in indices_dict:
        #                 raise ValueError(f"Invalid config key: '{key}' for file '{basename}'")
        #
        #             key_path_list = indices_dict[key]
        #             for key_path in key_path_list:
        #                 # results = extract_string_with_index(key_path)
        #                 tokens = extract_string_with_index(key_path)
        #                 if isinstance(tokens, List) and tokens:
        #                     replace_value_from_list_index(conf, tokens[0], value)
        #                 else:
        #                     conf.put(key_path, value)
        # todo: do merge
        merged[basename] = (excluded_key_list, key_indices)

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
                real_path, excluded_key_list, key_indices = build_reverse_order_index(str(f))
                config_file_index[real_path] = (excluded_key_list, key_indices)

    return config_file_index
