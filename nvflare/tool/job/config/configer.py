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
from typing import Any, Dict, List, Optional, Tuple

from pyhocon import ConfigFactory, ConfigTree

from nvflare.fuel.utils.config import ConfigFormat
from nvflare.tool.job.config.config_indexer import KeyIndex, build_reverse_order_index
from nvflare.tool.job.job_client_const import (
    APP_CONFIG_DIR,
    APP_CONFIG_FILE_BASE_NAMES,
    APP_CONFIG_KEY,
    APP_SCRIPT_KEY,
    CONFIG_FED_CLIENT_CONF,
    DEFAULT_APP_NAME,
    JOB_META_BASE_NAME,
    META_APP_NAME,
)


def merge_configs_from_cli(cmd_args, app_names: List[str]) -> Tuple[Dict[str, Dict[str, tuple]], bool]:
    app_indices: Dict[str, Dict[str, Tuple]] = build_config_file_indices(cmd_args.job_folder, app_names)

    app_cli_config_dict: Dict[str, Dict[str, Dict[str, str]]] = get_cli_config(cmd_args, app_names)
    config_modified = False
    if app_cli_config_dict:
        config_modified = True
        return merge_configs(app_indices, app_cli_config_dict), config_modified
    else:
        return app_indices, config_modified


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


def filter_indices(app_indices_configs: Dict[str, Dict[str, Tuple]]) -> Dict[str, Dict[str, Dict[str, KeyIndex]]]:
    app_results = {}
    for app_name in app_indices_configs:
        indices_configs = app_indices_configs.get(app_name)
        result = {}
        for file, (config, excluded_key_list, key_indices) in indices_configs.items():
            result[file] = filter_config_name_and_values(excluded_key_list, key_indices)

        app_results[app_name] = result

    return app_results


def filter_config_name_and_values(
    excluded_key_list: List[str], key_indices: Dict[str, List[KeyIndex]]
) -> Dict[str, KeyIndex]:
    temp_results = {}
    for key, key_index_list in key_indices.items():
        for key_index in key_index_list:
            if key not in excluded_key_list and key_index.value not in excluded_key_list:
                # duplicated key will be over-written by last one
                temp_results[key] = key_index

    return temp_results


def _cast_type(key_index, cli_value):
    """Casts cli_value to correct type.

    Since build_reverse_order_index is using pyhocon, we need to do the same here.
    """
    if key_index.value is None:
        return cli_value

    new_value = ConfigFactory.parse_string(f"{key_index.key}={cli_value}")[key_index.key]
    return new_value


def merge_configs(
    app_indices_configs: Dict[str, Dict[str, tuple]], app_cli_file_configs: Dict[str, Dict[str, Dict]]
) -> Dict[str, Dict[str, tuple]]:
    """Merges configurations from indices_configs and cli_file_configs.

    Args:
        app_indices_configs (Dict[str, Dict[str, tuple]]): A dictionary containing indices and configurations.
        app_cli_file_configs (Dict[str, Dict[str, Dict]]): A dictionary containing CLI configurations.

    Returns:
        Dict[str, Dict[str, Tuple]]: A dictionary of {app_name: merged configurations}.
            Each of the merged configurations can be expressed in a Tuple: config, excluded_key_List, key_indices
    """
    app_merged = {}
    for app_name in app_indices_configs:
        indices_configs = app_indices_configs[app_name]

        cli_file_configs = app_cli_file_configs.get(app_name, None)
        if cli_file_configs:
            merged = {}
            for file, (config, excluded_key_list, key_indices) in indices_configs.items():
                basename = os.path.basename(file)
                if len(key_indices) > 0:
                    cli_configs = cli_file_configs.get(basename, None)
                    if cli_configs:
                        for key, cli_value in cli_configs.items():
                            if key not in key_indices:
                                # not every client has app_config, app_script
                                if key not in [APP_SCRIPT_KEY, APP_CONFIG_KEY]:
                                    raise ValueError(f"Invalid config key: '{key}' for file '{file}'")
                            else:
                                indices = key_indices.get(key)
                                for key_index in indices:
                                    new_value = _cast_type(key_index, cli_value)
                                    key_index.value = new_value
                                    parent_key = key_index.parent_key
                                    if parent_key and isinstance(parent_key.value, ConfigTree):
                                        parent_key.value.put(key_index.key, new_value)

                merged[basename] = (config, excluded_key_list, key_indices)
            app_merged[app_name] = merged
        elif app_name == META_APP_NAME:
            new_indices_configs = {}
            for k, v in indices_configs.items():
                new_indices_configs[os.path.basename(k)] = v
            app_merged[app_name] = new_indices_configs
        else:
            app_merged[app_name] = indices_configs

    return app_merged


def get_root_index(key_index: KeyIndex) -> Optional[KeyIndex]:
    if key_index is None or key_index.parent_key is None:
        return key_index

    if key_index.parent_key is not None:
        if key_index.parent_key.parent_key is None or key_index.parent_key.parent_key.key == "":
            return key_index.parent_key
        else:
            return get_root_index(key_index.parent_key)

    return None


def get_cli_config(cmd_args: Any, app_names: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Extracts configurations from command-line arguments and return them in a dictionary.

    Args:
        app_names: application names
        cmd_args: Command-line arguments containing configuration data.

    Returns:
        A dictionary containing the configurations extracted from the command-line arguments.
    """
    app_cli_config_dict = {}
    if cmd_args.config_file:
        cli_configs = cmd_args.config_file
        app_cli_config_dict = _parse_cli_config(cli_configs, app_names)

    # replace "script"
    if "script" in cmd_args and cmd_args.script:
        script = os.path.basename(cmd_args.script)

        if app_cli_config_dict:
            key = CONFIG_FED_CLIENT_CONF
            for _, cli_config_dict in app_cli_config_dict.items():

                if key in cli_config_dict:
                    cli_config_dict[key].update({APP_SCRIPT_KEY: script})
                else:
                    cli_config_dict[key] = {APP_SCRIPT_KEY: script}
        else:
            app_cli_config_dict = {DEFAULT_APP_NAME: {CONFIG_FED_CLIENT_CONF: {APP_SCRIPT_KEY: script}}}

    return app_cli_config_dict


def _is_meta_file(filename: str) -> bool:
    for postfix in ConfigFormat.extensions():
        if filename == f"{JOB_META_BASE_NAME}{postfix}":
            return True
    return False


def _parse_cli_config(cli_configs: List[str], app_names: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Extracts configurations from command-line arguments and return them in a dictionary.

    Args:
        app_names: application names
        cli_configs: Array of CLI config option in the format of
           -f filename.conf  key1=v1 key2=v2
           or
           -f app/filename.conf  key1=v1 key2=v2
        separated by space
    Returns:
        A dictionary containing the configurations extracted from the command-line arguments.
    """

    app_cli_config_dict = {}
    if cli_configs:
        for arr in cli_configs:
            config_file = os.path.basename(arr[0])
            app_name = get_app_name_from_path(arr[0])
            config_data = arr[1:]
            config_dict = {}
            app_name = DEFAULT_APP_NAME if not app_name else app_name

            if app_name not in app_names and app_name != DEFAULT_APP_NAME and app_name != META_APP_NAME:
                raise ValueError(
                    f"Please specify one of the app names {app_names}. For example '<app_name>/xxx.conf k1=v1 k2=v2...'"
                )

            for conf in config_data:
                index = conf.find("=")
                if index == -1:
                    raise ValueError("Invalid config data, expecting key, value pair in the format key=value")
                conf_key = conf[0:index]
                conf_value = conf[index + 1 :]
                config_dict[conf_key] = conf_value

            if app_name not in app_cli_config_dict:
                app_cli_config_dict[app_name] = {}
            app_cli_config_dict[app_name][config_file] = config_dict

    return app_cli_config_dict


def build_config_file_indices(job_folder: str, app_names: List[str]) -> Dict[str, Dict[str, Tuple]]:
    config_included = APP_CONFIG_FILE_BASE_NAMES
    meta_base = JOB_META_BASE_NAME
    config_extensions = ConfigFormat.extensions()

    app_config_file_index = {}
    app_config_files = {}

    for ext in config_extensions:
        meta_file = os.path.join(job_folder, f"{meta_base}{ext}")
        if os.path.isfile(meta_file):
            app_config_files[META_APP_NAME] = [meta_file]
            break

    for app_name in app_names:
        app_dir = os.path.join(job_folder, app_name)
        for ext in config_extensions:
            for base in config_included:
                file = os.path.abspath(os.path.join(app_dir, APP_CONFIG_DIR, f"{base}{ext}"))
                if os.path.isfile(file):
                    config_files = app_config_files.get(app_name, [])
                    config_files.append(file)
                    app_config_files[app_name] = config_files

    for app_name, config_files in app_config_files.items():
        for f in config_files:
            real_path, config, excluded_key_list, key_indices = build_reverse_order_index(str(f))
            config_file_index = app_config_file_index.get(app_name, {})
            config_file_index[real_path] = (config, excluded_key_list, key_indices)
            app_config_file_index[app_name] = config_file_index

    return app_config_file_index


def get_app_name_from_path(path: str):
    # path is in the format of as app1/xxx.conf
    # path app1/xxx.conf
    # xxx.conf
    if _is_meta_file(os.path.basename(path)):
        return META_APP_NAME
    app_name = os.path.dirname(path)
    index = app_name.find("/")
    if index > 0:
        raise ValueError(f"Expecting <app_name>/<config file>, but '{path}' is given.")
    return app_name if app_name else DEFAULT_APP_NAME
