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
from typing import Any, Dict, List, Optional, Tuple

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree

from nvflare.fuel.utils.config import Config, ConfigFormat
from nvflare.fuel.utils.config_factory import ConfigFactory


def build_reverse_order_index(config_file_path: str) -> Tuple[str, Dict[str, List[str]], Dict[str, Any]]:
    config, config_file_path = load_pyhocon_conf(config_file_path)

    components: list = config.get("components", None)
    excluded_list = [comp.get("id") for comp in components] if components else []
    excluded_list.extend(
        [
            "name",
            "path",
            "id",
            "format_version",
            "tasks",
            "task_name",
            "train_task_name",
            "task_data_filters",
            "task_result_filters",
            "exchange_path",
            "job_folder_name",
        ]
    )
    indices: Dict[str, List[str]] = build_dict_reverse_order_index(config, excluded_keys=excluded_list)
    return config_file_path, indices, config


def load_pyhocon_conf(config_file_path):
    try:
        temp_conf: Config = ConfigFactory.load_config(config_file_path)
        if temp_conf:
            config_file_path = temp_conf.file_path
            if temp_conf.format == ConfigFormat.PYHOCON:
                config: ConfigTree = temp_conf.conf
            else:
                config: ConfigTree = CF.from_dict(temp_conf.to_dict())
        else:
            raise ValueError(f"Config is None for file:'{config_file_path}'.")

    except Exception as e:
        raise RuntimeError(f"filed to parse file {config_file_path}:", e)
    return config, config_file_path


def build_list_reverse_order_index(
    config_list: List, key_path_dict: Dict, key: str, root_path: str, excluded_keys: Optional[List[str]] = None
) -> Dict:
    """
    Recursively build a reverse order index for a list.
    """
    if excluded_keys is None:
        excluded_keys = []
    result = key_path_dict.get(root_path, [])
    for index, value in enumerate(config_list):
        key_path = f"{root_path}[{index}]"
        key_with_index = f"{key}[{index}]"
        if isinstance(value, list):
            if len(value) > 0:
                build_list_reverse_order_index(
                    value, key_path_dict, key_with_index, root_path=key_path, excluded_keys=excluded_keys
                )
            else:
                result.append(key_path)
                key_path_dict[key_with_index] = result
        elif isinstance(value, dict):
            build_dict_reverse_order_index(value, key_path_dict, root_path=key_path, excluded_keys=excluded_keys)
        elif is_primitive(value):
            if not (key == "path" and key_path.startswith("components")):
                result.append(key_path)
                key_path_dict[key_with_index] = result
        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")
    return key_path_dict


def is_primitive(value):
    return isinstance(value, int) or isinstance(value, float) or isinstance(value, str) or isinstance(value, bool)


def has_no_primitives_in_list(values: List):
    return any(not is_primitive(x) for x in values)


def build_dict_reverse_order_index(
    config_dict: Dict, key_path_dict: Optional[Dict] = None, root_path: str = "", excluded_keys: List[str] = None
) -> Dict[str, List[str]]:
    """
    Recursively build a reverse order index for a dictionary.
    """
    if excluded_keys is None:
        excluded_keys = []

    if key_path_dict is None:
        key_path_dict = {}

    for key, value in config_dict.items():
        if key in excluded_keys:
            continue
        if value in excluded_keys:
            continue

        key_path = f"{root_path}.{key}" if root_path else key
        result = key_path_dict.get(key, [])
        if isinstance(value, list):
            if len(value) > 0 and has_no_primitives_in_list(value):
                key_path_dict = build_list_reverse_order_index(
                    value, key_path_dict, key, root_path=key_path, excluded_keys=excluded_keys
                )
            else:
                result.append(key_path)
                key_path_dict[key] = result

        elif isinstance(value, dict):
            key_path_dict = build_dict_reverse_order_index(
                value, key_path_dict, root_path=key_path, excluded_keys=excluded_keys
            )
        elif is_primitive(value):
            if not (key == "path" and key_path.startswith("components")):
                result.append(key_path)
                key_path_dict[key] = result
        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")
    return key_path_dict
