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
from typing import Any
from typing import Callable, Dict, List, Optional, Type

from pyhocon import ConfigFactory as CF, ConfigTree

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.utils.class_utils import ModuleScanner


def convert_class_names_to_paths(class_names: List[str]):
    result = []
    module_scanner = ModuleScanner(["nvflare"], [])
    for clazz in class_names:
        module_name = module_scanner.get_module_name(clazz)
        if module_name is None:
            raise ConfigError('Cannot find component class "{}"'.format(clazz))
        class_path = f"{module_name}.{clazz}"
        result.append(class_path)
        print(clazz, class_path)

    return result


def build_reverse_order_index(config_file_path: str) -> (Dict[str, List[str]], Dict[str, Any]):
    try:
        config: ConfigTree = CF.parse_file(config_file_path)
    except Exception as e:
        raise RuntimeError(f"filed to parse file {config_file_path}:", e)

    components: list = config.get("components", None)
    excluded_list = [comp.get("id") for comp in components] if components else []
    excluded_list.extend(
        ["name", "id", "format_version", "tasks", "task_name", "train_task_name",
         "task_data_filters", "task_result_filters", "exchange_path"]
    )
    indices: Dict[str, List[str]] = build_dict_reverse_order_index(config, excluded_keys=excluded_list)
    return indices, config


def build_list_reverse_order_index(config_list: List,
                                   key_path_dict: Dict,
                                   key: str,
                                   root_path: str,
                                   excluded_keys: Optional[List[str]] = None) -> Dict:
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
                build_list_reverse_order_index(value, key_path_dict, key_with_index, root_path=key_path,
                                               excluded_keys=excluded_keys)
            else:
                result.append(key_path)
                key_path_dict[key_with_index] = result
        elif isinstance(value, dict):
            build_dict_reverse_order_index(value, key_path_dict, root_path=key_path, excluded_keys=excluded_keys)
        elif isinstance(value, (int, float, str, bool, Callable, Type)):
            if not (key == "path" and key_path.startswith("components")):
                result.append(key_path)
                key_path_dict[key_with_index] = result
        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")
    return key_path_dict


def build_dict_reverse_order_index(
        config_dict: Dict,
        key_path_dict: Optional[Dict] = None,
        root_path: str = "",
        excluded_keys: List[str] = None
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
            if len(value) > 0:
                key_path_dict = build_list_reverse_order_index(value, key_path_dict, key, root_path=key_path,
                                                               excluded_keys=excluded_keys)
            else:
                result.append(key_path)
                key_path_dict[key] = result

        elif isinstance(value, dict):
            key_path_dict = build_dict_reverse_order_index(value, key_path_dict, root_path=key_path,
                                                           excluded_keys=excluded_keys)
        elif isinstance(value, (int, float, str, bool, Callable, Type)):
            if not (key == "path" and key_path.startswith("components")):
                result.append(key_path)
                key_path_dict[key] = result
        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")
    return key_path_dict
