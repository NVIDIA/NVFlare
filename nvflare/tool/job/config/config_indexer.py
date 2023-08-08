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
from typing import Any, Dict, List

from nvflare.fuel.utils.config import Config
from nvflare.fuel.utils.config_factory import ConfigFactory
from pyhocon import ConfigFactory as CF, ConfigTree


def build_reverse_order_index(config_file_path: str) -> (Dict[str, List[str]], Dict[str, Any]):
    # config: Config = ConfigFactory.load_config(config_file_path)
    config: ConfigTree = CF.parse_file(config_file_path)
    components: list = config.get("components", None)
    component_ids = [comp.get("id") for comp in components] if components else []

    # config_dict: Dict[str, Any] = config.to_dict()
    indices: Dict[str, List[str]] = build_dict_reverse_order_index(config, excluded_keys = component_ids)
    return indices, config


def expand_indices(indices: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Expand the input dictionary of indices into a new dictionary with extended key paths.
    Example:
    input =  {"x22": ["x.x2.x22", "x.z[0].x2.x22"]}
    Output:
    {
        "x22": ["x.x2.x22", "x.z[0].x2.x22"],
        "x2.x22": ["x.x2.x22", "x.z[0].x2.x22"],
        "x.x2.x22": ["x.x2.x22"],
        "z[0].x2.x22": ["x.z[0].x2.x22"]
    }
    """
    new_key_paths = {}
    for key, items in indices.items():
        for sub_key in items:
            tokens = sub_key.split(".")
            for i in range(len(tokens) - 1):
                key_path = ".".join(tokens[i:])
                results = new_key_paths.get(key_path, [])
                results.append(sub_key)
                new_key_paths[key_path] = results
    return {**indices, **new_key_paths}


from typing import Callable, Dict, List, Optional, Type


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
            build_list_reverse_order_index(value, key_path_dict, key_with_index, root_path=key_path, excluded_keys = excluded_keys)
        elif isinstance(value, dict):
            build_dict_reverse_order_index(value, key_path_dict, root_path=key_path, excluded_keys = excluded_keys)
        elif isinstance(value, (int, float, str, bool, Callable, Type)):
            result.append(key_path)
            key_path_dict[key_with_index] = result
        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")
    return key_path_dict


def build_dict_reverse_order_index(
        config_dict: Dict,
        key_path_dict: Optional[Dict] = None,
        root_path: str = "",
        excluded_keys : List[str] = None
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

        key_path = f"{root_path}.{key}" if root_path else key
        result = key_path_dict.get(key, [])
        if isinstance(value, list):
            key_path_dict = build_list_reverse_order_index(value, key_path_dict, key, root_path=key_path, excluded_keys = excluded_keys)
        elif isinstance(value, dict):
            key_path_dict = build_dict_reverse_order_index(value, key_path_dict, root_path=key_path, excluded_keys = excluded_keys)
        elif isinstance(value, (int, float, str, bool, Callable, Type)):
            result.append(key_path)
            key_path_dict[key] = result
        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")
    return key_path_dict
