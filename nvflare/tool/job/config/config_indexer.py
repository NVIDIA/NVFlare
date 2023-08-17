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
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree

from nvflare.fuel.utils.config import Config, ConfigFormat
from nvflare.fuel.utils.config_factory import ConfigFactory


@dataclasses.dataclass
class KeyIndex:
    key: str
    value: Union[None, Any, ConfigTree] = None
    parent_key: Optional["KeyIndex"] = None
    children: List["KeyIndex"] = None
    index: Optional[int] = None
    component_name: Optional[str] = None


def build_reverse_order_index(config_file_path: str) -> Tuple:
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
            "submit_model_task_name",
            "validation_task_name",
            "validate_task_name",
            "task_data_filters",
            "task_result_filters",
            "exchange_path",
            "job_folder_name",
        ]
    )
    key_indices = build_dict_reverse_order_index(config, excluded_keys=[])
    return config_file_path, config, excluded_list, key_indices


def load_pyhocon_conf(config_file_path) -> Tuple[ConfigTree, str]:
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
    config_list: List,
    key: str,
    excluded_keys: Optional[List[str]],
    root_index: Optional[KeyIndex],
    key_indices: Optional[Dict],
) -> Dict:
    """
    Recursively build a reverse order index for a list.
    """
    if excluded_keys is None:
        excluded_keys = []
    if key_indices is None:
        key_indices = {}
    if root_index and root_index.children is None:
        root_index.children = []

    for index, value in enumerate(config_list):
        elmt_key = f"{key}[{index}]"
        key_index = KeyIndex(key=elmt_key, value=value, parent_key=root_index, index=index)
        root_index.children.append(key_index)

        if isinstance(value, list):
            if len(value) > 0:
                key_indices = build_list_reverse_order_index(
                    config_list=value,
                    key=elmt_key,
                    excluded_keys=excluded_keys,
                    root_index=key_index,
                    key_indices=key_indices,
                )
            else:
                key_indices[elmt_key] = key_index
                if key == "name":
                    key_index.component_name = value
        elif isinstance(value, ConfigTree):
            key_indices = build_dict_reverse_order_index(
                config=value, excluded_keys=excluded_keys, root_index=key_index, key_indices=key_indices
            )
        elif is_primitive(value):
            if key == "path":
                last_dot_index = value.rindex(".")
                class_name = value[last_dot_index + 1 :]
                key_index.component_name = class_name
            elif key == "name":
                key_index.component_name = value
            key_indices[elmt_key] = key_index
        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")
    return key_indices


def is_primitive(value):
    return isinstance(value, int) or isinstance(value, float) or isinstance(value, str) or isinstance(value, bool)


def has_no_primitives_in_list(values: List):
    return any(not is_primitive(x) for x in values)


def build_dict_reverse_order_index(
    config: ConfigTree,
    excluded_keys: List[str] = None,
    root_index: Optional[KeyIndex] = None,
    key_indices: Optional[Dict] = None,
) -> Dict:
    key_indices = {} if key_indices is None else key_indices
    if excluded_keys is None:
        excluded_keys = []

    root_index = KeyIndex(key="", value=None, parent_key=None, index=None) if root_index is None else root_index
    if root_index.children is None:
        root_index.children = []

    for key, value in config.items():
        if key in excluded_keys:
            continue
        if value in excluded_keys:
            continue

        key_index = KeyIndex(key=key, value=value, parent_key=root_index, index=None)
        root_index.children.append(key_index)

        if isinstance(value, list):
            if len(value) > 0 and has_no_primitives_in_list(value):
                key_indices = build_list_reverse_order_index(
                    config_list=value,
                    key=key,
                    excluded_keys=excluded_keys,
                    root_index=key_index,
                    key_indices=key_indices,
                )
            else:
                key_indices[key] = key_index

        elif isinstance(value, ConfigTree):
            key_indices = build_dict_reverse_order_index(
                config=value, excluded_keys=excluded_keys, root_index=key_index, key_indices=key_indices
            )
        elif is_primitive(value):
            parent_key = key_index.parent_key
            if key == "path":
                last_dot_index = value.rindex(".")
                class_name = value[last_dot_index + 1 :]
                key_index.component_name = class_name
                parent_key.component_name = key_index.component_name if parent_key.index is not None else None

            elif key == "name":
                key_index.component_name = value
                parent_key.component_name = key_index.component_name if parent_key.index else None
            key_indices[key] = key_index
        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")

    populate_key_component_names(key_indices)
    return key_indices


def update_index_comp_name(key_index: KeyIndex):
    parent_key = key_index.parent_key
    if parent_key is None:
        return key_index
    if not isinstance(key_index, KeyIndex):
        return key_index
    if parent_key.key == "args":
        grand_parent = parent_key.parent_key
        key_index.component_name = grand_parent.component_name
        return update_index_comp_name(parent_key)

    return key_index


def populate_key_component_names(key_indices: Dict):
    results = {}
    for key, key_index in key_indices.items():
        if key_index:
            key_index = update_index_comp_name(key_index)
            key_index.component_name = " " if key_index.component_name is None else key_index.component_name
        results[key] = key_index
    return results
