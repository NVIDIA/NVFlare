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
import dataclasses
import inspect
import os.path
from typing import Any, Dict, List, Optional, Tuple, Union

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree

from nvflare.fuel.utils.config import Config, ConfigFormat
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.fuel.utils.import_utils import optional_import


@dataclasses.dataclass
class KeyIndex:
    key: str
    value: Union[None, Any, ConfigTree] = None
    parent_key: Optional["KeyIndex"] = None
    index: Optional[int] = None
    component_name: Optional[str] = None


def build_reverse_order_index(input_config_file_path: str) -> Tuple[str, ConfigTree, List[str], Dict]:
    # use pyhocon to load config
    config_dir = os.path.dirname(input_config_file_path)
    config_dir = None if not config_dir else config_dir
    config_file_path = os.path.basename(input_config_file_path)
    config, config_file_path = load_pyhocon_conf(config_file_path, config_dir)

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
            "json_encoder_path",
        ]
    )
    key_indices = build_dict_reverse_order_index(config, excluded_keys=[])
    key_indices = add_default_values(excluded_list, key_indices)
    populate_key_component_names(key_indices)

    return config_file_path, config, excluded_list, key_indices


def load_pyhocon_conf(config_file_path: str, search_dir: Optional[str]) -> Tuple[ConfigTree, str]:
    """Loads config using pyhocon."""
    try:
        temp_conf: Config = ConfigFactory.load_config(config_file_path, [search_dir])
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

    for index, value in enumerate(config_list):
        elmt_key = f"{key}[{index}]"
        key_index = KeyIndex(key=elmt_key, value=value, parent_key=root_index, index=index)

        if value is None:
            continue

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
                add_to_indices(elmt_key, key_index, key_indices)
                if key == "name":
                    key_index.component_name = value
        elif isinstance(value, ConfigTree):
            key_indices = build_dict_reverse_order_index(
                config=value, excluded_keys=excluded_keys, root_index=key_index, key_indices=key_indices
            )
        elif is_primitive(value):
            if key == "path":
                has_dot = value.find(".") > 0
                if has_dot:
                    # we assume the path's pass value is class name
                    # there are cases, this maybe not.
                    # user may have to modify configuration manually in those cases
                    last_dot_index = value.rindex(".")
                    class_name = value[last_dot_index + 1 :]
                    key_index.component_name = class_name
            elif key == "name":
                # there are cases, where name is not component
                # user may have to modify configuration manually in those cases
                key_index.component_name = value

            add_to_indices(elmt_key, key_index, key_indices)
        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")
    return key_indices


def is_primitive(value):
    return (
        isinstance(value, int)
        or isinstance(value, float)
        or isinstance(value, str)
        or isinstance(value, bool)
        or value is None
    )


def has_none_primitives_in_list(values: List):
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
    root_index = KeyIndex(key="", value=config, parent_key=None, index=None) if root_index is None else root_index

    for key, value in config.items():
        if key in excluded_keys:
            continue
        if value in excluded_keys:
            continue

        key_index = KeyIndex(key=key, value=value, parent_key=root_index, index=None)
        if isinstance(value, list):
            if len(value) > 0 and has_none_primitives_in_list(value):
                key_indices = build_list_reverse_order_index(
                    config_list=value,
                    key=key,
                    excluded_keys=excluded_keys,
                    root_index=key_index,
                    key_indices=key_indices,
                )
            else:
                add_to_indices(key, key_index, key_indices)

        elif isinstance(value, ConfigTree):
            key_indices = build_dict_reverse_order_index(
                config=value, excluded_keys=excluded_keys, root_index=key_index, key_indices=key_indices
            )

        elif is_primitive(value):
            parent_key = key_index.parent_key
            if key == "path":
                has_dot = value.find(".") > 0
                if has_dot:
                    # we assume the path's pass value is class name
                    # there are cases, this maybe not.
                    # user may have to modify configuration manually in those cases
                    last_dot_index = value.rindex(".")
                    class_name = value[last_dot_index + 1 :]
                    key_index.component_name = class_name
                    parent_key.component_name = key_index.component_name if parent_key.index is not None else None
            elif key == "name":
                # what if the name is not component ?
                key_index.component_name = value
                parent_key.component_name = key_index.component_name if parent_key.index else None

            add_to_indices(key, key_index, key_indices)

        else:
            raise RuntimeError(f"Unhandled data type: {type(value)}")
    return key_indices


def add_to_indices(key, key_index, key_indices):
    indices = key_indices.get(key, [])
    if key_index not in indices:
        indices.append(key_index)
    key_indices[key] = indices


def add_class_defaults_to_key(excluded_keys, key_index, key_indices, results):
    if key_index is None or key_index.key != "path":
        return

    parent_key: KeyIndex = key_index.parent_key
    value = key_index.value
    has_dot = value.find(".") > 0
    if not has_dot:
        return
    # we assume the path's pass value is class name
    # there are cases, this maybe not.
    # user may have to modify configuration manually in those cases
    last_dot_index = value.rindex(".")
    class_path = value[:last_dot_index]
    class_name = value[last_dot_index + 1 :]
    module, import_flag = optional_import(module=class_path, name=class_name)
    if import_flag:
        params = inspect.signature(module.__init__).parameters
        args_config = None
        if parent_key and parent_key.value and isinstance(parent_key.value, ConfigTree):
            args_config = parent_key.value.get("args", None)
        for v in params.values():
            if (
                v.name != "self"
                and v.default is not None
                and v.name not in excluded_keys
                and v.default not in excluded_keys
            ):
                name_key = None
                arg_key = KeyIndex(
                    key="args", value=args_config, parent_key=parent_key, component_name=key_index.component_name
                )
                if isinstance(v.default, str):
                    if len(v.default) > 0:
                        name_key = KeyIndex(
                            key=v.name,
                            value=v.default,
                            parent_key=arg_key,
                            component_name=key_index.component_name,
                        )
                elif type(v.default) != type:
                    name_key = KeyIndex(
                        key=v.name,
                        value=v.default,
                        parent_key=arg_key,
                        component_name=key_index.component_name,
                    )

                if name_key:

                    name_indices: List[KeyIndex] = key_indices.get(v.name, [])
                    has_one = any(
                        k.parent_key is not None
                        and k.parent_key.key == "args"
                        and k.parent_key.parent_key.key == key_index.parent_key.key
                        for k in name_indices
                    )
                    if not has_one:
                        name_indices.append(name_key)
                        results[v.name] = name_indices


def update_index_comp_name(key_index: KeyIndex):
    parent_key = key_index.parent_key
    if parent_key is None:
        return key_index
    if not isinstance(key_index, KeyIndex):
        return key_index
    if parent_key.key == "args":
        grand_parent = parent_key.parent_key
        key_index.component_name = grand_parent.component_name
        update_index_comp_name(parent_key)

    return key_index


def add_default_values(excluded_keys, key_indices: Dict):
    results = key_indices.copy()

    for key, key_index_list in key_indices.items():
        for key_index in key_index_list:
            if key_index:
                add_class_defaults_to_key(excluded_keys, key_index, key_indices, results)
    return results


def populate_key_component_names(key_indices: Dict):
    results = {}
    for key, key_index_list in key_indices.items():
        for key_index in key_index_list:
            if key_index:
                key_index = update_index_comp_name(key_index)
                key_index.component_name = "" if key_index.component_name is None else key_index.component_name
            results[key] = key_index
    return results
