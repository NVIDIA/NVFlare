# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import collections

from .validation_utils import check_object_type


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def update_configs_with_envs(configs, env):
    for k, v in configs.items():
        if isinstance(v, list):
            length = len(v)
            for i in range(length):
                if isinstance(v[i], dict):
                    configs[k][i] = update_configs_with_envs(v[i], env)
        elif isinstance(v, dict):
            configs[k] = update_configs_with_envs(v, env)
        elif isinstance(v, str):
            configs[k] = v.format(**env)
    return configs


def merge_dict(dict1, dict2):
    return {**dict1, **dict2}


def extract_first_level_primitive(d):
    result = {}
    for k, v in d.items():
        if type(v) in (int, float, bool, str):
            result[k] = v
    return result


def augment(to_dict: dict, from_dict: dict, from_override_to=False, append_list="components") -> str:
    """Augments the to_dict with the content from the from_dict.

        - Items in from_dict but not in to_dict are added to the to_dict
        - Items in both from_dict and to_dict must be ether dicts or list of dicts,
          and augment will be done on these items recursively
        - Non-dict/list items in both from_dict and to_dict are considered conflicts.

    Args:
        to_dict: the dict to be augmented
        from_dict: content to augment the to_dict
        from_override_to: content in from_dict overrides content in to_dict when conflict happens
        append_list: str or list of str: item keys for list to be appended

    Returns:
        An error message if any; empty str if success.

    .. note::

       The content of the to_dict is updated

    """
    check_object_type("to_dict", to_dict, dict)
    check_object_type("from_dict", from_dict, dict)

    if isinstance(append_list, str):
        append_list = [append_list]
    elif not isinstance(append_list, list):
        return f"append_list must be str or list but got {type(append_list)}"

    for k, fv in from_dict.items():
        if k not in to_dict:
            to_dict[k] = fv
            continue

        tv = to_dict[k]
        if isinstance(fv, dict):
            if not isinstance(tv, dict):
                return f"type conflict in element '{k}': dict in from_dict but {type(tv)} in to_dict"
            err = augment(tv, fv)
            if err:
                return err
            continue

        if isinstance(fv, list):
            if not isinstance(tv, list):
                return f"type conflict in element '{k}': list in from_dict but {type(tv)} in to_dict"

            if k in append_list:
                # items in "from_dict" are appended to "to_dict"
                tv.extend(fv)
                continue

            if len(fv) != len(tv):
                return f"list length conflict in element '{k}': {len(fv)} in from_dict but {len(tv)} in to_dict"

            for i in range(len(fv)):
                # we only support list of dicts!
                fvi = fv[i]
                tvi = tv[i]
                if not isinstance(fvi, dict):
                    return f"invalid list item {i} in element '{k}' in from_dict: must be dict but got {type(fvi)}"

                if not isinstance(tvi, dict):
                    return f"invalid list item {i} in element '{k}' in to_dict: must be dict but got {type(tvi)}"

                err = augment(tv[i], fv[i])
                if err:
                    return err
            continue

        if type(fv) != type(tv):
            return f"type conflict in element '{k}': {type(fv)} in from_dict but {type(tv)} in to_dict"

        if from_override_to:
            to_dict[k] = fv

    return ""


def _update_component_dict(comp_list: list, target: dict) -> str:
    for c in comp_list:
        check_object_type("element in comp_list", c, dict)
        cid = c.get("id", None)
        if not cid:
            return "missing 'id' from a component"
        target[cid] = c
    return ""


def update_components(target_dict: dict, from_dict: dict) -> str:
    """update components in target_dict with components from the from_dict.
    If a component with the same ID exists in both target_dict and from_dict, the component in from_dict
    will replace the one in target_dict.
    If a component only exists in from_dict, it will be added to the component list of target_dict.
    Args:
        target_dict: the dict to be updated
        from_dict: the dict that will be used to update the target_dict
    Returns:
    """
    key_components = "components"

    from_comp_list = from_dict.get(key_components, None)
    if not from_comp_list:
        # no components to update
        return ""

    check_object_type("from_comp_list", from_comp_list, list)

    target_comp_list = target_dict.get(key_components, None)
    if not target_comp_list:
        target_dict[key_components] = from_comp_list
        return ""

    check_object_type("target_comp_list", target_comp_list, list)

    from_comp_dict = {}
    err = _update_component_dict(from_comp_list, from_comp_dict)
    if err:
        return f"error in from_dict: {err}"

    target_comp_dict = {}
    err = _update_component_dict(target_comp_list, target_comp_dict)
    if err:
        return f"error in target_dict: {err}"

    # determine components in both
    dups = []
    for cid in target_comp_dict.keys():
        if cid in from_comp_dict:
            dups.append(cid)

    for cid in dups:
        # remove from target_comp_dict
        target_comp_dict.pop(cid)

    new_target_comp_list = list(target_comp_dict.values())
    new_target_comp_list.extend(from_comp_list)

    target_dict[key_components] = new_target_comp_list
    return ""
