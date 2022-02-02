# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import json


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


def save_to_json(data, path, sort_keys=False, indent=None):
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=sort_keys, indent=indent)
