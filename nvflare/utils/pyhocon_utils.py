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

from typing import Dict

from pyhocon import ConfigTree


def _convert_conf_item(conf_item):
    result = {}
    if isinstance(conf_item, ConfigTree):
        if len(conf_item) > 0:
            for key, item in conf_item.items():
                new_key = key.strip('"')  # for dotted keys enclosed with "" to not be interpreted as nested key
                new_value = _convert_conf_item(item)
                result[new_key] = new_value
    elif isinstance(conf_item, list):
        if len(conf_item) > 0:
            result = [_convert_conf_item(item) for item in conf_item]
        else:
            result = []
    elif conf_item is True:
        return True
    elif conf_item is False:
        return False
    else:
        return conf_item

    return result


def to_dict(config: ConfigTree) -> Dict:
    """Convert HOCON input into a Dict"""
    return _convert_conf_item(config)
