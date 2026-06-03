# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.lighter import utils

CC_CONFIG_FILE_KEYS = ("file", "path", "config_file")


def get_cc_config_file(cc_config_ref):
    if isinstance(cc_config_ref, str):
        return cc_config_ref

    if isinstance(cc_config_ref, dict):
        for key in CC_CONFIG_FILE_KEYS:
            config_file = cc_config_ref.get(key)
            if config_file:
                if not isinstance(config_file, str):
                    raise TypeError(f"cc_config.{key} must be str but got {type(config_file)}")
                return config_file
        return None

    raise TypeError(f"cc_config must be str or dict but got {type(cc_config_ref)}")


def load_cc_config(cc_config_ref):
    if isinstance(cc_config_ref, str):
        return utils.load_yaml(cc_config_ref)

    if not isinstance(cc_config_ref, dict):
        raise TypeError(f"cc_config must be str or dict but got {type(cc_config_ref)}")

    config_file = get_cc_config_file(cc_config_ref)
    if config_file:
        cc_config = utils.load_yaml(config_file) or {}
        if not isinstance(cc_config, dict):
            raise TypeError(f"cc_config file {config_file} must load to dict but got {type(cc_config)}")
    else:
        cc_config = {}

    inline_config = {k: v for k, v in cc_config_ref.items() if k not in CC_CONFIG_FILE_KEYS}
    cc_config.update(inline_config)
    return cc_config
