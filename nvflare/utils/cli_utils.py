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
import pathlib
from typing import Optional, List

from pyhocon import ConfigTree, ConfigFactory as CF, HOCONConverter


def get_home_dir():
    from pathlib import Path
    return Path.home()


def get_hidden_nvflare_config_path() -> str:
    """
    Get the path for the hidden nvflare configuration file.

    Returns:
        str: The path to the hidden nvflare configuration file.
    """
    home_dir = get_home_dir()
    hidden_nvflare_dir = pathlib.Path(home_dir) / ".nvflare"

    try:
        hidden_nvflare_dir.mkdir(exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Error creating the hidden nvflare directory: {e}")

    hidden_nvflare_config_file = hidden_nvflare_dir / "config.conf"
    return str(hidden_nvflare_config_file)


def load_config(config_file_path) -> Optional[ConfigTree]:
    if os.path.isfile(config_file_path):
        return CF.parse_file(config_file_path)
    else:
        return None


def find_startup_kit_location():
    hidden_nvflare_config_file = get_hidden_nvflare_config_path()
    nvflare_config = load_config(hidden_nvflare_config_file)
    return nvflare_config.get_string("startup_kit.path", None) if nvflare_config else None


def create_startup_kit_config(nvflare_config: ConfigTree, startup_kit_dir: Optional[str] = None) -> ConfigTree:
    """
    Args:
        startup_kit_dir: specified startup kit location
        nvflare_config (ConfigTree): The existing nvflare configuration.

    Returns:
        ConfigTree: The merged configuration tree.
    """
    startup_kit_dir = get_startup_kit_dir(startup_kit_dir)
    if not startup_kit_dir or not os.path.isdir(startup_kit_dir):
        raise ValueError(f"startup_kit_dir '{startup_kit_dir}' must be a valid and non-empty path. "
                         f"use 'nvflare poc' command to 'prepare' if you are using POC mode. Or use"
                         f" 'nvflare config' to setup startup_kit_dir location if you are in production")

    conf_str = f"""
        startup_kit {{
            path = {startup_kit_dir}
        }}
    """
    conf: ConfigTree = CF.parse_string(conf_str)

    return conf.with_fallback(nvflare_config)


def get_startup_kit_dir(startup_kit_dir: Optional[str] = None) -> str:
    if startup_kit_dir:
        return startup_kit_dir

    # load from config file:
    startup_kit_dir = find_startup_kit_location()
    if startup_kit_dir is None:
        startup_kit_dir = os.getenv("NVFLARE_STARTUP_KIT_DIR")

    if startup_kit_dir is None or len(startup_kit_dir.strip()) == 0:
        raise ValueError("startup kit directory is not specified")
    else:
        return startup_kit_dir


def get_curr_dir():
    return os.path.curdir


def is_dir_empty(path: str):
    targe_dir = os.listdir(path)
    return len(targe_dir) == 0


def save_config(dst_config, dst_path, to_json=True):
    config_str = HOCONConverter.to_json(dst_config) if to_json else HOCONConverter.to_hocon(dst_config)
    with open(dst_path, "w") as outfile:
        outfile.write(f"{config_str}\n")


def save_startup_kit_config(startup_kit_dir: Optional[str] = None):
    hidden_nvflare_config_file = get_hidden_nvflare_config_path()
    conf = load_config(hidden_nvflare_config_file)
    nvflare_config = CF.parse_string("{}") if not conf else conf
    nvflare_config = create_startup_kit_config(nvflare_config, startup_kit_dir)
    save_config(nvflare_config, hidden_nvflare_config_file, to_json=False)


def find_in_list(arr: List, item) -> bool:
    if arr is None:
        return False

    found = False
    for a in arr:
        if a.__eq__(item):
            return True

    return found


def append_if_not_in_list(arr: List, item) -> List:
    if item is None:
        return arr

    if arr is None:
        arr = []

    if not find_in_list(arr, item):
        arr.append(item)

    return arr







