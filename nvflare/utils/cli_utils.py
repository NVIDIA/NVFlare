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
from pathlib import Path
from typing import List, Optional

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree, HOCONConverter

from nvflare.fuel.utils.config import ConfigFormat


def get_home_dir() -> Path:
    return Path.home()


def get_hidden_nvflare_config_path(hidden_nvflare_dir: str) -> str:
    """
    Get the path for the hidden nvflare configuration file.
    Args:
        hidden_nvflare_dir: ~/.nvflare directory
    Returns:
        str: The path to the hidden nvflare configuration file.
    """
    hidden_nvflare_config_file = os.path.join(hidden_nvflare_dir, "config.conf")
    return str(hidden_nvflare_config_file)


def create_hidden_nvflare_dir():
    hidden_nvflare_dir = get_hidden_nvflare_dir()
    if not hidden_nvflare_dir.exists():
        try:
            hidden_nvflare_dir.mkdir(exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Error creating the hidden nvflare directory: {e}")

    return hidden_nvflare_dir


def get_hidden_nvflare_dir() -> pathlib.Path:
    home_dir = get_home_dir()
    hidden_nvflare_dir = pathlib.Path(home_dir) / ".nvflare"
    return hidden_nvflare_dir


def load_config(config_file_path) -> Optional[ConfigTree]:
    if os.path.isfile(config_file_path):
        return CF.parse_file(config_file_path)
    else:
        return None


def find_startup_kit_location() -> str:
    nvflare_config = load_hidden_config()
    return nvflare_config.get_string("startup_kit.path", None) if nvflare_config else None


def load_hidden_config() -> ConfigTree:
    hidden_dir = create_hidden_nvflare_dir()
    hidden_nvflare_config_file = get_hidden_nvflare_config_path(str(hidden_dir))
    nvflare_config = load_config(hidden_nvflare_config_file)
    return nvflare_config


def create_startup_kit_config(nvflare_config: ConfigTree, startup_kit_dir: Optional[str] = None) -> ConfigTree:
    """
    Args:
        startup_kit_dir: specified startup kit location
        nvflare_config (ConfigTree): The existing nvflare configuration.

    Returns:
        ConfigTree: The merged configuration tree.
    """
    startup_kit_dir = get_startup_kit_dir(startup_kit_dir)
    conf_str = f"""
        startup_kit {{
            path = {startup_kit_dir}
        }}
    """
    conf: ConfigTree = CF.parse_string(conf_str)

    return conf.with_fallback(nvflare_config)


def check_dir(dir_path: str):
    if not dir_path or not os.path.isdir(dir_path):
        raise ValueError(f"directory {dir_path} doesn't exists")


def get_startup_kit_dir(startup_kit_dir: Optional[str] = None) -> str:

    if not startup_kit_dir:
        # load from config file:
        startup_kit_dir = find_startup_kit_location()
        if startup_kit_dir is None:
            startup_kit_dir = os.getenv("NVFLARE_STARTUP_KIT_DIR")

        if startup_kit_dir is None or len(startup_kit_dir.strip()) == 0:
            raise ValueError("startup kit directory is not specified")

    check_startup_dir(startup_kit_dir)
    return startup_kit_dir


def check_startup_dir(startup_kit_dir):
    if not startup_kit_dir or not os.path.isdir(startup_kit_dir):
        raise ValueError(
            f"startup_kit_dir '{startup_kit_dir}' must be a valid and non-empty path. "
            f"use 'nvflare poc' command to 'prepare' if you are using POC mode. Or use"
            f" 'nvflare config' to setup startup_kit_dir location if you are in production"
        )


def find_job_template_location(job_template_dir: Optional[str] = None):
    def check_job_template_dir(job_temp_dir: str):
        if job_temp_dir:
            if os.path.isdir(job_temp_dir):
                return job_temp_dir
            else:
                raise ValueError(f"Invalid job template directory {job_temp_dir}")

    template_dir = check_job_template_dir(job_template_dir)
    if template_dir:
        return template_dir

    nvflare_config = load_hidden_config()
    job_template_dir = nvflare_config.get_string("job_template.path", None) if nvflare_config else None
    job_template_dir = check_job_template_dir(job_template_dir)

    nvflare_home = os.environ.get("NVFLARE_HOME", None)
    if nvflare_home:
        job_template_dir = os.path.join(nvflare_home, "integration", "job_templates")

    job_template_dir = check_job_template_dir(job_template_dir)

    if not job_template_dir:
        raise ValueError("required job_template directory is not specified. please check ~/.nvflare/config.conf")

    return job_template_dir


def get_curr_dir():
    return os.path.curdir


def is_dir_empty(path: str):
    targe_dir = os.listdir(path)
    return len(targe_dir) == 0


def save_config(dst_config, dst_path, to_json=False):
    fmt = ConfigFormat.JSON if to_json else ConfigFormat.PYHOCON
    ext = ConfigFormat.extensions(fmt)[0]
    if dst_path.endswith(ext):
        dst_config_path = dst_path
    else:
        filename = f"{os.path.basename(dst_path).split('.')[0]}{ext}"
        dst_config_path = os.path.join(os.path.dirname(dst_path), filename)

    config_str = HOCONConverter.to_json(dst_config) if to_json else HOCONConverter.to_hocon(dst_config)
    with open(dst_config_path, "w") as outfile:
        outfile.write(f"{config_str}\n")


def save_startup_kit_config(startup_kit_dir: Optional[str] = None):
    hidden_nvflare_config_file = get_hidden_nvflare_config_path(str(create_hidden_nvflare_dir()))
    conf = load_hidden_config()
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
